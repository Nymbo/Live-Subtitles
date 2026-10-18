"""Live speech engines: audio in, (committed words, pending words, phrase id) out.

An engine is driven by one thread (`live.LiveRecognizer` in the app, a simulated clock
in `bench/run.py`), so both run exactly this code. `process(audio)` takes the next
16 kHz mono samples, or None when no audio arrived for IDLE_POLL_S (Windows sends no
loopback packets while nothing plays). It returns a list of updates (empty when nothing
changed): the words committed since the last update, the words still in progress, and
the id of the phrase (utterance) they belong to.

This module holds the CPU engine (Nemotron through sherpa-onnx) and the shared input
leveller; the GPU engine (Parakeet-unified) is in `unified.py`.

Design notes from the bake-offs (Tears of Steel and the synthetic scenes, see README):
- No VAD gate. Silero dropped dialogue under music and cost Parakeet 18 WER points.
- Reset the Nemotron stream only at safe points. Nemotron punctuates and capitalises per
  utterance; one endless stream drifts into lowercase with no punctuation at all. But
  sherpa's endpoint rules reset on "no token for 0.8 s" and force a cut after N seconds,
  both of which land mid-speech and lose words (37% WER vs 28.7%). So we reset ourselves:
  after 2 s of audio with no new token, or, in a long unbroken stretch, at the first 1 s
  gap. A new stream is seeded with the last 1.5 s of audio so a word starting at the
  boundary keeps its onset.
- Emitted tokens are final (greedy RNN-T), so captions never flicker or rewrite.
"""

from __future__ import annotations

import logging
import re
import time
import numpy as np

from .models import LiveModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
IDLE_POLL_S = 0.2
STEP_SAMPLES = int(SAMPLE_RATE * 0.08)
PRIME_SAMPLES = int(SAMPLE_RATE * 1.5)   # audio re-fed into each new utterance's stream
FLUSH_SILENCE_S = 1.6       # zeros fed after audio stops, so the last chunk gets decoded
PENDING_FLUSH_S = 1.2       # commit a trailing word after this much decoded audio with no token
RESET_GAP_S = 2.0           # audio seconds without a token -> safe to start a new utterance
LONG_UTTERANCE_S = 40.0     # past this, accept a shorter gap...
LONG_RESET_GAP_S = 1.0      # ...of this many seconds

_TAG = re.compile(r"<[^>]*>")

# (newly committed words, words in progress, phrase id those words belong to)
Update = tuple[list[str], list[str], int]


class Leveler:
    """Slow automatic gain for the recogniser's input (never for what you hear).

    Film mixes put dialogue anywhere from -20 to -50 dBFS. Nemotron's features are not
    level-normalised, so quiet dialogue decodes worse. This brings speech-level audio
    toward TARGET_DB with at most MAX_GAIN_DB of boost, never boosts near-silence (so
    room tone and hiss stay quiet), and soft-limits peaks.
    """

    TARGET_DB = -23.0
    MAX_GAIN_DB = 24.0
    MIN_GAIN_DB = -6.0
    FLOOR_DB = -62.0            # frames below this hold the current gain
    FRAME = 320                 # 20 ms
    ATTACK_S, RELEASE_S = 0.3, 2.5   # level estimate: rises fast, falls slowly

    def __init__(self):
        self.level_db = self.TARGET_DB
        self.gain_db = 0.0
        self._rest = np.zeros(0, np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._rest, audio]) if self._rest.size else audio
        n = x.size // self.FRAME * self.FRAME
        self._rest = x[n:].copy()
        if not n:
            return np.zeros(0, np.float32)
        frames = x[:n].reshape(-1, self.FRAME)
        out = np.empty_like(frames)
        dt = self.FRAME / SAMPLE_RATE
        a_up, a_down = 1 - np.exp(-dt / self.ATTACK_S), 1 - np.exp(-dt / self.RELEASE_S)
        for i, f in enumerate(frames):
            db = 10 * np.log10(float(np.mean(f * f)) + 1e-12)
            if db > self.FLOOR_DB:
                self.level_db += (db - self.level_db) * (a_up if db > self.level_db else a_down)
                want = min(self.MAX_GAIN_DB, max(self.MIN_GAIN_DB, self.TARGET_DB - self.level_db))
                self.gain_db += (want - self.gain_db) * 0.25
            g = 10 ** (self.gain_db / 20)
            out[i] = np.tanh(f * g * 1.1) / 1.1 if g > 1 else f * g
        return out.reshape(-1)


class NemotronEngine:
    """NVIDIA's cache-aware streaming Nemotron through sherpa-onnx, on the CPU."""

    def __init__(self, model: LiveModel, threads: int = 4, leveler: bool = True):
        self.model = model
        self.threads = max(1, threads)
        self.leveler = Leveler() if leveler else None
        self.name = model.name

    def load(self) -> None:
        import sherpa_onnx

        d = self.model.path
        t = time.perf_counter()
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            num_threads=self.threads,
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
        )
        logger.info("Loaded %s in %.1fs (%d threads)", self.model.name, time.perf_counter() - t, self.threads)
        self.stream = self._new_stream()
        self.tail = np.zeros(0, dtype=np.float32)    # the last PRIME_S of audio, to seed a new stream
        self.seen = 0                 # tokens of the current utterance already handled
        self.pending = ""             # the word being spelled out, shown but not committed
        self.fed = 0.0                # audio seconds fed so far
        self.utt_start = 0.0          # audio time the current utterance started
        self.last_token_audio = 0.0   # audio time at which the last new token appeared
        self.idle_fed = FLUSH_SILENCE_S
        self.silence = np.zeros(int(SAMPLE_RATE * IDLE_POLL_S), dtype=np.float32)
        self.lag = self.model.latency_ms / 1000
        self.phrase = 0               # counts utterances (stream resets)

    def _new_stream(self, prime: np.ndarray | None = None):
        stream = self.rec.create_stream()
        if self.model.language:
            stream.set_option("language", self.model.language)
        if prime is not None and prime.size:
            stream.accept_waveform(SAMPLE_RATE, prime)
        return stream

    def process(self, audio: np.ndarray | None) -> list[Update]:
        """Feed audio (None: nothing arrived for IDLE_POLL_S). Returns the caption updates."""
        if audio is None:
            if self.idle_fed >= FLUSH_SILENCE_S:
                updates = []
                if self.pending:   # audio stopped mid-word; the flush decoded all it will
                    word, self.pending = self.pending, ""
                    updates.append(([word], [], self.phrase))
                return updates
            audio = self.silence
            self.idle_fed += IDLE_POLL_S
        else:
            self.idle_fed = 0.0
        if self.leveler is not None:
            audio = self.leveler.process(audio)
        rec = self.rec
        updates: list[Update] = []
        committed: list[str] = []
        before = self.pending
        # Step in small pieces even when audio arrived in a burst (after a CPU stall,
        # say), so token gaps, and so the reset points, are measured at fine grain.
        for i in range(0, audio.size, STEP_SAMPLES):
            piece = audio[i:i + STEP_SAMPLES]
            self.stream.accept_waveform(SAMPLE_RATE, piece)
            self.fed += piece.size / SAMPLE_RATE
            self.tail = np.concatenate([self.tail, piece])[-PRIME_SAMPLES:]
            while rec.is_ready(self.stream):
                rec.decode_stream(self.stream)

            tokens = rec.get_result_all(self.stream).tokens
            new = tokens[self.seen:]
            self.seen = len(tokens)
            for tok in new:
                tok = _TAG.sub("", tok)
                if not tok:
                    continue
                if tok.startswith(" "):
                    if self.pending:
                        committed.append(self.pending)
                    self.pending = tok.strip()
                else:
                    self.pending += tok
            if new:
                self.last_token_audio = self.fed

            # Safe utterance boundary: nothing new for a while, so nothing is mid-word.
            # `fed` runs ahead of what has been decoded by up to a chunk, so the gap is
            # padded by the chunk length to mean "seconds of *decoded* audio with no token".
            gap = self.fed - max(self.last_token_audio, self.utt_start)
            long_utt = self.fed - self.utt_start > LONG_UTTERANCE_S
            if gap >= RESET_GAP_S + self.lag or (long_utt and gap >= LONG_RESET_GAP_S + self.lag):
                if self.pending:
                    committed.append(self.pending)
                    self.pending = ""
                logger.debug("New utterance at %.2fs (gap %.2fs, %d tokens)", self.fed, gap, self.seen)
                # A fresh stream seeded with the recent tail, rather than reset(): a word
                # may be starting in audio not yet decoded, and reset() would cut it off
                # from its own onset. The tail produced no tokens (that is the gap), so
                # re-decoding it cannot repeat a word.
                self.stream = self._new_stream(self.tail)
                self.seen = 0
                self.utt_start = self.fed
                if committed:
                    updates.append((committed, [], self.phrase))
                    committed, before = [], ""
                self.phrase += 1

        if self.pending and self.fed - self.last_token_audio > PENDING_FLUSH_S + self.lag:
            committed.append(self.pending)
            self.pending = ""
        if committed or self.pending != before:
            updates.append((committed, [self.pending] if self.pending else [], self.phrase))
        return updates
