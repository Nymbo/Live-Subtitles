"""Live recogniser: streams captured audio through Nemotron (sherpa-onnx) and emits words.

Design notes from the model bake-off (Tears of Steel, see README):
- No VAD gate. Silero dropped dialogue under music and cost Parakeet 18 WER points.
- Reset the stream only at safe points. Nemotron punctuates and capitalises per
  utterance; one endless stream drifts into lowercase with no punctuation at all. But
  sherpa's endpoint rules reset on "no token for 0.8 s" (barely above one 560 ms chunk)
  and force a cut after N seconds, both of which land mid-speech and lose words (37% WER
  vs 28.7%). So we reset ourselves: after 2 s of audio with no new token, or, in a long
  unbroken stretch, at the first 1 s gap. That kept 27.5% WER *and* the punctuation.
  RNNTOnlineStream reset keeps buffered audio; only the model state starts fresh.
- Emitted tokens are final (greedy RNN-T), so captions never flicker or rewrite.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Callable

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

WordsCallback = Callable[[list[str], str], None]   # (newly committed words, word in progress)


class LiveRecognizer(threading.Thread):
    def __init__(self, model: LiveModel, threads: int, chunks: "queue.Queue[np.ndarray]",
                 on_words: WordsCallback, on_ready: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None):
        super().__init__(name="live-recognizer", daemon=True)
        self.model = model
        self.threads = max(1, threads)
        self.chunks = chunks
        self.on_words = on_words
        self.on_ready = on_ready
        self.on_error = on_error
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("Live recogniser crashed")
            if self.on_error:
                self.on_error(str(exc))

    def _create(self):
        import sherpa_onnx

        d = self.model.path
        t = time.perf_counter()
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            num_threads=self.threads,
            enable_endpoint_detection=False,
            decoding_method="greedy_search",
        )
        logger.info("Loaded %s in %.1fs (%d threads)", self.model.name, time.perf_counter() - t, self.threads)
        return rec

    def _new_stream(self, rec, prime: np.ndarray | None = None):
        stream = rec.create_stream()
        if self.model.language:
            stream.set_option("language", self.model.language)
        if prime is not None and prime.size:
            stream.accept_waveform(SAMPLE_RATE, prime)
        return stream

    def _run(self) -> None:
        rec = self._create()
        stream = self._new_stream(rec)
        tail = np.zeros(0, dtype=np.float32)    # the last PRIME_S of audio, to seed a new stream
        # Drop audio that queued up while the model loaded; captions should start "now".
        while not self.chunks.empty():
            self.chunks.get_nowait()
        if self.on_ready:
            self.on_ready()

        seen = 0                 # tokens of the current utterance already handled
        pending = ""             # the word being spelled out, shown but not committed
        fed = 0.0                # audio seconds fed so far
        utt_start = 0.0          # audio time the current utterance started
        last_token_audio = 0.0   # audio time at which the last new token appeared
        idle_fed = FLUSH_SILENCE_S
        silence = np.zeros(int(SAMPLE_RATE * IDLE_POLL_S), dtype=np.float32)
        lag = self.model.latency_ms / 1000

        while not self._stop.is_set():
            try:
                parts = [self.chunks.get(timeout=IDLE_POLL_S)]
                while True:   # drain whatever else is waiting so decoding catches up in one go
                    try:
                        parts.append(self.chunks.get_nowait())
                    except queue.Empty:
                        break
                audio = np.concatenate(parts)
                idle_fed = 0.0
            except queue.Empty:
                if idle_fed >= FLUSH_SILENCE_S:
                    if pending:   # audio stopped mid-word; the flush decoded all it will
                        self.on_words([pending], "")
                        pending = ""
                    continue
                audio = silence
                idle_fed += IDLE_POLL_S
            committed: list[str] = []
            before = pending
            # Step in small pieces even when audio arrived in a burst (after a CPU stall,
            # say), so token gaps, and so the reset points, are measured at fine grain.
            for i in range(0, audio.size, STEP_SAMPLES):
                piece = audio[i:i + STEP_SAMPLES]
                stream.accept_waveform(SAMPLE_RATE, piece)
                fed += piece.size / SAMPLE_RATE
                tail = np.concatenate([tail, piece])[-PRIME_SAMPLES:]
                while rec.is_ready(stream):
                    rec.decode_stream(stream)

                tokens = rec.get_result_all(stream).tokens
                new = tokens[seen:]
                seen = len(tokens)
                for tok in new:
                    tok = _TAG.sub("", tok)
                    if not tok:
                        continue
                    if tok.startswith(" "):
                        if pending:
                            committed.append(pending)
                        pending = tok.strip()
                    else:
                        pending += tok
                if new:
                    last_token_audio = fed

                # Safe utterance boundary: nothing new for a while, so nothing is mid-word.
                # This also refreshes the state every couple of seconds through music and
                # silence, so the next line starts clean (that is what keeps its first words).
                # `fed` runs ahead of what has been decoded by up to a chunk, so the gap is
                # padded by the chunk length to mean "seconds of *decoded* audio with no token".
                gap = fed - max(last_token_audio, utt_start)
                long_utt = fed - utt_start > LONG_UTTERANCE_S
                if gap >= RESET_GAP_S + lag or (long_utt and gap >= LONG_RESET_GAP_S + lag):
                    if pending:
                        committed.append(pending)
                        pending = ""
                    logger.debug("New utterance at %.2fs (gap %.2fs, %d tokens)", fed, gap, seen)
                    # A fresh stream seeded with the recent tail, rather than reset(): a word
                    # may be starting in audio not yet decoded, and reset() would cut it off
                    # from its own onset. The tail produced no tokens (that is the gap), so
                    # re-decoding it cannot repeat a word.
                    stream = self._new_stream(rec, tail)
                    seen = 0
                    utt_start = fed

            if pending and fed - last_token_audio > PENDING_FLUSH_S + lag:
                committed.append(pending)
                pending = ""
            if committed or pending != before:
                self.on_words(committed, pending)
