"""Live captions from NVIDIA Parakeet-unified 0.6B, streamed on a CUDA GPU.

parakeet-unified-en-0.6b is one model trained for both offline and streaming use, with
punctuation and capitals built in. sherpa-onnx's streaming export is "buffered": each
step re-encodes a window of left context + new audio + a little lookahead, and only the
new frames are decoded. That costs a full 0.6B encoder pass per step, about 3x slower than
real time on the CPU, so this engine needs CUDA (an fp32 pass over a 6 s window takes
~65 ms on a GTX 1070; int8 and fp16 are slower there, Pascal has no fast paths for either).

On the 16-scene benchmark (bench/README) it scored 14.4% WER on the real-world scenes
against 17.0% for Nemotron 560 ms on the CPU, at about 0.35x real time with the CPU left free.

The model runs in a child process (`python -m live_subtitles unified-worker`): onnxruntime-gpu
and sherpa-onnx each ship an onnxruntime.dll and cannot share one process, and the decode
loop stays off the UI process's GIL. `UnifiedStreamer` is the engine itself (the
benchmark drives it in-process); `UnifiedEngine` is the app's proxy to the worker.

Protocol on the worker's stdin: a JSON line {"samples": k} then k float32 samples, or
{"idle": true} when no audio arrived. Each request gets one JSON line back: {"updates":
[[committed, pending, phrase], ...]}. The worker's first line is {"event": "ready", ...}
or {"event": "error", "message": ...}.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

from .engines import IDLE_POLL_S, SAMPLE_RATE, Leveler, Update

logger = logging.getLogger(__name__)

HOP = 160                   # 10 ms feature frames
SUB = 8                     # feature frames per encoder frame
LEFT_FRAMES = 560           # left context, feature frames (the export's training setting: 5.6 s)
RIGHT_FRAMES = 16           # lookahead, feature frames (160 ms)
MIN_CHUNKS = 3              # encoder frames decoded per step: 3 = a step every 240 ms...
MAX_CHUNKS = 8              # ...more when audio has queued up, so a busy GPU catches up
FLUSH_S = 0.5               # zeros fed after audio stops, so the last words get decoded
PHRASE_GAP_S = 0.6          # decoded audio with no token that starts a new phrase
MAX_SYMBOLS = 10            # tokens per encoder frame, at most
PROJECT_DIR = Path(__file__).resolve().parent.parent


class UnifiedStreamer:
    """The engine: `process(audio | None) -> list[Update]`, like `engines.NemotronEngine`."""

    def __init__(self, model_dir: Path, leveler: bool = True, left_frames: int = LEFT_FRAMES,
                 use_gpu: bool = True):
        self.dir = Path(model_dir)
        self.leveler = Leveler() if leveler else None
        self.left = left_frames - left_frames % SUB
        self.use_gpu = use_gpu
        self.name = f"parakeet-unified ({self.left * 10} ms context)"
        self.provider = ""

    # -- setup ---------------------------------------------------------------------------
    def load(self) -> None:
        import onnxruntime as ort

        t = time.perf_counter()
        if self.use_gpu:
            ort.preload_dlls()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("onnxruntime has no CUDA provider")
        ort.set_default_logger_severity(3)

        def opts(threads: int):
            o = ort.SessionOptions()
            o.intra_op_num_threads = threads
            o.inter_op_num_threads = 1
            # Spinning threads would steal the CPU from everything else for no gain here.
            o.add_session_config_entry("session.intra_op.allow_spinning", "0")
            o.add_session_config_entry("session.inter_op.allow_spinning", "0")
            return o

        enc_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(self.dir / "encoder.onnx"), opts(2), providers=enc_providers)
        self.provider = self.enc.get_providers()[0]
        if self.use_gpu and self.provider != "CUDAExecutionProvider":
            raise RuntimeError("the encoder did not load on the GPU")
        self.dec = ort.InferenceSession(str(self.dir / "decoder.onnx"), opts(1), providers=["CPUExecutionProvider"])
        self.joi = ort.InferenceSession(str(self.dir / "joiner.onnx"), opts(1), providers=["CPUExecutionProvider"])
        self.dec_in = [i.name for i in self.dec.get_inputs()]
        self.tokens: dict[int, str] = {}
        for line in (self.dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
            tok, idx = line.rsplit(" ", 1)
            self.tokens[int(idx)] = tok
        self.blank = len(self.tokens) - 1
        with files("onnx_asr.preprocessors").joinpath("data").joinpath("fbanks.npz").open("rb") as f:
            self.fbank = np.load(f)["nemo128"].astype(np.float32)        # (257, 128), Slaney mel
        self.window = np.pad(np.hanning(400), (56, 56)).astype(np.float32)
        self.reset()
        # Warm up CUDA kernels so the first caption isn't late.
        w = np.zeros((1, 128, self.left + MIN_CHUNKS * SUB + RIGHT_FRAMES), np.float32)
        self.enc.run(None, {"audio_signal": w, "length": np.array([w.shape[2]], np.int64)})
        logger.info("Loaded %s on %s in %.1fs", self.name, self.provider, time.perf_counter() - t)

    def reset(self) -> None:
        # Features: pre-emphasised audio waiting to be framed (starting at sample `pre_at`;
        # NeMo centres frames, so frame i spans samples [160 i - 256, 160 i + 256)).
        self.pre = np.zeros(256, np.float32)
        self.pre_at = -256
        self.last_sample = 0.0
        self.feats = np.zeros((0, 128), np.float32)
        self.feat_at = 0                # frame index of feats[0]
        self.n_feats = 0                # frames computed so far
        self.center = 0                 # next frame to decode
        # Decoder state.
        self.s0 = np.zeros((2, 1, 640), np.float32)
        self.s1 = np.zeros((2, 1, 640), np.float32)
        self.dec_out, self.n0, self.n1 = self._decode(self.blank, self.s0, self.s1)
        # Words.
        # The word being spelled out: shown, not committed. It stays open through a pause,
        # because the model emits a sentence's final period only once the next sentence
        # starts (it can't know the sentence ended until then); committed early, the period
        # would arrive with no word to attach to.
        self.word = ""
        self.shown_word = ""
        self.last_token_t = 0.0         # decoded-audio time of the last token
        self.phrase = 0
        self.phrase_words = 0
        self.idle_fed = FLUSH_S

    # -- streaming -----------------------------------------------------------------------
    def process(self, audio: np.ndarray | None) -> list[Update]:
        if audio is None:
            if self.idle_fed >= FLUSH_S:
                return self._flush_pending(force=True)
            n = int(SAMPLE_RATE * IDLE_POLL_S)
            self.idle_fed += IDLE_POLL_S
            audio = np.zeros(n, np.float32)
        else:
            self.idle_fed = 0.0
            if self.leveler:
                audio = self.leveler.process(audio.astype(np.float32, copy=False))
        self._features(audio)
        committed: list[str] = []
        while True:
            ready = self.n_feats - RIGHT_FRAMES - self.center
            chunks = min(MAX_CHUNKS, ready // SUB)
            if chunks < MIN_CHUNKS:
                break
            committed += self._step(chunks)
        if committed or self.word != self.shown_word:
            self.shown_word = self.word
            return [(committed, [self.word] if self.word else [], self.phrase)]
        return []

    def _features(self, audio: np.ndarray) -> None:
        if audio.size:
            x = audio.astype(np.float32, copy=False)
            emph = np.empty_like(x)
            emph[0] = x[0] - 0.97 * self.last_sample
            emph[1:] = x[1:] - 0.97 * x[:-1]
            self.last_sample = float(x[-1])
            self.pre = np.concatenate([self.pre, emph])
        # Frame i needs pre-emphasised samples up to 160 i + 256.
        end_sample = self.pre_at + self.pre.size
        n_ready = (end_sample - 256) // HOP + 1 - self.n_feats
        if n_ready <= 0:
            return
        first = self.n_feats * HOP - 256 - self.pre_at          # offset of frame n_feats in pre
        span = self.pre[first:first + (n_ready - 1) * HOP + 512]
        frames = np.lib.stride_tricks.sliding_window_view(span, 512)[::HOP][:n_ready] * self.window
        spec = np.abs(np.fft.rfft(frames, 512)).astype(np.float32) ** 2
        new = np.log(spec @ self.fbank + 2 ** -24).astype(np.float32)
        self.feats = np.concatenate([self.feats, new])
        self.n_feats += n_ready
        keep_from = self.n_feats * HOP - 256 - self.pre_at
        self.pre = self.pre[keep_from:]
        self.pre_at += keep_from
        # Keep what the next window needs.
        drop = (self.center - self.left) - self.feat_at
        if drop > 0:
            self.feats = self.feats[drop:]
            self.feat_at += drop

    def _step(self, chunks: int) -> list[str]:
        c = self.center
        a, b = c - self.left, c + chunks * SUB + RIGHT_FRAMES
        w = self.feats[max(0, a - self.feat_at):b - self.feat_at]
        if a < 0:
            w = np.concatenate([np.zeros((-a, 128), np.float32), w])
        w = (w - w.mean(0, keepdims=True)) / (w.std(0, keepdims=True) + 1e-5)
        enc = self.enc.run(None, {"audio_signal": w.T[None].astype(np.float32, copy=False),
                                  "length": np.array([w.shape[0]], np.int64)})[0]
        first = self.left // SUB
        out: list[str] = []
        for f in range(first, first + chunks):
            et = enc[:, :, f:f + 1]
            for _ in range(MAX_SYMBOLS):
                logits = self.joi.run(None, {"encoder_outputs": et, "decoder_outputs": self.dec_out})[0]
                idx = int(np.argmax(logits.reshape(-1)[: self.blank + 1]))
                if idx == self.blank:
                    break
                self.s0, self.s1 = self.n0, self.n1
                self.dec_out, self.n0, self.n1 = self._decode(idx, self.s0, self.s1)
                t = (c + (f - first) * SUB) * HOP / SAMPLE_RATE
                out += self._token(self.tokens.get(idx, ""), t)
        self.center = c + chunks * SUB
        return out

    def _decode(self, token: int, s0, s1):
        o = self.dec.run(None, {self.dec_in[0]: np.array([[token]], np.int32),
                                self.dec_in[1]: np.array([1], np.int32),
                                self.dec_in[2]: s0, self.dec_in[3]: s1})
        return o[0], o[2], o[3]

    def _token(self, tok: str, t: float) -> list[str]:
        if not tok or (tok.startswith("<") and tok.endswith(">")):
            return []
        text = tok[1:] if tok.startswith("▁") else tok
        if text and not any(ch.isalnum() for ch in text):
            # Punctuation belongs to the word before it, even when it comes as a new piece
            # ("▁."). With no open word (audio stopped and it was committed) it is dropped.
            if self.word:
                self.word += text
            return []
        out = []
        if t - self.last_token_t >= PHRASE_GAP_S and self.phrase_words:
            self.phrase += 1
            self.phrase_words = 0
        self.last_token_t = t
        if tok.startswith("▁"):
            if self.word:
                out.append(self.word)
            self.word = text
            self.phrase_words += 1
        else:
            self.word += text
        return out

    def _flush_pending(self, force: bool) -> list[Update]:
        """Audio stopped: commit the word in progress."""
        if self.word and force:
            word, self.word, self.shown_word = self.word, "", ""
            return [([word], [], self.phrase)]
        return []


# -- child process ---------------------------------------------------------------------
def model_dir() -> Path:
    from .models import UNIFIED_MODEL

    return UNIFIED_MODEL.path


def worker_main(argv: list[str]) -> int:
    out = sys.stdout
    try:
        streamer = UnifiedStreamer(model_dir(), leveler="--no-leveler" not in argv)
        streamer.load()
    except Exception as exc:
        out.write(json.dumps({"event": "error", "message": str(exc)}) + "\n")
        out.flush()
        return 1
    out.write(json.dumps({"event": "ready", "provider": streamer.provider, "name": streamer.name}) + "\n")
    out.flush()
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return 0
        msg = json.loads(header)
        if msg.get("idle"):
            updates = streamer.process(None)
        else:
            raw = stdin.read(int(msg["samples"]) * 4)
            updates = streamer.process(np.frombuffer(raw, np.float32))
        out.write(json.dumps({"updates": updates}) + "\n")
        out.flush()


# -- parent side -----------------------------------------------------------------------
def cuda_available() -> tuple[bool, str]:
    """Is there an NVIDIA GPU with a working driver? Asks the driver (nvcuda.dll) directly,
    so the tray process never loads onnxruntime."""
    import ctypes

    try:
        cuda = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False, "no NVIDIA driver"
    count = ctypes.c_int(0)
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
        return False, "no CUDA device"
    return True, ""


def _python_exe() -> str:
    """The venv's console python: pythonw has no usable stdio for the pipe protocol."""
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")
    return str(console if console.exists() else exe)


class UnifiedEngine:
    """The app's engine: forwards audio to the worker process and relays its updates.

    `process` never waits on the GPU: it hands the audio over and returns whatever the
    worker has finished since the last call, so capture keeps flowing even if a step runs
    long. `load` blocks until the worker reports ready (or raises with its error).
    """

    def __init__(self, leveler: bool = True):
        self.leveler = leveler
        self.name = "parakeet-unified (GPU)"
        self.provider = ""
        self._proc: subprocess.Popen | None = None
        self._ready = threading.Event()
        self._error = ""
        self._updates: list[Update] = []
        self._lock = threading.Lock()
        self._inflight = 0

    def load(self, timeout: float = 120.0) -> None:
        if self._proc is not None:          # the app loads it up front; LiveRecognizer calls again
            return
        cmd = [_python_exe(), "-m", "live_subtitles", "unified-worker", *([] if self.leveler else ["--no-leveler"])]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, env=env,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read, name="unified-reader", daemon=True).start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError("the GPU speech worker did not start")
        if self._error:
            self.close()
            raise RuntimeError(self._error)

    def process(self, audio: np.ndarray | None) -> list[Update]:
        if self._error:
            raise RuntimeError(self._error)
        # While the worker is still busy, don't queue up idle ticks behind real audio.
        if audio is not None or self._inflight == 0:
            self._send(audio)
        with self._lock:
            out, self._updates = self._updates, []
        return out

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(3)
            except Exception:
                proc.kill()

    def _send(self, audio: np.ndarray | None) -> None:
        proc = self._proc
        if proc is None:
            raise RuntimeError("the GPU speech worker is not running")
        try:
            if audio is None:
                proc.stdin.write(b'{"idle": true}\n')
            else:
                data = np.ascontiguousarray(audio, dtype=np.float32)
                proc.stdin.write((json.dumps({"samples": int(data.size)}) + "\n").encode())
                proc.stdin.write(data.tobytes())
            proc.stdin.flush()
            with self._lock:
                self._inflight += 1
        except OSError as exc:
            self._error = f"the GPU speech worker stopped ({exc})"
            raise RuntimeError(self._error) from exc

    def _read(self) -> None:
        proc = self._proc
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("event") == "ready":
                self.provider = msg.get("provider", "")
                self.name = msg.get("name", self.name)
                self._ready.set()
            elif msg.get("event") == "error":
                self._error = msg.get("message", "the GPU speech worker failed")
                self._ready.set()
            elif "updates" in msg:
                with self._lock:
                    self._inflight -= 1
                    self._updates += [(c, p, ph) for c, p, ph in msg["updates"]]
        if self._proc is proc and not self._error:
            self._error = "the GPU speech worker exited"
        self._ready.set()
