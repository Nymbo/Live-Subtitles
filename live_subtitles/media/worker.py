"""The media worker: speech models for media mode, in their own process.

`python -m live_subtitles media-worker` (started by the app) holds the models that need
ONNX Runtime, which the tray process must never import (see AGENTS.md):

- words:     Parakeet TDT 0.6B v3 with word times, for text anchors and English tracks
             generated from the audio. int8 on the CPU by default, so the GPU stays free
             for the live-caption bridge; `gpu` moves it.
- vad:       Silero VAD, speech spans for speech-span alignment.
- translate: Whisper medium with the translate task: Korean/Japanese/... speech straight
             to English text, one VAD segment at a time. Uses the GPU when there is one.
- tags:      AST AudioSet tagging, for generated sound cues (`tagger.py`).

Models load on first use. Protocol on stdin: a JSON line {"id", "op", "samples": k,
"t0": seconds, ...} followed by k float32 samples (16 kHz mono); each request gets one
JSON line back with the same id. The first line out is {"event": "ready"}.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
SR = 16000
WHISPER = "onnx-community/whisper-medium-ONNX"
# Whisper's favourite inventions over music and silence.
HALLUCINATIONS = re.compile(
    r"^(?:thank(?:s| you)(?: (?:so much|very much))?(?: for watching)?[.!]*|please subscribe.*|"
    r"subtitles? by.*|translated by.*|amara\.org.*|you\.?|bye\.?|\.+|music|♪+)$", re.I)


class Models:
    def __init__(self, gpu_words: bool, gpu_translate: bool):
        self.gpu_words, self.gpu_translate = gpu_words, gpu_translate
        self._words = self._vad = self._whisper = self._tagger = None

    def words_model(self):
        if self._words is None:
            from ..transcribe import load_model

            self._words, provider = load_model(self.gpu_words)
            logger.info("words model on %s", provider)
        return self._words

    def vad(self):
        if self._vad is None:
            import onnx_asr

            self._vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
        return self._vad

    def whisper(self):
        if self._whisper is None:
            import onnx_asr
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.gpu_translate:
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
            model = onnx_asr.load_model(WHISPER, providers=providers)
            inner = model.asr
            # The decoder prompt is [sot, language, task, notimestamps]; swap the task.
            inner._transcribe_input[0, 2] = inner._tokens["<|translate|>"]
            self._whisper = model
        return self._whisper

    def tagger(self):
        if self._tagger is None:
            from .tagger import AudioTagger

            self._tagger = AudioTagger()
        return self._tagger


def speech_spans(models: Models, audio: np.ndarray, t0: float, max_s: float = 20.0) -> list[tuple[float, float]]:
    vad = models.vad()
    segs = next(iter(vad.segment_batch(audio[None], np.array([len(audio)], np.int64), SR,
                                       max_speech_duration_s=max_s, min_silence_duration_ms=250)))
    return [(t0 + a / SR, t0 + b / SR) for a, b in segs]


def recognise_words(models: Models, audio: np.ndarray, t0: float) -> list[tuple[str, float, float]]:
    from ..transcribe import window_words

    if len(audio) < SR // 2:
        return []
    res = models.words_model().recognize(audio, sample_rate=SR)
    return [(w.text, w.start, w.end) for w in window_words(res, t0)]


def translate(models: Models, audio: np.ndarray, t0: float, language: str) -> list[tuple[float, float, str]]:
    """English text for each speech segment: [(start, end, text)]."""
    out = []
    model = models.whisper()
    lang = (language or "").split("-")[0].lower() or None
    for a, b in speech_spans(models, audio, t0, max_s=18.0):
        if b - a < 0.6:
            continue
        piece = audio[int((a - t0) * SR):int((b - t0) * SR)]
        try:
            text = model.recognize(piece, sample_rate=SR, language=lang) if lang else model.recognize(piece, sample_rate=SR)
        except Exception:
            # An unknown language token: let Whisper detect it.
            text = model.recognize(piece, sample_rate=SR)
        text = re.sub(r"\s+", " ", text or "").strip()
        if text and not HALLUCINATIONS.match(text):
            out.append((a, b, text))
    return out


def handle(models: Models, msg: dict, audio: np.ndarray) -> dict:
    op, t0 = msg["op"], float(msg.get("t0", 0.0))
    if op == "words":
        return {"words": recognise_words(models, audio, t0)}
    if op == "vad":
        return {"spans": speech_spans(models, audio, t0)}
    if op == "translate":
        return {"segments": translate(models, audio, t0, msg.get("language", ""))}
    if op == "tags":
        return {"tags": models.tagger().tag(audio, t0)}
    if op == "load":
        for name in msg.get("models", []):
            {"words": models.words_model, "vad": models.vad, "translate": models.whisper,
             "tags": models.tagger}[name]()
        return {"loaded": msg.get("models", [])}
    raise ValueError(f"unknown op {op}")


def worker_main(argv: list[str]) -> int:
    out = sys.stdout
    models = Models(gpu_words="--gpu-words" in argv, gpu_translate="--cpu-translate" not in argv)
    out.write(json.dumps({"event": "ready"}) + "\n")
    out.flush()
    stdin = sys.stdin.buffer
    while True:
        header = stdin.readline()
        if not header:
            return 0
        msg = json.loads(header)
        n = int(msg.get("samples", 0))
        audio = np.frombuffer(stdin.read(n * 4), np.float32) if n else np.zeros(0, np.float32)
        try:
            reply = handle(models, msg, audio)
        except Exception as exc:
            logger.exception("media worker: %s failed", msg.get("op"))
            reply = {"error": str(exc)}
        reply["id"] = msg.get("id")
        out.write(json.dumps(reply) + "\n")
        out.flush()


# -- parent side -------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent


class MediaWorker:
    """The app's handle on the worker. `call` blocks the calling (background) thread."""

    def __init__(self, gpu_words: bool = False, gpu_translate: bool = True):
        self.args = [*(["--gpu-words"] if gpu_words else []), *([] if gpu_translate else ["--cpu-translate"])]
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next = 0

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        from ..unified import _python_exe

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen([_python_exe(), "-m", "live_subtitles", "media-worker", *self.args],
                                      cwd=PROJECT_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, env=env,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        line = self._proc.stdout.readline()
        if not line or json.loads(line).get("event") != "ready":
            self.close()
            raise RuntimeError("the media worker did not start")

    def call(self, op: str, audio: np.ndarray | None = None, **kw) -> dict:
        with self._lock:
            self.start()
            self._next += 1
            data = np.ascontiguousarray(audio if audio is not None else np.zeros(0), dtype=np.float32)
            header = {"id": self._next, "op": op, "samples": int(data.size), **kw}
            try:
                self._proc.stdin.write((json.dumps(header) + "\n").encode())
                if data.size:
                    self._proc.stdin.write(data.tobytes())
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except OSError as exc:
                self.close()
                raise RuntimeError(f"the media worker stopped ({exc})") from exc
            if not line:
                self.close()
                raise RuntimeError("the media worker exited")
            reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(3)
            except Exception:
                proc.kill()
