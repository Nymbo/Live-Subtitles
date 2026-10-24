"""Subtitles generated from a video's own audio, ahead of the playhead.

`python -m live_subtitles media-generate SOURCE --start S [--translate ko] [--sounds]`
reads SOURCE (a local file, or a stream URL the browser extension saw, with the page's
Referer) through ffmpeg from S seconds on, and prints JSON lines as it goes:

    {"event": "cues", "cues": [[start, end, [lines]], ...]}     media times, in order
    {"event": "sounds", "cues": [...]}                           generated sound cues
    {"event": "progress", "position": seconds}
    {"event": "done"} | {"event": "error", "message": ...}

Decoding runs much faster than real time, so starting at the playhead keeps it ahead.
English audio goes through Parakeet TDT (the same path as `transcribe`); other languages
through Whisper's translate task, so the cues are English either way. The audio is only
streamed through ffmpeg; nothing but the cues is kept.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys

import numpy as np

from ..subtitles import Cue, Word, build_cues, split_lines

logger = logging.getLogger(__name__)
SR = 16000
BLOCK_S = 60.0          # translation and sound tagging run per block of this much audio


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def ffmpeg_audio(source: str, start: float = 0.0, duration: float | None = None, referer: str = "",
                 user_agent: str = "") -> subprocess.Popen:
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if source.startswith(("http://", "https://")):
        headers = ""
        if referer:
            headers += f"Referer: {referer}\r\n"
            from urllib.parse import urlsplit

            parts = urlsplit(referer)
            headers += f"Origin: {parts.scheme}://{parts.netloc}\r\n"
        if headers:
            cmd += ["-headers", headers]
        if user_agent:
            cmd += ["-user_agent", user_agent]
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", source]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def text_cues(segments: list[tuple[float, float, str]], max_chars: int = 42, max_lines: int = 2) -> list[Cue]:
    """Translated segments -> cues, splitting a long segment by words across several cues."""
    out = []
    for a, b, text in segments:
        words = text.split()
        chunks, current = [], []
        for w in words:
            if current and split_lines(current + [w], max_chars, max_lines) is None:
                chunks.append(current)
                current = []
            current.append(w)
        if current:
            chunks.append(current)
        total = sum(len(" ".join(c)) for c in chunks) or 1
        t = a
        for c in chunks:
            share = (b - a) * len(" ".join(c)) / total
            lines = split_lines(c, max_chars, max_lines) or [" ".join(c)]
            out.append(Cue(t, t + share, lines))
            t += share
    return out


def cues_json(cues: list[Cue]) -> list:
    return [[round(c.start, 3), round(c.end, 3), c.lines] for c in cues]


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="live_subtitles media-generate")
    ap.add_argument("source")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--referer", default="")
    ap.add_argument("--user-agent", default="")
    ap.add_argument("--translate", default="", help="audio language to translate to English (e.g. ko, ja)")
    ap.add_argument("--sounds", action="store_true", help="also tag sound events")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    if not shutil.which("ffmpeg"):
        emit("error", message="ffmpeg must be on PATH")
        return 2

    from .worker import Models, translate
    from ..transcribe import iter_words, load_model

    models = Models(gpu_words=not args.cpu, gpu_translate=not args.cpu)
    try:
        proc = ffmpeg_audio(args.source, args.start, None, args.referer, args.user_agent)
        if args.translate:
            models.whisper()
            _translated(proc, args, models, translate)
        else:
            model, provider = load_model(not args.cpu)
            models._words = model
            emit("status", message=f"Generating on {provider.replace('ExecutionProvider', '')}")
            _english(proc, args, models, iter_words)
        proc.wait()
        if proc.returncode:
            err = proc.stderr.read().decode(errors="replace").strip()
            if err:
                raise RuntimeError(err[-300:])
        emit("done")
        return 0
    except Exception as exc:
        logger.exception("media-generate failed")
        emit("error", message=str(exc))
        return 1


class _Tee:
    """Reads the ffmpeg pipe for iter_words while keeping the samples for sound tagging."""

    def __init__(self, stream, keep: bool):
        self.stream, self.keep, self.buf = stream, keep, bytearray()

    def read(self, n: int) -> bytes:
        data = self.stream.read(n)
        if self.keep:
            self.buf += data
        return data

    def take(self) -> np.ndarray:
        n = len(self.buf) - len(self.buf) % 4
        out = np.frombuffer(bytes(self.buf[:n]), np.float32).copy()
        del self.buf[:n]
        return out


def _english(proc, args, models, iter_words) -> None:
    tee = _Tee(proc.stdout, args.sounds)
    pending: list[Word] = []
    sound_at = args.start
    for batch in iter_words(models._words, tee, 0.0, offset=args.start):
        pending += batch
        if pending:
            emit("progress", position=round(pending[-1].end, 2))
        # Emit everything up to the last clear pause; the words after it may still grow a cue.
        cut = _last_pause(pending)
        if cut:
            emit("cues", cues=cues_json(build_cues(pending[:cut])))
            pending = pending[cut:]
        if args.sounds and len(tee.buf) >= BLOCK_S * SR * 4:
            audio = tee.take()
            emit("sounds", cues=_sound_cues(models, audio, sound_at))
            sound_at += len(audio) / SR
    if pending:
        emit("cues", cues=cues_json(build_cues(pending)))
    if args.sounds and tee.buf:
        emit("sounds", cues=_sound_cues(models, tee.take(), sound_at))


def _last_pause(words: list[Word], gap: float = 0.7) -> int:
    for i in range(len(words) - 1, 0, -1):
        if words[i].start - words[i - 1].end > gap and words[-1].end - words[i].start > 3:
            return i
    return 0


def _translated(proc, args, models, translate) -> None:
    t0 = args.start
    block = int(BLOCK_S * SR)
    buf = np.zeros(0, np.float32)
    eof = False
    while not eof:
        while buf.size < block:
            raw = proc.stdout.read(SR * 4 * 10)
            if not raw:
                eof = True
                break
            raw = raw[: len(raw) - len(raw) % 4]
            buf = np.concatenate([buf, np.frombuffer(raw, np.float32)])
        if buf.size == 0:
            break
        # Cut the block at a quiet moment so no line is split across two blocks.
        cut = buf.size if eof else _quiet_cut(buf, block)
        piece, buf = buf[:cut], buf[cut:]
        segments = translate(models, piece, t0, args.translate)
        emit("cues", cues=cues_json(text_cues(segments)))
        if args.sounds:
            emit("sounds", cues=_sound_cues(models, piece, t0))
        t0 += piece.size / SR
        emit("progress", position=round(t0, 2))


def _quiet_cut(buf: np.ndarray, near: int) -> int:
    """The quietest 100 ms in the last 10 s before `near` (a cut between lines)."""
    lo = max(0, near - 10 * SR)
    hop = SR // 10
    frames = buf[lo:near][: (near - lo) // hop * hop].reshape(-1, hop)
    if not len(frames):
        return near
    return lo + int(np.argmin(np.square(frames).mean(axis=1))) * hop + hop // 2


def _sound_cues(models, audio: np.ndarray, t0: float) -> list:
    if audio.size < SR * 2:
        return []
    events = models.tagger().tag(audio, t0)
    return [[round(a, 2), round(b, 2), [caption]] for caption, a, b, _p in events]


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    return run(argv)
