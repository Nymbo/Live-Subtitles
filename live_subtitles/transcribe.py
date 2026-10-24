"""Video/audio file -> subtitle files, with Parakeet TDT 0.6B v3 (offline, full context).

Runs in its own process (`python -m live_subtitles transcribe ...`). That keeps the GPU
ONNX Runtime used here out of the tray app, where sherpa-onnx has loaded its own CPU
build of onnxruntime.dll; two different DLLs with one name cannot share a process.

The audio is decoded by ffmpeg and transcribed in overlapping 20 s windows; Parakeet drops
more speech over longer windows of film audio. No VAD: on film audio Silero discarded
dialogue under music (44-48% WER with it, 26% without).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import settings as settings_mod
from .subtitles import Word, build_cues, to_ass, to_srt, to_vtt

logger = logging.getLogger(__name__)

SR = 16000
MODEL = "nemo-parakeet-tdt-0.6b-v3"
WORD_TAIL_S = 0.35      # a word ends ~this long after its last token starts (TDT gives starts only)


def emit(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def probe(path: Path) -> tuple[float, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("No audio track found")
    duration = float(info.get("format", {}).get("duration") or 0.0)
    return duration, int(streams[0].get("channels") or 2)


def decode(path: Path, channels: int, dialogue_boost: bool) -> subprocess.Popen:
    if dialogue_boost and channels >= 6:
        mix = ["-af", "pan=mono|c0=0.6*FC+0.2*FL+0.2*FR"]
    else:
        mix = ["-ac", "1"]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", *mix,
           "-ar", str(SR), "-f", "f32le", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def load_model(use_gpu: bool):
    import onnxruntime as ort
    import onnx_asr

    providers = ["CPUExecutionProvider"]
    if use_gpu:
        try:
            ort.preload_dlls()
        except Exception:
            logger.warning("Could not preload CUDA DLLs", exc_info=True)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
    model = onnx_asr.load_model(MODEL, quantization=None if use_gpu else "int8", providers=providers)
    return model.with_timestamps(), providers[0]


def window_words(result, offset: float) -> list[Word]:
    """Group sub-word tokens into words with absolute times."""
    groups: list[list] = []        # [text, first_token_time, last_token_time]
    for tok, ts in zip(result.tokens or [], result.timestamps or []):
        t = offset + ts
        if tok.startswith(" ") or not groups:
            groups.append([tok.strip(), t, t])
        else:
            groups[-1][0] += tok
            groups[-1][2] = t
    words = []
    for i, (text, start, last) in enumerate(groups):
        if not text:
            continue
        end = last + WORD_TAIL_S
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][1])
        words.append(Word(text, start, max(end, start + 0.08)))
    return words


def transcribe_audio(model, stream, duration: float, window_s: float, overlap_s: float,
                     on_progress=None) -> list[Word]:
    """Read f32le mono from `stream` and transcribe in overlapping windows."""
    words: list[Word] = []
    for batch in iter_words(model, stream, duration, window_s, overlap_s, on_progress):
        words += batch
    return words


def iter_words(model, stream, duration: float, window_s: float = 20.0, overlap_s: float = 4.0,
               on_progress=None, offset: float = 0.0):
    """Like transcribe_audio, yielding each window's words as soon as it is decoded
    (media mode's generator writes cues while it is still running). `offset` is the
    absolute time of the stream's first sample."""
    win, hop = int(window_s * SR), int((window_s - overlap_s) * SR)
    half = overlap_s / 2
    buf = np.zeros(0, dtype=np.float32)
    start = offset
    first = True
    eof = False
    while True:
        while not eof and buf.size < win:
            raw = stream.read(SR * 4 * 10)
            if not raw:
                eof = True
                break
            raw = raw[: len(raw) - len(raw) % 4]
            buf = np.concatenate([buf, np.frombuffer(raw, dtype=np.float32)])
        if buf.size == 0:
            break
        last = eof and buf.size <= win
        piece = buf[:win]
        res = model.recognize(piece, sample_rate=SR)
        lo = offset if first else offset + half
        hi = offset + piece.size / SR if last else offset + window_s - half
        yield [w for w in window_words(res, offset) if lo <= w.start < hi]
        if on_progress:
            done = offset + piece.size / SR - start
            on_progress(min(1.0, done / duration) if duration else 0.0)
        if last:
            break
        buf = buf[hop:]
        offset += hop / SR
        first = False


def output_paths(video: Path, out_dir: Path | None, formats: list[str], overwrite: bool) -> dict[str, Path]:
    folder = out_dir or video.parent
    plain = {fmt: folder / f"{video.stem}.{fmt}" for fmt in formats}
    if overwrite or not any(p.exists() for p in plain.values()):
        return plain
    # Never clobber subtitles that were already there; VLC still auto-loads <stem>.*.srt.
    return {fmt: folder / f"{video.stem}.live-subtitles.{fmt}" for fmt in formats}


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="live_subtitles transcribe",
                                 description="Write subtitle files for video or audio files.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--formats", default=None, help="comma list of srt,ass,vtt (default: from settings)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", default=None)
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (int8) instead of the GPU")
    # 10-30 s windows scored 22.7-24.5% WER on Tears of Steel; 60 s+ dropped more speech (30-31%).
    ap.add_argument("--window", type=float, default=20.0, help="seconds of audio per pass")
    ap.add_argument("--overlap", type=float, default=4.0)
    ap.add_argument("--ref-height", type=int, default=1080,
                    help="screen height the overlay style was designed on (scales the ASS style)")
    ap.add_argument("--json", action="store_true", help="machine-readable progress on stdout")
    args = ap.parse_args(argv)

    if not args.json:
        global emit
        emit = _human_emit  # noqa: F811

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        emit("error", message="ffmpeg and ffprobe must be on PATH")
        return 2

    cfg = settings_mod.load()
    formats = [f.strip().lower() for f in (args.formats or ",".join(cfg.files.formats)).split(",") if f.strip()]
    formats = [f for f in formats if f in ("srt", "ass", "vtt")] or ["srt"]
    overwrite = cfg.files.overwrite if args.overwrite is None else args.overwrite
    use_gpu = cfg.files.use_gpu and not args.cpu

    emit("status", message="Loading Parakeet…")
    t = time.perf_counter()
    model, provider = load_model(use_gpu)
    emit("status", message=f"Model ready on {provider.replace('ExecutionProvider', '')} "
                           f"({time.perf_counter() - t:.0f}s)")

    failures = 0
    for i, path in enumerate(args.files, 1):
        path = path.resolve()
        try:
            duration, channels = probe(path)
            emit("start", file=str(path), index=i, total=len(args.files), duration=duration)
            proc = decode(path, channels, cfg.engine.dialogue_boost)
            t = time.perf_counter()
            last = [0.0]

            def progress(frac, _path=str(path)):
                if frac - last[0] >= 0.01 or frac >= 1.0:
                    last[0] = frac
                    emit("progress", file=_path, fraction=round(frac, 4))

            words = transcribe_audio(model, proc.stdout, duration, args.window, args.overlap, progress)
            proc.wait()
            if proc.returncode:
                raise RuntimeError(proc.stderr.read().decode(errors="replace").strip() or "ffmpeg failed")
            cues = build_cues(words, max_chars=cfg.files.max_line_chars, max_lines=cfg.files.max_cue_lines)
            outputs = output_paths(path, args.out_dir, formats, overwrite)
            writers = {"srt": to_srt, "vtt": to_vtt,
                       "ass": lambda c: to_ass(c, cfg.appearance, args.ref_height)}
            for fmt, out in outputs.items():
                out.parent.mkdir(parents=True, exist_ok=True)
                # utf-8-sig: VLC and most players read a BOM-marked SRT as UTF-8 without guessing.
                out.write_text(writers[fmt](cues), encoding="utf-8-sig" if fmt == "srt" else "utf-8")
            emit("done", file=str(path), outputs=[str(p) for p in outputs.values()], cues=len(cues),
                 words=len(words), seconds=round(time.perf_counter() - t, 1))
        except Exception as exc:
            failures += 1
            logger.exception("Failed on %s", path)
            emit("error", file=str(path), message=str(exc))
    return 1 if failures else 0


def _human_emit(event: str, **data) -> None:
    if event == "progress":
        print(f"\r  {data['fraction'] * 100:5.1f}%", end="", flush=True)
    elif event == "done":
        print(f"\r  done in {data['seconds']}s: {data['cues']} cues")
        for p in data["outputs"]:
            print(f"  -> {p}")
    elif event == "start":
        print(f"[{data['index']}/{data['total']}] {data['file']}")
    else:
        print(data.get("message", ""), file=sys.stderr if event == "error" else sys.stdout)
