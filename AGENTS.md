# Agent Instructions

Live Subtitles is a Windows tray app (PySide6): live captions of whatever the PC is
playing, in a draggable overlay, plus `python -m live_subtitles transcribe <video>` for
SRT/ASS/VTT files. `README.md` records the benchmark behind every model and pipeline
choice; read it before changing a model, an engine, or the caption pipeline.

## Layout

- `live_subtitles/` is the app. `app.py` wires capture -> recogniser thread (`live.py`) ->
  caption builder (`captions.py`) -> overlay (`overlay.py`).
- Two live engines share one contract (`process(audio | None) -> list[Update]`):
  `unified.py` (Parakeet-unified on a CUDA GPU, the default for English) and
  `engines.py` (Nemotron streaming on the CPU: the fallback, and every other language).
- `bench/` is the live-caption benchmark: scene builders (`testset.py`, `realset.py`), a
  simulated-clock runner (`run.py`), `report.py`, real-time load (`cpu.py`), and an
  offline phrase comparison (`phrases.py`).
- `tests/test_core.py` covers the pure-Python parts.

## Rules that are easy to break

- **Two ONNX Runtimes must never share a process.** sherpa-onnx (Nemotron, punctuation)
  ships its own `onnxruntime.dll`; `onnxruntime-gpu` (the GPU engine, file transcription)
  ships another with the same name. The tray process imports only sherpa-onnx, never
  `onnxruntime` or `onnx_asr`: the GPU engine runs in `python -m live_subtitles
  unified-worker`, file transcription in a child process started by `jobs.py`. Importing
  either in the tray process breaks whichever loads second.
- **Pins for Pascal GPUs** (the dev machine has a GTX 1070): `onnxruntime-gpu` stays below
  1.27 (the first CUDA 13 build drops Pascal) and `nvidia-cudnn-cu12` at 9.10.2.21 (9.11+
  fails on Pascal). sherpa-onnx's CUDA wheels have no Pascal kernels; don't switch to them.
- **Data lives outside the repo.** Settings in `%APPDATA%\Live-Subtitles`; models, logs,
  benchmark audio and results in `%LOCALAPPDATA%\Live-Subtitles`. `APP_NAME` names those
  folders, so changing it orphans users' settings and models; `DISPLAY_NAME` is the
  user-facing name.
- **Captions never rewrite.** A committed word's text and line are final; only the words
  still in progress may change. The benchmark's rewrite and line-jump counts should stay 0.
- Measure model or pipeline changes with the benchmark (`bench.run` on all scenes, then
  `bench.report` against the previous label) and update the README's tables.

## Environment and checks

- `.\setup.ps1` builds `.venv` with uv from `requirements.txt`. Don't install with bare
  pip; change `requirements.txt` and re-run the script.
- Verify with the unit tests and the import check. A full launch puts an overlay on the
  screen and captures system audio, so only launch it when the task needs that:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -c "import live_subtitles.app, live_subtitles.settings_window, live_subtitles.transcribe, live_subtitles.unified, bench.run; print('imports ok')"
```
