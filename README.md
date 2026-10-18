# Live Subtitles

Live captions for whatever Windows is playing (a stream, a film, a call, a game), drawn
in a draggable, fully styleable overlay that disappears when nobody is talking. For the
times there are no subtitles at all. It also writes real subtitle files (SRT, ASS,
WebVTT) for video files on disk, for VLC and other players.

Everything runs locally. English captions use NVIDIA's Parakeet-unified on an NVIDIA GPU
(about 0.4 s behind speech, under one CPU core), or Nemotron streaming on the CPU when
there is no GPU and for 30 other languages. See [Models, and why](#models-and-why) for
the benchmark behind those choices.

A Windows tray app (WASAPI loopback capture).

## Install

Needs Windows 10 or 11, Python 3.13 and [uv](https://docs.astral.sh/uv/). For the GPU
engine, an NVIDIA GPU with ~3.2 GB of free VRAM and a current driver; CUDA itself comes
from pip. For subtitle files, ffmpeg and ffprobe on PATH.

```powershell
git clone https://github.com/Nymbo/Live-Subtitles.git
cd Live-Subtitles
.\setup.ps1
.\.venv\Scripts\pythonw.exe run.pyw          # the tray app, no console window
```

`.\.venv\Scripts\python.exe -m live_subtitles` runs it with a console (log output). A
second launch opens the running copy's settings. The speech model downloads on first use
(2.5 GB for the GPU engine, ~470 MB for a CPU model) into `%LOCALAPPDATA%\Live-Subtitles`.

Subtitle files without the GUI:

```powershell
.\.venv\Scripts\python.exe -m live_subtitles transcribe "D:\Movies\film.mkv" --formats srt,ass
```

Files land next to the video as `film.srt`, which VLC loads automatically. An existing
`film.srt` is never overwritten unless you ask; the new file becomes
`film.live-subtitles.srt` instead.

## Using it

- **Tray icon:** click to start or stop live subtitles, double-click for settings,
  right-click for the menu (lock, show/hide, subtitles for a video, jobs, logs).
- **Overlay:** while unlocked, drag it to move, drag either side to resize, Ctrl+scroll
  to change the text size, double-click to re-centre, right-click for options. Locking
  makes it click-through, so it never catches clicks meant for the player.
- **Hotkeys** (change them in Settings): Ctrl+Alt+S start/stop, Ctrl+Alt+L lock/unlock,
  Ctrl+Alt+H show/hide.
- **Settings → Look:** seven presets, plus your own saved ones. Font, size, weight,
  italic and uppercase; text colour; outline width and colour; shadow distance, softness
  and colour; a box behind each line or one around all lines, with its colour, padding
  and corner radius; alignment, line spacing and fade-out. Every colour has alpha. The
  preview draws with the overlay's own renderer over a dark or bright scene.
- **Settings → Captions:** lines on screen, characters per line, overlay width,
  roll-up or pop-on, how long until quiet clears the screen, a new line per sentence,
  and hiding filler words.
- **Settings → Audio & model:** which output to listen to, centre-channel dialogue boost
  for 5.1/7.1, language (English, auto-detect, or 30 locales), the recogniser (automatic,
  GPU or CPU), and the CPU engine's latency and threads.
- **Video files:** the `.ass` output carries the overlay's look (font, colours,
  outline/box, shadow), so VLC shows the same style as the live subtitles.

Settings live in `%APPDATA%\Live-Subtitles\settings.json`. Models and logs live in
`%LOCALAPPDATA%\Live-Subtitles\`.

## Models, and why

**English live captions run NVIDIA Parakeet-unified 0.6B on the GPU** (`unified.py`),
with **Nemotron streaming 560 ms on the CPU** as the fallback when there is no usable
NVIDIA GPU, and for every other language. The status line says which one is running.
Files are transcribed by Parakeet TDT 0.6B v3.

The choice comes from the benchmark below (`bench/`): 16 scenes, about 80 minutes of
audio, each run through the app's own caption pipeline on a simulated real-time clock.

| Live engine (English) | WER, real-world | WER, synthetic | Sentence F1 | Words on screen p50 / p90 | Speed | CPU |
|---|---|---|---|---|---|---|
| **Parakeet-unified, GPU** (default) | **13.8%** | 17.4% | **0.60** | 0.36 / 0.61 s | 0.28x real time | **0.8 cores** |
| Nemotron streaming 560 ms, CPU (fallback) | 17.0% | **14.9%** | 0.40 | 0.40 / 0.81 s | 0.21x | 9.4 cores at 4 threads; 1.9 at 2 |
| Nemotron streaming 160 ms, CPU | 17.6% | 15.0% | 0.37 | **0.21 / 0.54 s** | 0.44x | 9.3 cores |

Per real-world scene (WER %, Parakeet-unified vs Nemotron 560): TV monologue 6.3 vs 6.5;
podcasts 16.4 / 10.0 / 9.2 vs 19.0 / 12.6 / 8.8; earnings calls from Nigeria, China,
Brazil and Australia 14.0 / 16.2 / 22.2 / 18.2 vs 22.7 / 33.7 / 22.5 / 23.1; TED talks
5.0 / 7.2 vs 4.6 / 5.6; *Tears of Steel* 27.5 vs 28.1. Real-time load with a film playing
(`bench.cpu`): Parakeet-unified used 0.3 CPU cores across both processes and about 20% of
a GTX 1070; neither engine fell behind.

What that decided:

- **Parakeet-unified on the GPU is the default.** It is trained for streaming and offline
  use at once and punctuates and capitalises itself, so captions need no separate
  punctuation model. sherpa-onnx exports it as *buffered* streaming: every 240 ms the
  whole encoder re-reads 5.6 s of left context plus 160 ms of lookahead and decodes only
  the new frames. That is about 3x slower than real time on the CPU, so it needs CUDA:
  the fp32 export (2.5 GB, downloaded on first use) takes ~65 ms per step on a GTX 1070.
  The int8 export is slower on that GPU (its ops fall back to the CPU), and Pascal has no
  fast fp16. Shorter left context was faster but lost accuracy on film audio (*Tears of
  Steel*: 27.5% at 5.6 s, 33.7% at 3.2 s, 37.0% at 2.4 s).
- **It runs in a worker process** (`python -m live_subtitles unified-worker`), because
  onnxruntime-gpu and sherpa-onnx cannot share a process (see below), and so the decode
  loop stays off the UI's GIL. Its CPU sessions are capped and don't spin-wait.
- **A sentence's period arrives late.** The model emits it only when the next sentence
  starts, so the newest word stays "in progress" (shown, not committed) through a pause
  and the period attaches to it. Committing it on a timer left periods as words of their own.
- **The input leveller stays on** for both engines (slow AGC toward -23 dBFS, no boost for
  near-silence). For Parakeet-unified it took the China call from 22.0% to 16.2% WER and
  was neutral elsewhere.
- **The GPU engine is weaker on two synthetic scenes**: read speech under the loudest
  effects at 0 dB (28.7% vs 19.7%) and very quiet speech (12.5% vs 8.1%). On the real
  recordings it won or tied everywhere except the two TED talks. If a film's quiet
  dialogue goes missing, Settings → Recogniser → CPU is the comparison to try.
- **The CPU fallback is Nemotron 560 ms on 2 threads.** 160 ms was no more accurate and
  needed about 9 cores to stay ahead of real time (it stalled on one scene); 560 ms keeps
  up on 1.9 cores. Nemotron emits final tokens only, so its captions never rewrite.
  English uses the English-only model, which beat the multilingual one on English; other
  languages and auto-detect use Nemotron 3.5. Its punctuation comes from a separate model
  (sherpa-onnx's online CNN-BiLSTM, 31 MB), which may revise only the newest 4 words.
- **Caption lines never jump.** A word's line is fixed when it is placed; only words still
  in progress can move. The benchmark counts 0 line jumps and 0 word rewrites per minute
  for both engines (the earlier layout re-broke lines on every punctuation change).
- **Also measured and rejected:** a two-pass design (Nemotron live, Parakeet re-decoding
  each phrase after a pause) matched the GPU engine's accuracy but rewrote captions after
  every pause. Whistle 55M has no GPU runtime and its stream fell behind real time.
  Whisper large-v3-turbo took 0.75-1.25 s per pass on the GTX 1070. sherpa-onnx's CUDA
  build has no kernels for Pascal GPUs. VAD gates (Silero, TEN) dropped dialogue under music.
- **File mode uses Parakeet TDT 0.6B v3** in 20 s windows; it sees each scene whole.
- **Two ONNX Runtimes, separate processes.** sherpa-onnx bundles a CPU `onnxruntime.dll`,
  and the GPU engine and file mode need `onnxruntime-gpu`. Both DLLs share one name, so
  the tray app never imports onnxruntime: the GPU engine and file transcription each run
  in a child process.

## Benchmark

```powershell
.\.venv\Scripts\python.exe -m bench.testset              # synthetic scenes + Tears of Steel
uv run --no-project --with pyarrow --with huggingface_hub --with numpy python -m bench.realset fetch
.\.venv\Scripts\python.exe -m bench.realset build        # real-world scenes
.\.venv\Scripts\python.exe -m bench.run unified --label live-unified
.\.venv\Scripts\python.exe -m bench.run nemotron-560 --label live-nemotron-560
.\.venv\Scripts\python.exe -m bench.report live-unified live-nemotron-560
.\.venv\Scripts\python.exe -m bench.cpu gpu             # real-time CPU/GPU load
```

Scenes, audio and results live in `%LOCALAPPDATA%\Live-Subtitles\bench`.

- **Synthetic (5 x ~3 min):** LibriSpeech read speech, clean, over film music (+6 dB),
  over the loudest effects (0 dB), very quiet (-42 dBFS), and two overlapping speakers.
- **Real-world (11 scenes, ~70 min):** *Tears of Steel* (official subtitles); The Late
  Show's "Meanwhile" segments (OpenAI's Whisper long-form set: monologue, laughter, music
  stings); three Rev16 podcasts (narrative over music, two hosts, a live show); Earnings-22
  calls from Nigeria, China, Brazil and Australia; two TED-LIUM talks. Long recordings
  are cut to 5-minute excerpts at pauses where Parakeet's timed words agree with the
  human transcript, and scored against the human transcript.
- **Your own clips:** `python -m bench.realset local <folder>` turns every video or audio
  file with a same-named `.srt` beside it into a scene.

`bench.run` reports WER, sentence-end F1, delay from the end of each spoken word to it
appearing and to it being committed, rewrites of text already on screen, line jumps,
phantom words in silence, speed, CPU and memory. `bench.phrases` compares offline models
on the same phrase clips (it ruled out the second-pass candidates).

## Layout

```
live_subtitles/
  app.py              tray app: wiring, single instance, settings persistence
  capture.py          WASAPI loopback -> 16 kHz mono (soxr), centre-channel downmix
  live.py             recogniser thread: capture queue -> engine -> caption updates
  engines.py          CPU engine: Nemotron streaming (sherpa-onnx), input leveller
  unified.py          GPU engine: Parakeet-unified (onnxruntime-gpu) and its worker process
  models.py           model catalogue + first-use download (sherpa-onnx releases, Hugging Face)
  captions.py         words -> on-screen lines (roll-up / pop-on, sentence breaks)
  renderer.py         subtitle painting shared by overlay and preview
  overlay.py          the draggable, click-through, always-on-top window
  settings.py         settings dataclasses, JSON load/save, presets
  settings_window.py  settings UI with live preview
  hotkeys.py          global hotkeys (RegisterHotKey)
  tray.py             tray icon and menu
  jobs.py             runs file transcription in a child process, progress window
  transcribe.py       video -> Parakeet -> cues -> SRT/ASS/VTT (CLI entry point)
  subtitles.py        cue building (2x42 chars, reading speed) and file writers
run.pyw               tray launcher (pythonw, no console)
setup.ps1             builds .venv with uv
tests/test_core.py    unit tests for the pure-Python parts
bench/                live-caption benchmark: scenes, runner, report, real-time load
```

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Known limits

- Exclusive-fullscreen games and players draw over every window. Browser fullscreen and
  borderless players are fine.
- The overlay listens to one output device. If Windows switches the default device
  (say, headphones plugged in), stop and start live subtitles to follow it.
- The GPU engine needs an NVIDIA GPU with CUDA and about 3.2 GB of free VRAM. Without
  one, captions run on the CPU (Nemotron), which punctuates less well.
- A sentence's final period appears once the next sentence starts: the model can't know
  the sentence ended until then.

## Licence

The code is MIT-licensed (see `LICENSE`). The speech models are not part of this
repository; they download on first use under their own licences: NVIDIA's Parakeet and
Nemotron models under the terms on their Hugging Face model cards, sherpa-onnx's exports
and punctuation model under sherpa-onnx's.
