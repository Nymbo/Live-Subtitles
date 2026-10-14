# Live-Subtitles

On-screen subtitles for whatever Windows is playing (a stream, a video, a call), drawn
in a draggable, fully styleable overlay. Also writes real subtitle files (SRT, ASS, WebVTT)
for video files on disk, for VLC and other players.

A tray app, not a Gradio Space. Windows only (WASAPI loopback capture).

## Run

```powershell
cd P:\Repos\HF-Dev
.\scripts\setup-app-env.ps1 -AppName "Live-Subtitles"     # once
.\scripts\launch-app.ps1 -AppName "Live-Subtitles"         # tray app
```

No console window: `.\Live-Subtitles\.venv\Scripts\pythonw.exe .\Live-Subtitles\run.pyw`
(run from `Live-Subtitles\`). A second launch opens the running copy's settings.

Subtitle files without the GUI:

```powershell
cd P:\Repos\HF-Dev\Live-Subtitles
.\.venv\Scripts\python.exe -m live_subtitles transcribe "D:\Movies\film.mkv" --formats srt,ass
```

ffmpeg and ffprobe must be on PATH. Files land next to the video as `film.srt`, which VLC
loads automatically. An existing `film.srt` is never overwritten unless you ask; the new
file becomes `film.live-subtitles.srt` instead.

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
  for 5.1/7.1, language (English, auto-detect, or 30 locales), latency, and CPU threads.
- **Video files:** the `.ass` output carries the overlay's look (font, colours,
  outline/box, shadow), so VLC shows the same style as the live subtitles.

Settings live in `%APPDATA%\Live-Subtitles\settings.json`. Models and logs live in
`%LOCALAPPDATA%\Live-Subtitles\`.

## Models, and why

The models were chosen by a bake-off on *Tears of Steel* (Blender Foundation, CC-BY;
12 minutes of dialogue under music and effects), scored against its official English
subtitles (335 words). Small reference, so treat differences of 2-3 points as noise.

| Setup | WER | Speed | Word delay p50 / p90 |
|---|---|---|---|
| **Live: Nemotron EN streaming 160 ms** (default, "Fastest") | **30.8%** | RTF ~0.4 on CPU | **0.32 / 0.32 s** |
| Nemotron EN streaming 560 ms ("Balanced") | 28.1% | RTF 0.15 on CPU | 0.56 / 0.72 s |
| Nemotron EN streaming 1120 ms | 27.5% | RTF 0.07 | 0.96 / 1.28 s |
| Nemotron 3.5 multilingual, en-US, 560 / 1120 ms | 34.0 / 31.6% | | |
| Parakeet v3 live (rolling re-decode on GPU) | 27.8-31.6% | GPU 65-140% busy | 1.1 / 1.9 s |
| **Files: Parakeet TDT 0.6B v3, 20 s windows** (this app) | **24.2%** | 12 min film in ~30 s | offline |
| Parakeet v3, 60-120 s windows | 30-31% | | offline |
| Parakeet v3 behind Silero VAD (default / 0.3) | 44 / 41% | | |

What that decided:

- **Live punctuation comes from a separate model** for English (sherpa-onnx's online
  CNN-BiLSTM punctuation and casing model, 31 MB). Against the official subtitles,
  Nemotron's own sentence-end punctuation scored F1 0.22; re-punctuating the last 40
  words scored 0.68-0.76. Recent words get revised as context arrives. The newest word
  gets no period until speech pauses for 0.8 s, and the first words of the 40-word
  window are context only, so they don't get capitalised as if they started a sentence.
  Shorter utterance gaps (1.0-1.5 s) were also tried; they lost words and did not
  punctuate better.
- **Live mode defaults to the 160 ms model.** Words appear about 0.3 s after they are
  spoken, for ~2 points more WER than 560 ms, which stays in Settings as "Balanced".
- **Live mode uses NVIDIA's cache-aware streaming Nemotron** through sherpa-onnx. It
  matches Parakeet's live accuracy at under half the delay, on about a tenth of the
  compute, on the CPU, so the GPU stays free for video. It emits final tokens only,
  so captions never rewrite themselves. English uses the English-only model, which beat
  the multilingual one on English. Other languages and auto-detect use Nemotron 3.5.
- **File mode uses Parakeet TDT 0.6B v3.** It sees each scene whole, which makes it the
  most accurate option. Windows longer than ~30 s drop speech on film audio.
- **No VAD anywhere.** Silero discards dialogue that has music underneath it.
- **Utterance boundaries are handled carefully** (`live.py`). An endless stream loses all
  punctuation and capitals, so the recogniser starts a new utterance after 2 s of
  decoded audio with no token. Each new stream is seeded with the last 1.5 s of audio,
  so a word starting at the boundary keeps its onset. sherpa's built-in endpointing
  (0.8 s gap, forced length cuts) lost words mid-speech: 37% WER.
- **Two ONNX Runtimes, two processes.** sherpa-onnx bundles a CPU `onnxruntime.dll`, and
  file mode needs `onnxruntime-gpu`. Both DLLs share one name, so the tray app never
  imports onnx-asr; files are transcribed by a child process.

## Layout

```
live_subtitles/
  app.py              tray app: wiring, single instance, settings persistence
  capture.py          WASAPI loopback -> 16 kHz mono (soxr), centre-channel downmix
  live.py             Nemotron streaming recogniser thread (sherpa-onnx)
  models.py           live model catalogue + first-use download from sherpa-onnx releases
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
tests/test_core.py    unit tests for the pure-Python parts
```

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Known limits

- Exclusive-fullscreen games and players draw over every window. Browser fullscreen and
  borderless players are fine.
- The overlay listens to one output device. If Windows switches the default device
  (say, headphones plugged in), stop and start live subtitles to follow it.
- Nemotron sometimes runs long dialogue together without punctuation. Lines still break
  at the width limit.
