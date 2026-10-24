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

## Movies and TV

Live captions always trail the speech. For a movie or an episode the subtitles usually
already exist, timed by a person, so media mode shows those instead: whole lines that
appear when the line starts, the way subtitles are meant to be read.

It runs by itself while the app is open (tray menu → *Media subtitles for movies and
TV*, or Ctrl+Alt+V):

1. **What's playing.** The title of the browser or player window in front ("The Bear
   S02E04", "Dune: Part Two (2024)", a release file name in VLC) is looked up on TMDB, or
   on IMDb when no TMDB key is set. YouTube, Twitch and the like are ignored. When a page's
   title names nothing useful, right-click the subtitles → *What is this?* once; the
   answer is remembered for that site.
2. **The subtitles.** The cache first, then OpenSubtitles, Wyzie Subs, SubDL and Addic7ed
   (through Gestdown) in parallel, scored on downloads, uploader trust, sound
   descriptions and release, with machine translations ranked last. TV works with no
   account; movies need a key for one of the first three (all free; Settings → Movies &
   TV, stored DPAPI-encrypted).
3. **Sync.** The track is checked against what the speakers play. English audio: the
   media worker recognises ten-second stretches (Parakeet TDT, int8 on the CPU, about 0.7 s
   each) and matches rare three-word runs against the track's lines. Until that locks, the
   overlay keeps showing live captions, so there is never a gap. Other languages (Korean,
   Japanese, ...) with English subtitles: voice activity is cross-correlated with the
   track's cue spans instead, which takes a couple of minutes of speech; until then the
   overlay says "Syncing subtitles…" rather than guess.
4. **Keeping it synced.** Recognition keeps running. Frame-rate mismatches (a 25 fps
   track on 23.976 fps video drifts 4.3%), seeks and cut scenes all show up as anchors
   that disagree with the current mapping, and the mapping follows. A track whose words
   never match what is heard is the wrong episode or cut, and the next candidate is tried.
5. **No subtitles anywhere.** English: live captions carry on, and what they heard is
   saved as a track for next time. Other languages: Whisper medium translates each line
   to English as it finishes (a few seconds late), also saved. With the browser extension,
   which knows the stream's address, `media-generate` makes the track from the stream
   ahead of the playhead instead.
6. **Sound descriptions.** On by default. Tracks with them ([door creaks], speaker names)
   rank first; turning them off strips them from the same track. Tracks without them get
   sparse generated cues ([music], [laughter], [phone ringing]) from an AudioSet tagger
   (AST), placed only in gaps between lines.

Right-click the subtitles for 0.5 s earlier/later (saved per title; also Ctrl+Alt+[ and
Ctrl+Alt+]), *Wrong subtitles: try others* (Ctrl+Alt+W), and sound descriptions.
Everything is cached in `%LOCALAPPDATA%\Live-Subtitles\media.db`, so a rewatch starts
synced.

Measured on Tears of Steel (official SRT, deliberately mistimed by 23.4 s at PAL speed,
playback started a minute in): text anchors locked within the first lines of dialogue
and found the frame-rate ratio about two minutes in; from then on lines landed within
0.03-0.07 s. Speech-span sync on the same film locked after about two minutes of speech,
0.19 s off.

### The browser extension (optional)

`extension\` is a small Chrome/Edge extension (chrome://extensions → Developer mode →
Load unpacked). It reports the page's video position, pause and seeks to the app on
127.0.0.1:47219, plus the stream and subtitle addresses the player requested. With it,
timing is exact from the first second and survives seeks instantly, and tracks can be
generated ahead of playback. Without it everything else still works from the audio alone.

Generating a track from a file (or a stream URL) without the app:

```powershell
.\.venv\Scripts\python.exe -m live_subtitles media-generate "D:\Movies\film.mkv" --start 600 --sounds
```

## Models, and why

**English live captions run NVIDIA Parakeet-unified 0.6B on the GPU** (`unified.py`),
with **Nemotron streaming 560 ms on the CPU** as the fallback when there is no usable
NVIDIA GPU, and for every other language. The status line says which one is running.
Files are transcribed by Parakeet TDT 0.6B v3.

The choice comes from the benchmark below (`bench/`): 16 scenes, about 80 minutes of
audio, each run through the app's own caption pipeline on a simulated real-time clock.

| Live engine (English) | WER, real-world | WER, synthetic | Sentence F1 | Words on screen p50 / p90 | Speed | CPU |
|---|---|---|---|---|---|---|
| **Parakeet-unified, GPU** (default) | 12.1% | 16.3% | **0.60** | 0.36 / 0.61 s | 0.28x real time | **0.8 cores** |
| Nemotron streaming 560 ms, CPU (fallback) | **11.4%** | **15.0%** | 0.40 | 0.40 / 0.81 s | 0.21x | 9.4 cores at 4 threads; 1.9 at 2 |
| Nemotron streaming 160 ms, CPU | 12.1% | 15.1% | 0.37 | **0.21 / 0.54 s** | 0.44x | 9.3 cores |

WER is scored on Whisper's English normalisation of both texts, as the Open ASR
Leaderboard does. Until 2026-10-18 the benchmark scored without it, which counted
Nemotron's spelled-out numbers ("twenty twenty one" for "2021") as errors and made
Parakeet-unified look well ahead (13.8% vs 17.0%); `bench.run` still reports that older
score as `wer_plain`.

Per real-world scene (WER %, Parakeet-unified vs Nemotron 560): TV monologue 6.3 vs 6.2;
podcasts 14.9 / 9.3 / 8.4 vs 16.6 / 9.8 / 8.5; earnings calls from Nigeria, China,
Brazil and Australia 11.1 / 12.5 / 19.0 / 14.5 vs 10.4 / 12.5 / 18.5 / 11.9; TED talks
4.5 / 6.9 vs 3.3 / 3.7; *Tears of Steel* 25.6 vs 24.0. Scored on the nine cut recordings
whole instead (6.5 hours, no excerpt cuts), the result holds: Parakeet-unified 11.7% vs
Nemotron 10.8%, Nemotron ahead on all but the Crimetown podcast. Most of Parakeet-unified's
extra errors are dropped words (more deletions on 8 of the 9).

Those dropped words are disfluencies, not lost speech. The references are verbatim;
Nemotron keeps stutters and fillers, and Parakeet-unified writes clean text: "i was [i was]
out in", "until we [until we] moved", "[you know]". Most of its extra deletions are
repeats, "you know" and false starts, spread through phrases rather than before pauses.
With deleted repeats and "you know"/"i mean" not counted (`wer_clean`), the uncut
recordings tie: 10.2% vs 10.3%. Its raw decoder output scores the same as the caption
pipeline's on every scene, so the pipeline's in-progress rule loses nothing.

Real-time load with a film playing
(`bench.cpu`): Parakeet-unified used 0.3 CPU cores across both processes and about 20% of
a GTX 1070; neither engine fell behind.

What that decided:

- **Parakeet-unified on the GPU is the default.** Its accuracy is close to Nemotron's
  (12.1% vs 11.4% WER, a tie once its cleaned-up disfluencies are excused) rather than better; it wins on punctuation, on the slowest words
  (p90 delay) and on leaving the CPU free. It is trained for streaming and offline
  use at once and punctuates and capitalises itself, so captions need no separate
  punctuation model. sherpa-onnx exports it as *buffered* streaming: every 240 ms the
  whole encoder re-reads 5.6 s of left context plus 160 ms of lookahead and decodes only
  the new frames. That is about 3x slower than real time on the CPU, so it needs CUDA:
  the fp32 export (2.5 GB, downloaded on first use) takes ~65 ms per step on a GTX 1070.
  The int8 export is slower on that GPU (its ops fall back to the CPU), and Pascal has no
  fast fp16. Shorter left context was faster but lost accuracy on film audio (*Tears of
  Steel*: 25.6% at 5.6 s, 30.3% at 3.2 s, 33.9% at 2.4 s).
- **It runs in a worker process** (`python -m live_subtitles unified-worker`), because
  onnxruntime-gpu and sherpa-onnx cannot share a process (see below), and so the decode
  loop stays off the UI's GIL. Its CPU sessions are capped and don't spin-wait.
- **A sentence's period arrives late.** The model emits it only when the next sentence
  starts, so the newest word stays "in progress" (shown, not committed) through a pause
  and the period attaches to it. Committing it on a timer left periods as words of their own.
- **The input leveller stays on** for both engines (slow AGC toward -23 dBFS, no boost for
  near-silence). It helps Parakeet-unified slightly on very quiet speech (11.3% vs 11.9%
  WER) and is neutral elsewhere, for both engines.
- **The GPU engine is weaker on two synthetic scenes**: read speech under the loudest
  effects at 0 dB (27.1% vs 20.1%) and very quiet speech (11.3% vs 9.0%). On the real
  recordings the two are close: Nemotron is ahead on 7 of 11 scenes, Parakeet-unified on
  the 3 podcasts, and the China call is a tie. If a film's quiet dialogue goes missing,
  Settings → Recogniser → CPU is the comparison to try.
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
  each phrase after a pause) was about as accurate (11.5-13.2%) but rewrote captions after
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
  human transcript, and scored against the human transcript. `bench.realset build --full`
  builds the same recordings uncut (`full-*` scenes, about 6.5 hours), to check that the
  cuts don't favour one engine.
- **Your own clips:** `python -m bench.realset local <folder>` turns every video or audio
  file with a same-named `.srt` beside it into a scene.

`bench.run` reports WER (Whisper-normalised; `bench.report --rescore` re-scores saved
results after a scorer change; `wer_clean` excuses deleted disfluencies), sentence-end F1, delay from the end of each spoken word to it
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
  media/              movies and TV: pre-timed subtitles synced from the audio
    controller.py     the app side: window title -> track -> sync thread -> overlay lines
    titles.py         window/page titles -> title, year, season, episode
    tmdb.py           TMDB lookups (and IMDb's keyless suggestions)
    sources.py        OpenSubtitles, Wyzie, SubDL, Gestdown; scoring
    subfile.py        SRT/VTT/ASS parsing, encodings, zips, sound descriptions
    align.py          text anchors, speech-span correlation, the time map (pure numpy)
    clock.py          audio clock (counts playing samples) and the extension's clock
    store.py          media.db: tracks, nudges, rejections, remembered titles
    worker.py         media-worker process: Parakeet TDT, Silero VAD, Whisper, AST
    generate.py       media-generate: subtitles from a file or stream, ahead of playback
    tagger.py         AudioSet sound events (AST) for generated sound descriptions
    web.py            127.0.0.1 endpoint for the extension
    dialogs.py        the "What is this?" window
    winwatch.py       foreground window title and program (Win32)
    secrets.py        DPAPI encryption for API keys
extension/            optional Chrome/Edge extension (video position, stream URLs)
run.pyw               tray launcher (pythonw, no console)
setup.ps1             builds .venv with uv
tests/test_core.py    unit tests for the pure-Python parts
tests/test_media.py   media mode: titles, subtitle files, alignment, clock, cache
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
- Media mode without the extension learns the playback position from the audio, so it
  needs a few lines of dialogue (English) or a couple of minutes of speech (other
  languages) to lock, and again after a seek. Other sounds playing at the same time (a
  notification, a second tab) briefly confuse the audio clock until the next lock.
- Media mode only works while this PC plays the audio (a browser or player on this
  desktop), since the audio is what it syncs to.
- Generated translations (no English track anywhere) are Whisper medium's, a few seconds
  behind live; a person's subtitles are far better when they exist.

## Licence

The code is MIT-licensed (see `LICENSE`). `bench/whisper_normalizer/` is OpenAI
Whisper's English text normaliser, MIT-licensed by OpenAI (its own `LICENSE` is beside
it). The speech models are not part of this
repository; they download on first use under their own licences: NVIDIA's Parakeet and
Nemotron models under the terms on their Hugging Face model cards, sherpa-onnx's exports
and punctuation model under sherpa-onnx's.
