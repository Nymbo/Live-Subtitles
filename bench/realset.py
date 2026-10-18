r"""Real-world benchmark scenes: TV, podcasts, accented calls and talks, with human transcripts.

Two steps, because the sources are parquet files and the app venv has no pyarrow:

    uv run --no-project --with pyarrow --with huggingface_hub --with numpy python -m bench.realset fetch
    .\\.venv\\Scripts\\python.exe -m bench.realset build

`fetch` downloads the recordings into %LOCALAPPDATA%\\Live-Subtitles\\bench\\src\\real.
`build` cuts each into an excerpt and writes scenes next to the synthetic ones:

- tv-*        The Late Show "Meanwhile" segments (OpenAI's Whisper long-form test set):
              a comedy monologue over audience laughter, applause and music stings.
- pod-*       Rev16 podcasts: conversation and crosstalk, narrative with a music bed, a live show.
- call-*      Earnings-22: earnings calls with speakers from Nigeria, China, Brazil, Australia.
- talk-*      TED-LIUM long-form: talks in a hall, with audience laughter.

Your own clips make the best test of all:

    .\.venv\Scripts\python.exe -m bench.realset local D:\Clips [--start 600 --length 600]

turns every video or audio file in the folder that has a same-named .srt beside it into a
scene `own-<name>`, scored against those subtitles (an excerpt when --length is given).

The transcripts have no timestamps, so `build` runs Parakeet TDT 0.6B v3 offline over the
whole recording, aligns its timed words with the human transcript, and cuts each excerpt
at a pause where the two agree. The excerpt's reference is the human text (scored), and
Parakeet's word times (where it agrees with the reference) give the delay timing.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import subprocess

import numpy as np

from live_subtitles.paths import DATA_DIR

SR = 16000
BENCH_DIR = DATA_DIR / "bench"
SRC = BENCH_DIR / "src" / "real"
EXCERPT_S = 300.0

# name: (repo, parquet path, row selector, excerpt start s)
SOURCES = {
    "pod-crimetown": ("rev16", "whisper_subset/test/0000.parquet", ("file_number", "14"), 0.0),
    "pod-podsave": ("rev16", "whisper_subset/test/0000.parquet", ("file_number", "17"), 240.0),
    "pod-murder-live": ("rev16", "whisper_subset/test/0000.parquet", ("file_number", "24"), 300.0),
    "call-nigeria": ("earnings22", "full/test/0001.parquet", ("file_id", "MP3-A"), 600.0),
    "call-china": ("earnings22", "full/test/0000.parquet", ("file_id", "4474955"), 200.0),
    "call-brazil": ("earnings22", "full/test/0000.parquet", ("file_id", "4467717"), 600.0),
    "call-australia": ("earnings22", "full/test/0000.parquet", ("file_id", "4481952"), 600.0),
    "talk-mead": ("tedlium-long-form", "default/test/0000.parquet", ("index", 5), 60.0),
    "talk-cameron": ("tedlium-long-form", "default/test/0000.parquet", ("index", 7), 300.0),
}
MEANWHILE = ("meanwhile", "default/test/0000.parquet")
MEANWHILE_ROWS = range(0, 64, 6)       # 11 segments, about 10 minutes


def _decode(data: bytes) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768


def _write_wav(path, audio: np.ndarray) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


# -- fetch (uv env with pyarrow) ------------------------------------------------------
def fetch() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    SRC.mkdir(parents=True, exist_ok=True)

    def table(repo: str, path: str, cols: list[str] | None = None):
        f = pq.ParquetFile(fs.open(f"datasets/distil-whisper/{repo}@~parquet/{path}"))
        return f, f.read(columns=cols)

    by_file: dict[tuple[str, str], list[str]] = {}
    for name, (repo, path, _, _) in SOURCES.items():
        if not (SRC / f"{name}.wav").exists():
            by_file.setdefault((repo, path), []).append(name)
    for (repo, path), names in by_file.items():
        f, meta = table(repo, path, [c for c in pq.ParquetFile(fs.open(
            f"datasets/distil-whisper/{repo}@~parquet/{path}")).schema_arrow.names if c != "audio"])
        rows = meta.to_pylist()
        starts = np.cumsum([0] + [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)])
        for name in names:
            key, val = SOURCES[name][2]
            idx = val if key == "index" else next(i for i, r in enumerate(rows) if r[key] == val)
            g = int(np.searchsorted(starts, idx, side="right") - 1)
            print(f"{name}: row {idx} (row group {g})...", flush=True)
            grp = f.read_row_group(g, columns=["audio"])
            audio = _decode(grp.column("audio")[idx - int(starts[g])].as_py()["bytes"])
            _write_wav(SRC / f"{name}.wav", audio)
            text = rows[idx].get("transcription") or rows[idx].get("text")
            (SRC / f"{name}.txt").write_text(text, encoding="utf-8")
            print(f"  {audio.size / SR / 60:.1f} min", flush=True)

    if not (SRC / "tv-meanwhile.wav").exists():
        f, t = table(*MEANWHILE)
        rows = t.to_pylist()
        clips, texts = [], []
        for i in MEANWHILE_ROWS:
            clips.append(_decode(rows[i]["audio"]["bytes"]))
            texts.append(rows[i]["text"])
        gap = np.zeros(int(1.5 * SR), np.float32)
        audio = np.concatenate([x for c in clips for x in (c, gap)])
        _write_wav(SRC / "tv-meanwhile.wav", audio)
        (SRC / "tv-meanwhile.txt").write_text("\n".join(texts), encoding="utf-8")
        print(f"tv-meanwhile: {len(clips)} segments, {audio.size / SR / 60:.1f} min", flush=True)


# -- build (app venv: Parakeet on the GPU) ---------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)          # <unk:rock_music_intro>, <inaudible>
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_one(name: str, start_s: float, length_s: float) -> None:
    from bench import metrics as M
    from bench.testset import read_wav, ref_word_times, save

    audio = read_wav(SRC / f"{name}.wav")
    ref_raw = clean_text((SRC / f"{name}.txt").read_text(encoding="utf-8")).split()
    ref = [(M.norm_word(w), w) for w in ref_raw]
    ref = [(n, w) for n, w in ref if n]
    lo = max(0.0, start_s - 30)
    hi = min(audio.size / SR, start_s + length_s + 60)
    words = ref_word_times(audio[int(lo * SR):int(hi * SR)])
    for w in words:
        w["start"] += lo
        w["end"] += lo
    hyp = [M.norm_word(w["word"]) for w in words]
    sm = difflib.SequenceMatcher(None, [n for n, _ in ref], hyp, autojunk=False)
    h2r = {}
    for a, b, size in sm.get_matching_blocks():
        if size < 3:          # lone matches ("the", "and") can land anywhere in an hour of text
            continue
        for k in range(size):
            h2r[b + k] = a + k

    def cut(after_s: float) -> int:
        """The first hyp word after `after_s` that follows a pause (>= 0.4 s) and, with the
        word before it, agrees with the reference: a clean place to start or end."""
        for j in range(1, len(words)):
            if (words[j]["start"] >= after_s and j in h2r and j - 1 in h2r
                    and h2r[j] == h2r[j - 1] + 1 and words[j]["start"] - words[j - 1]["end"] + 0.25 >= 0.4):
                return j
        return len(words)

    whole = length_s >= 1e8
    j0 = 0 if start_s <= 0 and 0 in h2r else cut(start_s)
    if whole:
        j1, r1 = len(words), len(ref)
    else:
        j1 = cut(words[j0]["start"] + length_s)
        if j1 >= len(words):        # no clean pause: end on the last word both agree on
            j1 = max(j for j in h2r if j > j0) + 1
            r1 = h2r[j1 - 1] + 1
        else:
            r1 = h2r[j1]
    r0 = h2r[j0] if j0 in h2r else 0
    t0 = max(0.0, words[j0]["start"] - 0.6)
    t1 = hi if whole else words[j1 - 1]["end"] + 0.6
    clip = audio[int(t0 * SR):int(t1 * SR)]

    # Per-reference-word times: matched words take Parakeet's, the rest interpolate.
    times = np.full(len(ref), np.nan)
    for j, r in h2r.items():
        if j0 <= j < j1:
            times[r] = words[j]["start"] - t0
    idx = np.arange(r0, r1)
    known = idx[~np.isnan(times[r0:r1])]
    rt = np.interp(idx, known, times[known]) if len(known) else idx * 0.0
    segs, cur = [], []
    for k, i in enumerate(idx):
        cur.append((ref[i][1], rt[k]))
        pause = k + 1 < len(idx) and rt[k + 1] - rt[k] - 0.35 >= 0.5
        if pause or re.search(r"[.!?]['\")]*$", ref[i][1]) or len(cur) >= 25 or k + 1 == len(idx):
            segs.append({"start": float(cur[0][1]), "end": float(cur[-1][1] + 0.4),
                         "text": " ".join(w for w, _ in cur)})
            cur = []
    ref_words = [{"word": w["word"], "start": w["start"] - t0, "end": w["end"] - t0}
                 for j, w in enumerate(words) if j0 <= j < j1 and j in h2r]
    save(name, clip, segs, ref_words, "real")


def local(folder: str, start_s: float, length_s: float) -> None:
    from pathlib import Path

    from bench.testset import ffmpeg_decode, ref_word_times, save, srt_cues

    for srt in sorted(Path(folder).glob("*.srt")):
        media = next((m for m in srt.parent.glob(srt.stem + ".*") if m.suffix.lower() != ".srt"), None)
        if media is None:
            continue
        audio = ffmpeg_decode(media)
        end = start_s + length_s if length_s else audio.size / SR
        clip = audio[int(start_s * SR):int(end * SR)]
        cues = [{**c, "start": c["start"] - start_s, "end": c["end"] - start_s} for c in srt_cues(srt)
                if c["start"] >= start_s and c["end"] <= end]
        name = "own-" + re.sub(r"[^a-z0-9]+", "-", srt.stem.lower()).strip("-")
        save(name, clip, cues, ref_word_times(clip), "film")


def build(names: list[str] | None) -> None:
    for name in names or [*SOURCES, "tv-meanwhile"]:
        if name == "tv-meanwhile":
            build_one(name, 0.0, 1e9)
        else:
            build_one(name, SOURCES[name][3], EXCERPT_S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fetch", "build", "local"))
    ap.add_argument("names", nargs="*", help="build: scenes to build (default all); local: the folder")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--length", type=float, default=0.0)
    args = ap.parse_args()
    if args.step == "fetch":
        fetch()
    elif args.step == "build":
        build(args.names)
    else:
        local(args.names[0], args.start, args.length)


if __name__ == "__main__":
    main()
