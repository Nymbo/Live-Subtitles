"""Builds the live-caption benchmark scenes (once) in %LOCALAPPDATA%\\Live-Subtitles\\bench.

    .\\.venv\\Scripts\\python.exe -m bench.testset [--tos-dir DIR]

Scenes, all 16 kHz mono WAV with a JSON sidecar (reference text, speech intervals, and
reference word times):

- tos      Tears of Steel (Blender Foundation, CC-BY), the whole 12-minute film: dialogue
           under score, gunfire and machinery, scored against the official English subtitles.
- clean    LibriSpeech dev-clean utterances with natural gaps and one 8 s silence.
- music    the same speech over Tears of Steel's music and effects at +6 dB SNR.
- action   the same over the loudest effects at 0 dB SNR.
- quiet    speech at about -42 dBFS under a faint bed (+10 dB SNR): whispered/low dialogue.
- overlap  two speakers talking over each other (the second 3 dB down).

Reference word times come from Parakeet TDT 0.6B v3 run offline on the *clean* speech
(and on the film), so caption delay can be measured per word. Only words both Parakeet
and the system under test agree on are timed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np

from live_subtitles.paths import DATA_DIR

SR = 16000
BENCH_DIR = DATA_DIR / "bench"
TOS_MOV = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
TOS_SRT = "https://download.blender.org/demo/movies/ToS/subtitles/TOS-en.srt"
LIBRI_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing/librispeech_asr_dummy"
              "&config=clean&split=validation&offset={offset}&length=100")
SCENE_S = 180.0
SEED = 7


# -- audio io ------------------------------------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def ffmpeg_decode(src: str | Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


# -- sources -------------------------------------------------------------------------
def srt_cues(path: Path) -> list[dict]:
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig")):
        rows = [r for r in block.strip().splitlines() if r.strip()]
        if len(rows) < 3 or "-->" not in rows[1]:
            continue
        a, b = (_srt_time(t) for t in rows[1].split("-->"))
        # Drop styling tags and SDH sound cues ("<i>", "[door slams]", "(LAUGHS)").
        text = re.sub(r"<[^>]+>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)", " ", " ".join(rows[2:]))
        text = re.sub(r"^[A-Z][A-Z .']+:\s*", "", text.strip())     # "NARRATOR: ..."
        if text.strip():
            cues.append({"start": a, "end": b, "text": " ".join(text.split())})
    return cues


def _srt_time(t: str) -> float:
    h, m, s = t.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def ensure_tos(tos_dir: Path | None) -> tuple[np.ndarray, list[dict]]:
    wav, srt = BENCH_DIR / "src" / "tos16k.wav", BENCH_DIR / "src" / "TOS-en.srt"
    wav.parent.mkdir(parents=True, exist_ok=True)
    if not srt.exists():
        if tos_dir and (tos_dir / "TOS-en.srt").exists():
            shutil.copy(tos_dir / "TOS-en.srt", srt)
        else:
            urllib.request.urlretrieve(TOS_SRT, srt)
    if not wav.exists():
        if tos_dir and (tos_dir / "tos16k.wav").exists():
            shutil.copy(tos_dir / "tos16k.wav", wav)
        else:
            mov = BENCH_DIR / "src" / "tos_720p.mov"
            if not mov.exists():
                print("Downloading Tears of Steel (365 MB)...", flush=True)
                urllib.request.urlretrieve(TOS_MOV, mov)
            write_wav(wav, ffmpeg_decode(mov))
    return read_wav(wav), srt_cues(srt)


def ensure_libri() -> list[dict]:
    """73 LibriSpeech dev-clean utterances: [{"audio", "text"}]."""
    cache = BENCH_DIR / "src" / "libri"
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "index.json"
    if not index.exists():
        rows = []
        for offset in (0, 100):
            with urllib.request.urlopen(LIBRI_ROWS.format(offset=offset), timeout=60) as r:
                rows += json.load(r)["rows"]
        items = []
        for row in rows:
            r = row["row"]
            dst = cache / f"{r['id']}.wav"
            if not dst.exists():
                tmp = cache / f"{r['id']}.flac"
                urllib.request.urlretrieve(r["audio"][0]["src"], tmp)
                write_wav(dst, ffmpeg_decode(tmp))
                tmp.unlink()
            items.append({"id": r["id"], "speaker": r["speaker_id"], "text": r["text"].lower()})
        index.write_text(json.dumps(items, indent=1), encoding="utf-8")
    items = json.loads(index.read_text(encoding="utf-8"))
    for it in items:
        it["audio"] = read_wav(cache / f"{it['id']}.wav")
    return items


def film_bed(tos: np.ndarray, cues: list[dict], loud: bool) -> np.ndarray:
    """Tears of Steel audio from outside every subtitle cue (music, effects, ambience)."""
    keep = np.ones(tos.size, bool)
    for c in cues:
        keep[max(0, int((c["start"] - 0.6) * SR)):int((c["end"] + 0.6) * SR)] = False
    keep[:int(15 * SR)] = False                       # the silent logo intro
    bed = tos[keep]
    if loud:   # the loudest second-long stretches: gunfire, explosions, machinery
        secs = bed[: bed.size // SR * SR].reshape(-1, SR)
        level = np.sqrt((secs ** 2).mean(axis=1))
        bed = secs[np.sort(np.argsort(level)[-int(len(secs) * 0.4):])].reshape(-1)
    return bed


# -- scene assembly ------------------------------------------------------------------
def lay_out(utts: list[dict], rng: np.random.Generator, overlap: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Place utterances on a timeline with gaps; returns the speech stem and segments."""
    stem = np.zeros(int(SCENE_S * SR) + SR * 40, np.float32)
    segs, t, i = [], 1.5, 0
    silence_done = False
    while t < SCENE_S and i < len(utts):
        u = utts[i]
        a = u["audio"] * (10 ** ((-26 - rms_db(u["audio"])) / 20))
        start = int(t * SR)
        stem[start:start + a.size] += a
        segs.append({"start": t, "end": t + a.size / SR, "text": u["text"], "speaker": u["speaker"]})
        i += 1
        if overlap and i < len(utts):
            b = utts[i]["audio"]
            b = b * (10 ** ((-29 - rms_db(b)) / 20))
            t2 = t + a.size / SR * 0.6
            s2 = int(t2 * SR)
            stem[s2:s2 + b.size] += b
            segs.append({"start": t2, "end": t2 + b.size / SR, "text": utts[i]["text"], "speaker": utts[i]["speaker"]})
            i += 1
            t = max(t + a.size / SR, t2 + b.size / SR) + rng.uniform(0.8, 2.0)
            continue
        t += a.size / SR + rng.uniform(0.6, 2.0)
        if not silence_done and t > SCENE_S / 2:
            t += 8.0          # a long quiet stretch: nothing should be captioned here
            silence_done = True
    end = int((max(s["end"] for s in segs) + 2.0) * SR)
    return stem[:end], segs


def mix(stem: np.ndarray, bed: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(0, max(1, bed.size - stem.size)))
    b = np.resize(bed[start:], stem.size)
    speech = stem[np.abs(stem) > 1e-4]
    gain = 10 ** ((rms_db(speech) - snr_db - rms_db(b)) / 20)
    return stem + b * gain


def ref_word_times(audio: np.ndarray) -> list[dict]:
    """Parakeet TDT v3 word times on clean audio (20 s windows, 2 s overlap)."""
    import onnx_asr
    import onnxruntime as ort

    ort.preload_dlls()
    model = ref_word_times.model = getattr(ref_word_times, "model", None) or onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]).with_timestamps()
    words: list[dict] = []
    win, hop = 20 * SR, 18 * SR
    for off in range(0, max(1, audio.size - 2 * SR), hop):
        res = model.recognize(audio[off:off + win], sample_rate=SR)
        lo = off / SR + (1.0 if off else 0.0)          # each window owns [lo, hi)
        hi = off / SR + (19.0 if off + win < audio.size else 1e9)
        cur: dict | None = None
        for tok, ts in zip(res.tokens or [], res.timestamps or []):
            t = off / SR + ts
            if tok.startswith(" ") or cur is None:
                if cur and lo <= cur["start"] < hi:
                    words.append(cur)
                cur = {"word": tok.strip(), "start": t, "end": t + 0.25}
            else:
                cur["word"] += tok
                cur["end"] = t + 0.25
        if cur and lo <= cur["start"] < hi:
            words.append(cur)
    return [w for w in words if w["word"]]


def save(name: str, audio: np.ndarray, segs: list[dict], words: list[dict], kind: str) -> None:
    write_wav(BENCH_DIR / f"{name}.wav", audio)
    (BENCH_DIR / f"{name}.json").write_text(json.dumps({
        "name": name, "kind": kind, "duration": audio.size / SR, "segments": segs, "ref_words": words,
    }, indent=1), encoding="utf-8")
    print(f"{name}: {audio.size / SR:.0f} s, {len(segs)} segments, {len(words)} timed words", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tos-dir", type=Path, help="folder already holding tos16k.wav and TOS-en.srt")
    args = ap.parse_args()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    tos, cues = ensure_tos(args.tos_dir)
    save("tos", tos, cues, ref_word_times(tos), "film")

    utts = ensure_libri()
    order = rng.permutation(len(utts))
    utts = [utts[i] for i in order]
    stem, segs = lay_out(utts, rng)
    words = ref_word_times(stem)
    save("clean", stem, segs, words, "synthetic")
    save("music", mix(stem, film_bed(tos, cues, loud=False), 6.0, rng), segs, words, "synthetic")
    save("action", mix(stem, film_bed(tos, cues, loud=True), 0.0, rng), segs, words, "synthetic")
    quiet = stem * (10 ** (-16 / 20))                 # -26 -> -42 dBFS
    save("quiet", mix(quiet, film_bed(tos, cues, loud=False), 10.0, rng), segs, words, "synthetic")

    o_stem, o_segs = lay_out(list(reversed(utts)), rng, overlap=True)
    save("overlap", o_stem, o_segs, ref_word_times(o_stem), "synthetic")


if __name__ == "__main__":
    main()
