"""Live-caption benchmark: runs an engine through the app's caption pipeline on a simulated clock.

    .\\.venv\\Scripts\\python.exe -m bench.run unified --label live-unified        # the GPU engine
    .\\.venv\\Scripts\\python.exe -m bench.run nemotron-560 --scenes clean,tos     # the CPU engine
    .\\.venv\\Scripts\\python.exe -m bench.report live-unified live-nemotron-560

Audio is offered in 50 ms packets at the times WASAPI would deliver them. The engine
processes whatever has arrived whenever it is free, exactly as `live.LiveRecognizer`
drains its queue, and each call's measured compute time advances the clock, so a slow
engine falls behind here as it would on screen. Caption-builder work (the punctuation
model runs there) is timed too and delays the display.

Per scene it reports WER, caption delay (reference word end -> word first on screen, and
-> word committed), sentence-end punctuation F1, rewrites of text already on screen,
words that jumped lines, words captioned where nobody speaks, CPU use and peak memory.
Build the scenes first with `python -m bench.testset` (synthetic + Tears of Steel) and
`python -m bench.realset` (real-world recordings; see its docstring).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

from bench import metrics as M
from bench.testset import BENCH_DIR, SR, read_wav
from live_subtitles import captions
from live_subtitles.captions import CaptionBuilder

PACKET_S = 0.05
TICK_S = 0.1
SYNTHETIC = ("clean", "music", "action", "quiet", "overlap")
REAL = ("tos", "tv-meanwhile", "pod-crimetown", "pod-podsave", "pod-murder-live", "call-nigeria", "call-china",
        "call-brazil", "call-australia", "talk-mead", "talk-cameron")
SCENES = SYNTHETIC + REAL
RESULTS = BENCH_DIR / "results"


def make_engine(spec: str, args):
    kind, _, opt = spec.partition("-")
    if kind == "nemotron":
        from live_subtitles import models
        from live_subtitles.engines import NemotronEngine

        model = models.live_model("en", int(opt or 160))
        if not model.is_installed():
            models.download(model)
        return NemotronEngine(model, threads=args.threads, leveler=not args.no_leveler)
    if kind == "unified":
        # Parakeet-unified on the GPU (live_subtitles/unified.py), in this process.
        # "unified" or "unified-<left context ms>".
        from live_subtitles.models import UNIFIED_MODEL
        from live_subtitles.unified import LEFT_FRAMES, UnifiedStreamer

        left = int(opt) // 10 if opt else LEFT_FRAMES
        return UnifiedStreamer(UNIFIED_MODEL.path, leveler=not args.no_leveler, left_frames=left)
    raise SystemExit(f"unknown engine {spec}")


def make_punctuator(args):
    if args.no_punct:
        return None
    from live_subtitles import models
    from live_subtitles.punctuate import Punctuator

    if not models.PUNCT_MODEL.is_installed():
        models.download(models.PUNCT_MODEL)
    return Punctuator().punctuate


def peak_memory_mb() -> float:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2**20


class Recorder:
    """Watches the caption builder and records what the viewer would have seen, and when."""

    def __init__(self, builder: CaptionBuilder):
        self.b = builder
        self.first_seen: list[float] = []      # per committed-word index
        self.committed_at: list[float] = []
        self.prev_shown: list[str] = []
        self.prev_pending: list[str] = []
        self.prev_lines: dict[int, int] = {}
        self.rewrites_cosmetic = self.rewrites_word = self.partial_rewrites = self.reflows = 0
        self.screen_changes = 0
        self.prev_display: list[str] = []
        self.archive: list[str] = []           # words from before each clear (quiet spell)
        original_clear = builder.clear

        def clear() -> None:
            self.archive += self.b.shown
            del self.first_seen[len(self.archive):]     # pending words that never committed
            del self.committed_at[len(self.archive):]
            self.prev_shown, self.prev_pending, self.prev_lines = [], [], {}
            original_clear()

        builder.clear = clear

    def all_words(self) -> list[str]:
        return self.archive + self.b.shown

    def word_lines(self) -> dict[int, int]:
        if hasattr(self.b, "line_of"):
            return dict(enumerate(self.b.line_of))
        out, k = {}, 0
        for li, line in enumerate(self.b._layout(self.b.shown)):
            for _ in line:
                out[k] = li
                k += 1
        return out

    def snapshot(self, t: float) -> None:
        shown = list(self.b.shown)
        pending = self.b.pending.split() if self.b.pending else []
        for k in range(len(self.prev_shown), len(shown)):
            self.committed_at.append(t)
        everything = shown + pending
        while len(self.first_seen) < len(self.archive) + len(everything):
            self.first_seen.append(t)
        # Words that were already final on screen and changed afterwards.
        for k in range(min(len(self.prev_shown), len(shown))):
            if shown[k] != self.prev_shown[k]:
                if M.norm_word(shown[k]) == M.norm_word(self.prev_shown[k]):
                    self.rewrites_cosmetic += 1
                else:
                    self.rewrites_word += 1
        # In-progress words replaced by something other than their own continuation.
        prev_all = self.prev_shown + self.prev_pending
        for k in range(len(self.prev_shown), min(len(prev_all), len(everything))):
            if not everything[k].startswith(prev_all[k].rstrip(".,!?…-")):
                self.partial_rewrites += 1
        lines = self.word_lines()
        # Only words that were on screen count: the last max_lines lines of the previous layout.
        top = max(self.prev_lines.values(), default=0) - self.b.max_lines + 1
        for k, li in lines.items():
            if k in self.prev_lines and self.prev_lines[k] >= top and self.prev_lines[k] != li:
                self.reflows += 1
        display = self.b.display_lines()
        if display != self.prev_display:
            self.screen_changes += 1
        self.prev_display = display
        self.prev_lines = lines
        self.prev_shown, self.prev_pending = shown, pending


def run_scene(engine, punctuate, scene: str, args) -> dict:
    meta = json.loads((BENCH_DIR / f"{scene}.json").read_text(encoding="utf-8"))
    audio = read_wav(BENCH_DIR / f"{scene}.wav")
    if args.limit_s:
        audio = audio[: int(args.limit_s * SR)]
    dur = audio.size / SR

    captions.KEEP_WORDS = 10**9           # keep everything so the whole run can be scored
    builder = CaptionBuilder(max_lines=2, max_chars=42, punctuate=punctuate)
    rec = Recorder(builder)
    engine.load()

    packet = int(PACKET_S * SR)
    n_packets = int(np.ceil(audio.size / packet))
    clock = 0.0
    next_tick = TICK_S
    i = 0
    proc_s = ui_s = 0.0
    lag_max = 0.0
    cpu0, wall0 = time.process_time(), time.perf_counter()

    def ui(fn, *a) -> None:
        nonlocal clock, ui_s
        t = time.perf_counter()
        changed = fn(*a)
        dt = time.perf_counter() - t
        ui_s += dt
        clock += dt
        if changed is not False:
            rec.snapshot(clock)

    def tick_until(t_end: float) -> None:
        nonlocal next_tick
        while next_tick <= t_end:
            if builder.tick(next_tick, args.clear_after):
                rec.snapshot(next_tick)
            next_tick += TICK_S

    end_t = dur + 3.0
    while clock < end_t:
        arrived = min(n_packets, int(clock / PACKET_S + 1e-9))
        if arrived > i:
            chunk = audio[i * packet: arrived * packet]
            i = arrived
            lag_max = max(lag_max, clock - (i * PACKET_S))
        elif i >= n_packets:
            chunk = None            # playback ended: the recogniser's queue wait times out
            tick_until(clock + 0.2)
            clock += 0.2
        else:
            tick_until((i + 1) * PACKET_S)
            clock = (i + 1) * PACKET_S
            continue
        t = time.perf_counter()
        updates = engine.process(chunk)
        dt = time.perf_counter() - t
        proc_s += dt
        clock += dt
        for committed, pending, _phrase in updates:
            ui(lambda c, p: builder.add(c, p, clock), committed, pending)
        tick_until(clock)
    builder.tick(clock + 1.0, 999)       # the final settle
    rec.snapshot(clock + 1.0)
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    close = getattr(engine, "close", None)
    if close:
        close()
    return score(meta, builder, rec, dur, extra={
        "rtf": round(proc_s / dur, 3), "ui_ms_per_min": round(ui_s * 1000 / dur * 60),
        "cpu_cores": round(cpu / wall, 2), "cpu_s_per_audio_s": round(cpu / dur, 3),
        "max_backlog_s": round(lag_max, 2), "peak_mem_mb": round(peak_memory_mb()),
    }, limit_s=args.limit_s)


def score(meta: dict, builder: CaptionBuilder, rec: Recorder, dur: float, extra: dict, limit_s: float) -> dict:
    segs = [s for s in meta["segments"] if not limit_s or s["end"] <= limit_s]
    ref_words = [w for w in meta["ref_words"] if not limit_s or w["end"] <= limit_s]
    hyp = rec.all_words()
    hyp_tok = M.tokens(hyp)
    # map normalised hyp tokens back to committed-word indices for timing
    owners = []
    for k, w in enumerate(hyp):
        owners += [k] * len(M.tokens([w]))

    # WER against the human reference (subtitles / LibriSpeech transcripts).
    ordered = sorted(segs, key=lambda s: s["start"])
    ref_tok, ref_ends = [], set()
    for s in ordered:
        toks = M.tokens(s["text"])
        ref_tok += toks
        if meta["kind"] == "synthetic" and toks:
            ref_ends.add(len(ref_tok) - 1)
    ref_n, hyp_n = [a for a, _ in ref_tok], [a for a, _ in hyp_tok]
    pairs = M.align(ref_n, hyp_n)
    out = {"scene": meta["name"], "duration_s": round(dur, 1), **M.wer(ref_n, hyp_n, pairs)}
    out["sentence_f1"] = M.sentence_f1([b for _, b in ref_tok], [b for _, b in hyp_tok], pairs,
                                       ref_ends if meta["kind"] == "synthetic" else None)

    # Delay: align to Parakeet's timed words, use exact matches only.
    timed = [(M.norm_word(w["word"]), w) for w in ref_words if M.norm_word(w["word"])]
    tp = M.align([a for a, _ in timed], hyp_n)
    seen, final = [], []
    for a, b in tp:
        if a is not None and b is not None and timed[a][0] == hyp_n[b]:
            k = owners[b]
            end = timed[a][1]["end"]
            seen.append(rec.first_seen[k] - end)
            final.append(rec.committed_at[k] - end)
    out.update({
        "delay_seen_p50": M.pct(seen, 50), "delay_seen_p90": M.pct(seen, 90),
        "delay_final_p50": M.pct(final, 50), "delay_final_p90": M.pct(final, 90), "timed_words": len(seen),
    })

    # Words put on screen where nobody is talking (1.5 s grace either side of speech).
    speech = [(s["start"] - 1.5, s["end"] + 1.5 + (out["delay_seen_p90"] or 1.0)) for s in segs]
    phantom = sum(1 for k in range(len(hyp)) if not any(a <= rec.first_seen[k] <= b for a, b in speech))
    minutes = dur / 60
    out.update({
        "phantom_words": phantom,
        "rewrites_word_per_min": round(rec.rewrites_word / minutes, 1),
        "rewrites_cosmetic_per_min": round(rec.rewrites_cosmetic / minutes, 1),
        "partial_rewrites_per_min": round(rec.partial_rewrites / minutes, 1),
        "line_jumps_per_min": round(rec.reflows / minutes, 1),
        "screen_changes_per_min": round(rec.screen_changes / minutes, 1),
        **extra,
    })
    out["text"] = " ".join(hyp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("--label")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-leveler", action="store_true")
    ap.add_argument("--no-punct", action="store_true")
    ap.add_argument("--clear-after", type=float, default=4.0)
    ap.add_argument("--limit-s", type=float, default=0.0)
    args = ap.parse_args()
    label = args.label or args.engine
    out_dir = RESULTS / label
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parakeet-unified punctuates itself (and sherpa-onnx can't share its process).
    punctuate = None if args.engine.startswith("unified") else make_punctuator(args)
    for scene in args.scenes.split(","):
        engine = make_engine(args.engine, args)
        res = run_scene(engine, punctuate, scene, args)
        res["label"] = label
        (out_dir / f"{scene}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "text"}), flush=True)


if __name__ == "__main__":
    main()
