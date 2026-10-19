"""Compare benchmark runs: a markdown table of `bench.run` results, one column per label.

    .\\.venv\\Scripts\\python.exe -m bench.report live-unified live-nemotron-560 live-nemotron-160
    .\\.venv\\Scripts\\python.exe -m bench.report live-unified live-nemotron-560 --metric delay_seen_p50

Rows are scenes, then the mean over the synthetic and the real-world scenes. Results
come from %LOCALAPPDATA%\\Live-Subtitles\\bench\\results\\<label>\\<scene>.json.

    .\\.venv\\Scripts\\python.exe -m bench.report --rescore

re-scores the WER of every saved result (each run's caption text is stored with it)
after a change to the scorer, without running any engine again.
"""

from __future__ import annotations

import argparse
import json

from bench import metrics as M
from bench.run import BENCH_DIR, FULL, REAL, RESULTS, SYNTHETIC

SUMMARY = ("wer", "wer_clean", "wer_plain", "sentence_f1", "delay_seen_p50", "delay_seen_p90", "delay_final_p50", "rewrites_word_per_min",
           "line_jumps_per_min", "rtf", "cpu_cores")


def load(label: str) -> dict[str, dict]:
    out = {}
    for f in (RESULTS / label).glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["scene"]] = d
    return out


def mean(rows: list[float]) -> float | None:
    rows = [r for r in rows if r is not None]
    return round(sum(rows) / len(rows), 2) if rows else None


def reference(scene: str, until_s: float | None = None) -> str:
    meta = json.loads((BENCH_DIR / f"{scene}.json").read_text(encoding="utf-8"))
    segs = sorted(meta["segments"], key=lambda s: s["start"])
    # a run with --limit-s scored only the segments that ended inside its audio
    if until_s is not None and until_s < meta["duration"] - 0.5:
        segs = [s for s in segs if s["end"] <= until_s]
    return " ".join(s["text"] for s in segs)


def rescore() -> None:
    n = 0
    for f in RESULTS.glob("*/*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "text" in d and (BENCH_DIR / f"{d['scene']}.json").exists():
            d.update(M.text_wer(reference(d["scene"], d.get("duration_s")), d["text"]))
            f.write_text(json.dumps(d, indent=1), encoding="utf-8")
            n += 1
    for f in (RESULTS / "phrases").glob("*.json"):       # bench.phrases: rows per backend
        rows = json.loads(f.read_text(encoding="utf-8"))
        for row in rows:
            hyp = RESULTS / "phrases" / f"{row['backend']}-{row['scene']}.txt"
            if hyp.exists():
                row.update(M.text_wer(reference(row["scene"]), " ".join(hyp.read_text(encoding="utf-8").splitlines())))
                n += 1
        f.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"re-scored {n} results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--metric", default="wer")
    ap.add_argument("--rescore", action="store_true", help="re-score every saved result's WER first")
    args = ap.parse_args()
    if args.rescore:
        rescore()
    if not args.labels:
        return
    runs = {lab: load(lab) for lab in args.labels}
    print(f"{args.metric} by scene\n")
    print("| scene | " + " | ".join(args.labels) + " |")
    print("|---|" + "---|" * len(args.labels))
    for scene in (*SYNTHETIC, *REAL, *FULL):
        if any(scene in r for r in runs.values()):
            print(f"| {scene} | " + " | ".join(str(r.get(scene, {}).get(args.metric, "")) for r in runs.values()) + " |")
    for name, group in (("mean, synthetic", SYNTHETIC), ("mean, real-world", REAL), ("mean, uncut", FULL)):
        print(f"| **{name}** | " + " | ".join(
            str(mean([r[s].get(args.metric) for s in group if s in r])) for r in runs.values()) + " |")
    print("\nmeans over the real-world scenes\n")
    print("| metric | " + " | ".join(args.labels) + " |")
    print("|---|" + "---|" * len(args.labels))
    for m in SUMMARY:
        print(f"| {m} | " + " | ".join(str(mean([r[s].get(m) for s in REAL if s in r])) for r in runs.values()) + " |")


if __name__ == "__main__":
    main()
