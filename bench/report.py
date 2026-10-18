"""Compare benchmark runs: a markdown table of `bench.run` results, one column per label.

    .\\.venv\\Scripts\\python.exe -m bench.report live-unified live-nemotron-560 live-nemotron-160
    .\\.venv\\Scripts\\python.exe -m bench.report live-unified live-nemotron-560 --metric delay_seen_p50

Rows are scenes, then the mean over the synthetic and the real-world scenes. Results
come from %LOCALAPPDATA%\\Live-Subtitles\\bench\\results\\<label>\\<scene>.json.
"""

from __future__ import annotations

import argparse
import json

from bench.run import REAL, RESULTS, SYNTHETIC

SUMMARY = ("wer", "sentence_f1", "delay_seen_p50", "delay_seen_p90", "delay_final_p50", "rewrites_word_per_min",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--metric", default="wer")
    args = ap.parse_args()
    runs = {lab: load(lab) for lab in args.labels}
    print(f"{args.metric} by scene\n")
    print("| scene | " + " | ".join(args.labels) + " |")
    print("|---|" + "---|" * len(args.labels))
    for scene in (*SYNTHETIC, *REAL):
        if any(scene in r for r in runs.values()):
            print(f"| {scene} | " + " | ".join(str(r.get(scene, {}).get(args.metric, "")) for r in runs.values()) + " |")
    for name, group in (("mean, synthetic", SYNTHETIC), ("mean, real-world", REAL)):
        print(f"| **{name}** | " + " | ".join(
            str(mean([r[s].get(args.metric) for s in group if s in r])) for r in runs.values()) + " |")
    print("\nmeans over the real-world scenes\n")
    print("| metric | " + " | ".join(args.labels) + " |")
    print("|---|" + "---|" * len(args.labels))
    for m in SUMMARY:
        print(f"| {m} | " + " | ".join(str(mean([r[s].get(m) for s in REAL if s in r])) for r in runs.values()) + " |")


if __name__ == "__main__":
    main()
