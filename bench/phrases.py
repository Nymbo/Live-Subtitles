"""Second-pass candidates head to head: each model transcribes the same phrase clips.

    .\\.venv\\Scripts\\python.exe -m bench.phrases parakeet-v3-gpu
    .\\.venv\\Scripts\\python.exe -m bench.phrases unified-cpu --threads 8

A phrase is a run of reference speech with gaps under 0.5 s (at most 12 s), padded by 0.6 s: about what
the live engine hands the accuracy pass. Reports WER, sentence-end F1 and decode time per
phrase. Each backend runs in its own process (sherpa-onnx and onnxruntime-gpu both ship
an onnxruntime.dll and cannot share one).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bench import metrics as M
from bench.run import RESULTS, SCENES
from bench.testset import BENCH_DIR, SR, read_wav
from live_subtitles.paths import MODELS_DIR

MAX_PHRASE_S = 12.0     # the live engine cuts long phrases at shorter gaps, about this long


def onnx_asr_backend(name: str, path: str | None = None):
    import onnx_asr
    import onnxruntime as ort

    ort.preload_dlls()
    model = onnx_asr.load_model(name, path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return lambda audio: model.recognize(audio, sample_rate=SR)


def sherpa_backend(kind: str, threads: int):
    import sherpa_onnx

    d = MODELS_DIR
    if kind in ("unified", "ultra-int8"):
        m = d / ("sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming" if kind == "unified"
                 else "parakeet-ultra-int8")
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(m / "encoder.int8.onnx"), decoder=str(m / "decoder.int8.onnx"),
            joiner=str(m / "joiner.int8.onnx"), tokens=str(m / "tokens.txt"), num_threads=threads,
            model_type="nemo_transducer", feature_dim=128 if kind == "ultra-int8" else 80)
    elif kind == "cohere":
        m = d / "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
        rec = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=str(m / "encoder.int8.onnx"), decoder=str(m / "decoder.int8.onnx"),
            tokens=str(m / "tokens.txt"), num_threads=threads, language="en")
    elif kind == "qwen3":
        m = d / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
        rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(m / "conv_frontend.onnx"), encoder=str(m / "encoder.int8.onnx"),
            decoder=str(m / "decoder.int8.onnx"), tokenizer=str(m / "tokenizer"), num_threads=threads)
    else:
        raise SystemExit(kind)

    def run(audio):
        s = rec.create_stream()
        s.accept_waveform(SR, audio)
        rec.decode_stream(s)
        return s.result.text

    return run


def make(backend: str, threads: int):
    if backend == "parakeet-v3-gpu":
        return onnx_asr_backend("nemo-parakeet-tdt-0.6b-v3")
    if backend == "parakeet-v2-gpu":
        return onnx_asr_backend("nemo-parakeet-tdt-0.6b-v2")
    if backend == "ultra-gpu":
        return onnx_asr_backend("nemo-parakeet-tdt-0.6b-v3", str(MODELS_DIR / "parakeet-ultra-fp32-onnx"))
    return sherpa_backend(backend.removesuffix("-cpu"), threads)


def phrases(meta: dict) -> list[tuple[float, float]]:
    groups: list[list[float]] = []
    for s in sorted(meta["segments"], key=lambda s: s["start"]):
        if groups and s["start"] - groups[-1][1] < 0.5 and s["end"] - groups[-1][0] <= MAX_PHRASE_S:
            groups[-1][1] = max(groups[-1][1], s["end"])
        else:
            groups.append([s["start"], s["end"]])
    out = []
    for i, (a, b) in enumerate(groups):           # pad by up to 0.6 s, never into a neighbour
        before = a - groups[i - 1][1] if i else 1.2
        after = groups[i + 1][0] - b if i + 1 < len(groups) else 1.2
        out.append((max(0.0, a - min(0.6, before / 2)), b + min(0.6, after / 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scenes", default=",".join(SCENES))
    args = ap.parse_args()
    t = time.perf_counter()
    recognize = make(args.backend, args.threads)
    load_s = time.perf_counter() - t
    recognize(np.zeros(SR, np.float32))
    out_dir = RESULTS / "phrases"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in args.scenes.split(","):
        meta = json.loads((BENCH_DIR / f"{scene}.json").read_text(encoding="utf-8"))
        audio = read_wav(BENCH_DIR / f"{scene}.wav")
        hyp, times = [], []
        for a, b in phrases(meta):
            t = time.perf_counter()
            hyp.append(recognize(audio[int(a * SR):int(b * SR)]).strip())
            times.append(time.perf_counter() - t)
        segs = sorted(meta["segments"], key=lambda s: s["start"])
        ref_tok, ends = [], set()
        for s in segs:
            ref_tok += M.tokens(s["text"])
            ends.add(len(ref_tok) - 1)
        hyp_tok = M.tokens(" ".join(hyp))
        ref_n, hyp_n = [x for x, _ in ref_tok], [x for x, _ in hyp_tok]
        pairs = M.align(ref_n, hyp_n)
        row = {"backend": args.backend, "scene": scene, **M.wer(ref_n, hyp_n, pairs),
               "sentence_f1": M.sentence_f1([r for _, r in ref_tok], [h for _, h in hyp_tok], pairs,
                                            ends if meta["kind"] == "synthetic" else None),
               "phrases": len(times), "ms_p50": round(float(np.median(times)) * 1000),
               "ms_p95": round(float(np.percentile(times, 95)) * 1000), "load_s": round(load_s, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out_dir / f"{args.backend}-{scene}.txt").write_text("\n".join(hyp), encoding="utf-8")
    (out_dir / f"{args.backend}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
