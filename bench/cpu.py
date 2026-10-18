"""Steady-state machine cost of live captions, in real time (as it runs during a film).

    .\\.venv\\Scripts\\python.exe -m bench.cpu gpu                      # the app's GPU engine (worker)
    .\\.venv\\Scripts\\python.exe -m bench.cpu nemotron-560 --threads 4  # the CPU engine

Feeds a scene to the app's own LiveRecognizer thread in 50 ms packets at playback speed,
then reports CPU (in cores, where 1.0 is one logical CPU fully busy; for `gpu` this
counts this process and the worker process), GPU utilisation (nvidia-smi, sampled once a
second), and how far behind the captions ran. `bench.run` measures accuracy and delay on
a simulated clock; this measures what the app costs the machine while a film plays.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
import subprocess
import threading
import time

from bench.run import make_engine, make_punctuator
from bench.testset import BENCH_DIR, SR, read_wav
from live_subtitles.live import LiveRecognizer


def process_tree(pid: int) -> list[int]:
    """`pid` and its children: a venv's python.exe is a launcher that runs the real
    interpreter as a child process, which is where the CPU time goes."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}').ProcessId"],
                         capture_output=True, text=True, timeout=30).stdout
    return [pid, *(int(x) for x in out.split() if x.isdigit())]


def process_cpu_s(pids: list[int]) -> float:
    """User + kernel CPU seconds of other processes (Windows)."""
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    total = 0.0
    for pid in pids:
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            continue
        try:
            t = [ctypes.c_ulonglong() for _ in range(4)]
            k32.GetProcessTimes(ctypes.c_void_p(h), *[ctypes.byref(x) for x in t])
            total += (t[2].value + t[3].value) / 1e7
        finally:
            k32.CloseHandle(ctypes.c_void_p(h))
    return total


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[int] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(1.0):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
                self.samples.append(int(out.split()[0]))
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", help="gpu (the app's worker process) or a bench.run engine spec")
    ap.add_argument("--scene", default="tos")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--no-leveler", action="store_true")
    ap.add_argument("--no-punct", action="store_true")
    args = ap.parse_args()
    audio = read_wav(BENCH_DIR / f"{args.scene}.wav")[int(args.start * SR): int((args.start + args.seconds) * SR)]
    if args.engine == "gpu":
        from live_subtitles.unified import UnifiedEngine

        punctuate = None
        engine = UnifiedEngine(leveler=not args.no_leveler)
    else:
        punctuate = None if args.engine.startswith("unified") else make_punctuator(args)
        engine = make_engine(args.engine, args)
    chunks: queue.Queue = queue.Queue(maxsize=400)
    words: list[str] = []

    def on_words(c, p, _phrase):
        words.extend(c)
        if punctuate and c:
            punctuate(words[-40:])          # the UI thread's per-update work

    ready = []
    rec = LiveRecognizer(engine, chunks, on_words, on_ready=lambda: ready.append(1))
    rec.start()
    while not ready:
        time.sleep(0.05)
    worker = getattr(getattr(engine, "_proc", None), "pid", None)
    worker = process_tree(worker) if worker else None
    gpu = GpuSampler()
    gpu.start()
    packet = int(0.05 * SR)
    cpu0, t0 = time.process_time(), time.perf_counter()
    wcpu0 = process_cpu_s(worker) if worker else 0.0
    for i in range(0, audio.size, packet):
        due = t0 + i / SR
        time.sleep(max(0.0, due - time.perf_counter()))
        chunks.put(audio[i:i + packet])
    while not chunks.empty():
        time.sleep(0.01)
    backlog = time.perf_counter() - t0 - audio.size / SR
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0 + ((process_cpu_s(worker) - wcpu0) if worker else 0.0)
    gpu.stop.set()
    rec.stop()
    rec.join(5)
    s = sorted(gpu.samples) or [0]
    print(json.dumps({"engine": args.engine, "scene": args.scene, "seconds": args.seconds,
                      "cpu_cores": round(cpu / wall, 2), "gpu_util_p50": s[len(s) // 2],
                      "gpu_util_max": s[-1], "final_backlog_s": round(backlog, 2), "words": len(words)}))


if __name__ == "__main__":
    main()
