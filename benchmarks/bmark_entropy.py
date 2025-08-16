"""
Benchmark the entropy computations.

func compute_entropy_rolling           size 262144 radius 64     ... time 0.012s memory 196.473MB
func compute_histogram_entropy_rolling size 262144 radius 64     ... time 0.002s memory 217.410MB
func compute_entropy_rolling           size 262144 radius 128    ... time 0.012s memory 231.473MB
func compute_histogram_entropy_rolling size 262144 radius 128    ... time 0.002s memory 217.984MB
func compute_entropy_rolling           size 262144 radius 256    ... time 0.012s memory 232.359MB
func compute_histogram_entropy_rolling size 262144 radius 256    ... time 0.002s memory 217.734MB
func compute_entropy_rolling           size 262144 radius 512    ... time 0.012s memory 232.734MB
func compute_histogram_entropy_rolling size 262144 radius 512    ... time 0.002s memory 217.984MB
func compute_entropy_rolling           size 262144 radius 1024   ... time 0.012s memory 232.672MB
func compute_histogram_entropy_rolling size 262144 radius 1024   ... time 0.002s memory 217.738MB
func compute_entropy_rolling           size 1048576 radius 64    ... time 0.036s memory 286.398MB
func compute_histogram_entropy_rolling size 1048576 radius 64    ... time 0.007s memory 224.953MB
func compute_entropy_rolling           size 1048576 radius 128   ... time 0.036s memory 289.336MB
func compute_histogram_entropy_rolling size 1048576 radius 128   ... time 0.007s memory 225.738MB
func compute_entropy_rolling           size 1048576 radius 256   ... time 0.036s memory 288.398MB
func compute_histogram_entropy_rolling size 1048576 radius 256   ... time 0.007s memory 224.941MB
func compute_entropy_rolling           size 1048576 radius 512   ... time 0.036s memory 289.336MB
func compute_histogram_entropy_rolling size 1048576 radius 512   ... time 0.007s memory 225.738MB
func compute_entropy_rolling           size 1048576 radius 1024  ... time 0.036s memory 288.398MB
func compute_histogram_entropy_rolling size 1048576 radius 1024  ... time 0.007s memory 224.895MB
func compute_entropy_rolling           size 4194304 radius 64    ... time 0.141s memory 512.652MB
func compute_histogram_entropy_rolling size 4194304 radius 64    ... time 0.033s memory 256.742MB
func compute_entropy_rolling           size 4194304 radius 128   ... time 0.139s memory 512.652MB
func compute_histogram_entropy_rolling size 4194304 radius 128   ... time 0.033s memory 256.742MB
func compute_entropy_rolling           size 4194304 radius 256   ... time 0.140s memory 512.652MB
func compute_histogram_entropy_rolling size 4194304 radius 256   ... time 0.033s memory 256.742MB
func compute_entropy_rolling           size 4194304 radius 512   ... time 0.139s memory 511.355MB
func compute_histogram_entropy_rolling size 4194304 radius 512   ... time 0.033s memory 255.742MB
func compute_entropy_rolling           size 4194304 radius 1024  ... time 0.140s memory 507.355MB
func compute_histogram_entropy_rolling size 4194304 radius 1024  ... time 0.033s memory 251.742MB
func compute_entropy_rolling           size 16777216 radius 64   ... time 0.550s memory 1387.527MB
func compute_histogram_entropy_rolling size 16777216 radius 64   ... time 0.135s memory 363.617MB
func compute_entropy_rolling           size 16777216 radius 128  ... time 0.541s memory 1403.527MB
func compute_histogram_entropy_rolling size 16777216 radius 128  ... time 0.135s memory 379.617MB
func compute_entropy_rolling           size 16777216 radius 256  ... time 0.546s memory 1403.527MB
func compute_histogram_entropy_rolling size 16777216 radius 256  ... time 0.135s memory 379.617MB
func compute_entropy_rolling           size 16777216 radius 512  ... time 0.548s memory 1403.500MB
func compute_histogram_entropy_rolling size 16777216 radius 512  ... time 0.138s memory 379.617MB
func compute_entropy_rolling           size 16777216 radius 1024 ... time 0.549s memory 1403.570MB
func compute_histogram_entropy_rolling size 16777216 radius 1024 ... time 0.135s memory 379.617MB
"""

import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable

import numpy as np
from numpy import typing as npt
import psutil

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import EntropyGuider


def run_benchmark(func: Callable, b: npt.NDArray[np.uint8], radius: int, dtype: npt.DTypeLike, runs: int, interval: float = 0.01) -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    stop = threading.Event()

    ts = []
    rss = []

    def poll() -> None:
        t0 = time.perf_counter()
        while not stop.is_set():
            ts.append(time.perf_counter() - t0)
            rss.append(proc.memory_info().rss)
            time.sleep(interval)

    threading.Thread(target=poll, daemon=True).start()
    func(b, radius, dtype=dtype)
    t0 = time.perf_counter()
    for _ in range(runs):
        func(b, radius)
    elapsed = time.perf_counter() - t0
    stop.set()
    time.sleep(interval)

    return  elapsed / runs, max(rss)


def main() -> None:
    sizes = [2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24]
    radii = [256]
    RUNS = 5

    for size in sizes:
        for radius in radii:
            b = np.random.randint(0, 256, size=size, dtype=np.uint8)
            for func in [
                # EntropyGuider.compute_entropy_scipy,
                # EntropyGuider.compute_entropy,
                # EntropyGuider.compute_entropy_rolling,
                # EntropyGuider.compute_entropy_rolling,
                # EntropyGuider.compute_histogram_entropy,
                EntropyGuider.compute_histogram_entropy_rolling,
            ]:
                for dtype in [np.float32, np.float64]:
                    print(f"func {func.__name__}{' ' * (len('compute_histogram_entropy_rolling') - len(func.__name__))} size {size} radius {radius} dtype {dtype} ...", end=" ", flush=True)
                    t, m = run_benchmark(func, b, radius, dtype, RUNS)
                    print(f"time {t:.4f}s memory {m / (1024 ** 2):.3f}MB", flush=True)


if __name__ == "__main__":
    main()
