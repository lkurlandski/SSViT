"""
Benchmark the entropy computations.
"""

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from typing import Callable

import numpy as np
from numpy import typing as npt
import psutil
import torch
from torch import ByteTensor

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binentropy import compute_entropy_naive_numpy
from src.binentropy import compute_entropy_rolling_numpy
from src.binentropy import compute_entropy_naive_torch
from src.binentropy import compute_entropy_rolling_torch


def run_benchmark(func: Callable[[ByteTensor | npt.NDArray[np.uint8], int], Any], b: npt.NDArray[np.uint8] | ByteTensor, radius: int, runs: int, interval: float = 0.01) -> tuple[float, float]:
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
    func(b, radius)
    t0 = time.perf_counter()
    for _ in range(runs):
        func(b, radius)
    elapsed = time.perf_counter() - t0
    stop.set()
    time.sleep(interval)

    return  elapsed / runs, max(rss)


def main() -> None:
    sizes = [2 ** 10]
    radii = [256]
    RUNS = 5

    for size in sizes:
        for radius in radii:
            bn = np.random.randint(0, 256, size=size, dtype=np.uint8)
            bt = torch.from_numpy(bn).to(torch.uint8)
            for func, b in [
                (compute_entropy_naive_numpy, bn),
                (compute_entropy_rolling_numpy, bn),
                (compute_entropy_naive_torch, bt),
                (compute_entropy_rolling_torch, bt),
            ]:
                print(f"func {func.__name__} size {size} radius {radius} ...", end=" ", flush=True)
                t, m = run_benchmark(func, b, radius, RUNS)
                print(f"time {t:.4f}s memory {m / (1024 ** 2):.3f}MB", flush=True)


if __name__ == "__main__":
    main()
