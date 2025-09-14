"""
Benchmark the speed of LIEF reading a binary file.

Benchmarking LIEF reading on 1024 files in bytes mode
First pass: 0.009004 seconds
Second pass: 0.008946 seconds

Benchmarking LIEF reading on 1024 files in memoryview mode
First pass: 0.037796 seconds
Second pass: 0.037811 seconds

Benchmarking LIEF reading on 1024 files in file mode
First pass: 0.027196 seconds
Second pass: 0.008756 seconds

Benchmarking LIEF reading on 1024 files in mmap mode
First pass: 0.051612 seconds
Second pass: 0.037748 seconds

Benchmarking LIEF reading on 4096 files in bytes mode
First pass: 0.003170 seconds
Second pass: 0.003066 seconds

Benchmarking LIEF reading on 4096 files in memoryview mode
First pass: 0.026238 seconds
Second pass: 0.026222 seconds

Benchmarking LIEF reading on 4096 files in file mode
First pass: 0.020081 seconds
Second pass: 0.003069 seconds
"""

from argparse import ArgumentParser
import mmap
from pathlib import Path
import sys
import time

import lief
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.binanal import _parse_pe_and_get_size


lief.logging.disable()


parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["bytes", "memoryview", "file", "mmap"])
args = parser.parse_args()


files = sorted(filter(lambda p: p.is_file(), Path("./data/sor").rglob("*")))[0:4096]
print(f"Benchmarking LIEF reading on {len(files)} files in {args.mode} mode")


def get_lief_input(file: Path, mode: str):
    if mode == "bytes":
        with open(file, "rb") as f:
            return f.read()
    if mode == "memoryview":
        with open(file, "rb") as f:
            return memoryview(f.read())
    if mode == "file":
        return str(file)
    if mode == "mmap":
        with open(file, "r+b") as f:
            return memoryview(mmap.mmap(f.fileno(), 0))


times = []
for file in tqdm(files, disable=True):
    data = get_lief_input(file, mode=args.mode)
    size = file.stat().st_size
    t_i = time.time()
    _parse_pe_and_get_size(data, size=size)
    t_f = time.time()
    times.append(t_f - t_i)

print(f"First pass: {sum(times) / len(times):.6f} seconds")


times = []
for file in tqdm(files, disable=True):
    data = get_lief_input(file, mode=args.mode)
    size = file.stat().st_size
    t_i = time.time()
    _parse_pe_and_get_size(data, size=size)
    t_f = time.time()
    times.append(t_f - t_i)

print(f"Second pass: {sum(times) / len(times):.6f} seconds")