"""
Benchmark bit packing functions.
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import packbits
from src.utils import unpackbits
from src.utils import pack_bool_tensor
from src.utils import unpack_bit_tensor


def time_numpy(axis: int) -> None:
    b = np.random.randint(0, 2, size=size, dtype=bool)

    u = np.packbits(b, axis=axis)
    t_0 = time.time()
    for _ in range(10):
        u = np.packbits(b, axis=axis)
    t_1 = time.time()
    print(f"  np.packbits: {round(t_1 - t_0, 2)}s")

    p = np.unpackbits(u, axis=axis)
    t_0 = time.time()
    for _ in range(10):
        p = np.unpackbits(u, axis=axis)
    t_1 = time.time()
    print(f"  np.unpackbits: {round(t_1 - t_0, 2)}s")

    if not np.equal(b, p).all():
        raise RuntimeError()


def time_torch(axis: int) -> None:
    b = torch.randint(0, 2, size=size, dtype=bool)

    u = packbits(b, axis=axis)
    t_0 = time.time()
    for _ in range(10):
        u = packbits(b, axis=axis)
    t_1 = time.time()
    print(f"  packbits: {round(t_1 - t_0, 2)}s")

    p = unpackbits(u, axis=axis)
    t_0 = time.time()
    for _ in range(10):
        p = unpackbits(u, axis=axis)
    t_1 = time.time()
    print(f"  unpackbits: {round(t_1 - t_0, 2)}s")

    if not torch.equal(b, p):
        raise RuntimeError()

    u = pack_bool_tensor(b)
    t_0 = time.time()
    for _ in range(10):
        u = pack_bool_tensor(b)
    t_1 = time.time()
    print(f"  pack_bool_tensor: {round(t_1 - t_0, 2)}s")

    p = unpack_bit_tensor(u, size[-1])
    t_0 = time.time()
    for _ in range(10):
        p = unpack_bit_tensor(u, size[-1])
    t_1 = time.time()
    print(f"  unpack_bit_tensor: {round(t_1 - t_0, 2)}s")

    if not torch.equal(b, p):
        raise RuntimeError()


sizes = [
    (2 ** 20, 8),
    (2 ** 20, 12),
    (2 ** 20, 16),
    (2 ** 24, 8),
    (2 ** 24, 12),
    (2 ** 24, 16),
    (16, 2 ** 20, 8),
    (16, 2 ** 20, 12),
    (16, 2 ** 20, 16),
    (16, 2 ** 24, 8),
    (16, 2 ** 24, 12),
    (16, 2 ** 24, 16),
]


for size in sizes:
    print(f"size: {size}")
    time_numpy(axis=len(size) - 2)
    time_torch(axis=len(size) - 2)

