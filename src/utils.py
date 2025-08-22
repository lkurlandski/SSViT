"""
Utilities.
"""

from __future__ import annotations
import os
import random
from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import BoolTensor
from torch import CharTensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_optimal_num_workers(ncpu: int = len(os.sched_getaffinity(0)), ngpu: int = torch.cuda.device_count()) -> int:
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if ngpu > ncpu:
        raise RuntimeError(f"Number of GPUs ({ngpu}) exceeds number of CPU cores ({ncpu}).")
    return max(0, ncpu // max(1, ngpu) - 1)


def get_optimal_num_worker_threads(num_workers: int = 0, ncpu: int = len(os.sched_getaffinity(0))) -> int:
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if num_workers - 1 > ncpu:
        raise RuntimeError(f"Number of worker processes ({num_workers} + 1) exceeds number of CPU cores ({ncpu}).")
    if num_workers == 0:
        return ncpu
    return max(1, ncpu // num_workers - 1)


class TensorError(ValueError):
    """
    Exception raised when the shape of a tensor does not match the expected shape.
    """

    def __init__(self, x: Tensor, s: Optional[tuple[Optional[int], ...]], t: Optional[torch.dtype]) -> None:
        super().__init__(
            f"Expected tensor with dtype {t} and shape {s}. Got tensor with dtype {x.dtype} and shape {tuple(x.shape)}."
        )


def check_tensor(x: Tensor, s: Optional[tuple[Optional[int], ...]] = None, t: Optional[torch.dtype | tuple[torch.dtype]] = None) -> None:
    if t is not None and isinstance(t, torch.dtype):
        t = (t,)
    if t is not None and x.dtype not in t:
        raise TensorError(x, s, t)
    if s is not None and len(x.shape) != len(s):
        raise TensorError(x, s, t)
    if s is not None:
        for i, j in zip(x.shape, s, strict=True):
            if j is not None and i != j:
                raise TensorError(x, s, t)


def smallest_unsigned_integer_dtype(n: int) -> torch.dtype:
    """
    Returns the smallest unsigned integer dtype that can represent the given number of bits.
    """
    if n <= 8:
        return torch.uint8
    if n <= 16:
        return torch.uint16
    if n <= 32:
        return torch.uint32
    if n <= 64:
        return torch.uint64
    raise ValueError(f"Cannot represent {n} bits with an unsigned integer dtype.")


@torch.no_grad()  # type: ignore[misc]
def pack_bool_tensor(b: BoolTensor) -> Tensor:
    """
    Packs a boolean tensor along the last dimension into an integer tensor.

    NOTE: unpack(pack(b)) == b, but pack(unpack(x)) != x.
    """
    if b.dtype is not torch.bool:
        raise TypeError(f"b must be bool, got {b.dtype}")
    if b.ndim < 1:
        raise ValueError("b must have at least 1 dimension")

    d = b.shape[-1]
    if not (1 <= d <= 64):
        raise ValueError(f"Cannot pack boolean tensor with {d} channels.")

    # Use torch.int64 because shift ops aren’t implemented for smaller uint types on CPU
    weights = (1 << torch.arange(d, device=b.device, dtype=torch.int64))
    x = (b.to(torch.int64) * weights).sum(dim=-1)

    # Cast down to the minimal unsigned dtype (storage stays compact for transfer)
    out_dtype = smallest_unsigned_integer_dtype(d)
    return x.to(out_dtype)


@torch.no_grad()  # type: ignore[misc]
def unpack_bit_tensor(x: Tensor, d: int, dtype: torch.dtype = torch.bool) -> Tensor:
    """
    Unpacks an integer tensor into a boolean (or other) tensor along a new last dimension.

    NOTE: unpack(pack(b)) == b, but pack(unpack(x)) != x.
    """
    if not (1 <= d <= 64):
        raise ValueError(f"Cannot unpack bit tensor with {d} channels.")
    if x.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        raise TypeError(f"x must be an unsigned integer tensor, got {x.dtype}")

    # Use torch.int64 workspace for safe shifts
    x = x.to(torch.int64)
    masks = (1 << torch.arange(d, device=x.device, dtype=torch.int64))
    z = (x.unsqueeze(-1) & masks) != 0
    return z.to(dtype)


@torch.no_grad()  # type: ignore[misc]
def packbits(x: BoolTensor, axis: int = -1) -> CharTensor:
    """
    Torch variant of numpy.packbits(..., bitorder='little').

    NOTE: uses torch.unfold() to work on views without requiring torch.contiguous().
    """
    if x.dtype is not torch.bool:
        raise TypeError(f"packbits: expected bool tensor, got {x.dtype}")
    if x.ndim == 0:
        raise ValueError("packbits: input must have at least 1 dimension")

    axis = axis if axis >= 0 else x.ndim + axis
    d = x.shape[axis]
    if d == 0:
        raise ValueError("packbits: cannot pack an empty axis")

    # Move target axis to the end (view only; no copy)
    y = x.movedim(axis, -1)  # [..., d]

    # Pad to multiple of 8 along the last dim (only small tail gets padded)
    pad = (-d) % 8
    if pad:
        y = F.pad(y, (0, pad), value=False)  # [..., d+pad]

    # Group bits into bytes using unfold (view; no copy)
    # Result shape: [..., B, 8] where B = (d+pad)//8
    y8 = y.unfold(dimension=-1, size=8, step=8)  # bool [..., B, 8]

    # Pack each 8-bit block into a byte (little-endian: first element -> LSB)
    weights = (1 << torch.arange(8, device=y8.device, dtype=torch.uint8))  # [8]
    out = (y8.to(torch.uint8) * weights).sum(dim=-1, dtype=torch.uint8)    # [..., B]

    # Move bytes axis back to original position
    return out.movedim(-1, axis)


@torch.no_grad()  # type: ignore[misc]
def unpackbits(x: CharTensor, count: int = -1, axis: int = -1) -> BoolTensor:
    """
    Torch variant of numpy.unpackbits(..., bitorder='little').

    NOTE: uses torch.unfold() to work on views without requiring torch.contiguous().
    """
    count = x.shape[axis] * 8 if count == -1 else count
    if x.dtype is not torch.uint8:
        raise TypeError(f"unpackbits: expected torch.uint8, got {x.dtype}")
    if x.ndim == 0:
        raise ValueError("unpackbits: input must have at least 1 dimension")
    if count < 1:
        raise ValueError(f"unpackbits: count must be >= 1, got {count}")

    axis = axis if axis >= 0 else x.ndim + axis

    # Move byte-axis to the end (view only)
    y = x.movedim(axis, -1)  # [..., B]
    total_bits = int(y.shape[-1]) * 8
    if count > total_bits:
        raise ValueError(f"unpackbits: count={count} exceeds capacity={total_bits} along axis")

    # Expand each byte to 8 bits: [..., B, 8] (viewed/broadcast, no big copy)
    masks = (1 << torch.arange(8, device=y.device, dtype=torch.uint8))  # [8]
    bits = (y.unsqueeze(-1) & masks) != 0  # bool [..., B, 8]

    # Flatten the (B,8) pair to total_bits; use flatten to avoid unnecessary copies
    bits_flat = bits.flatten(-2)           # [..., total_bits]
    z = bits_flat[..., :count]             # [..., count]

    # Move the expanded bit-axis back
    return z.movedim(-1, axis)
