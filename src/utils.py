"""
Utilities.
"""

from __future__ import annotations
from typing import Optional
import warnings

import torch
from torch import Tensor
from torch import BoolTensor


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
