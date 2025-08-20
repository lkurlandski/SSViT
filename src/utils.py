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


def pack_bool_tensor(b: BoolTensor) -> Tensor:
    """
    Packs a two-dimensional boolean tensor into a one-dimensional integer tensor.

    NOTE: unpack(pack(b)) == b, but pack(unpack(x)) != x.
    """
    check_tensor(b, (None, None), torch.bool)
    if not (1 <= b.shape[1] <= 64):
        raise ValueError(f"Cannot pack boolean tensor with {b.shape[1]} channels.")

    # Use torch.int64 because shift operations are not implemented for most uint dtypes.
    weights = (1 << torch.arange(b.shape[1], device=b.device, dtype=torch.int64))
    x = (b.to(torch.int64) * weights).sum(dim=1)

    # This mangles the value, but all we care about is the bits, so its okay.
    dtype = smallest_unsigned_integer_dtype(b.shape[1])
    return x.to(dtype)


def unpack_bit_tensor(x: Tensor, d: int, dtype: torch.dtype = torch.bool) -> Tensor:
    """
    Unpacks a one-dimensional unsigned integer tensor into a two-dimensional boolean tensor.

    NOTE: unpack(pack(b)) == b, but pack(unpack(x)) != x.
    """
    check_tensor(x, (None,), (torch.uint8, torch.uint16, torch.uint32, torch.uint64))
    if not (1 <= d <= 64):
        raise ValueError(f"Cannot unpack bit tensor with {d} channels.")

    # Use torch.int64 because shift operations are not implemented for most uint dtypes.
    x = x.to(torch.int64)
    masks = (1 << torch.arange(d, device=x.device, dtype=torch.int64))
    z = (x.unsqueeze(-1) & masks) != 0

    return z.to(dtype)
