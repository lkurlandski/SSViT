"""
Utilities.
"""

from __future__ import annotations
import math
import os
import random
from typing import Literal
from typing import NamedTuple
from typing import Optional
import warnings

import numpy as np
import psutil
import torch
from torch import Tensor
from torch import BoolTensor
from torch import ByteTensor
from torch import IntTensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_optimal_num_workers(ncpu: int = psutil.cpu_count(logical=False), ngpu: int = torch.cuda.device_count()) -> int:
    # TODO: its unclear how the CPU check will behave with SLURM and torchrun.
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if ngpu > ncpu:
        raise RuntimeError(f"Number of GPUs ({ngpu}) exceeds number of CPU cores ({ncpu}).")
    return max(0, ncpu // max(1, ngpu) - 1)


def get_optimal_num_worker_threads(num_workers: int = 0, ncpu: int = psutil.cpu_count(logical=False)) -> int:
    # TODO: its unclear how the CPU check will behave with SLURM and torchrun.
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if num_workers - 1 > ncpu:
        raise RuntimeError(f"Number of worker processes ({num_workers} + 1) exceeds number of CPU cores ({ncpu}).")
    if num_workers == 0:
        return ncpu
    return max(1, (ncpu - 1) // num_workers)


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
def packbits(x: BoolTensor, axis: int = -1) -> ByteTensor:
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
def unpackbits(x: ByteTensor, count: int = -1, axis: int = -1) -> BoolTensor:
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


@torch.no_grad()  # type: ignore[misc]
def mask_select_packed_slow(packed: ByteTensor, mask: BoolTensor, axis: int = -1) -> ByteTensor:
    """
    Slice a bit-packed tensor along `axis` using a boolean mask that refers to the
    *unpacked* bit positions, then return the result *re-packed* along that axis.

    Args:
        packed: uint8 (ByteTensor) produced by `packbits` (little-endian).
        mask: is 1D of length `d` (the original bit length before packing).
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected packed uint8 tensor, got {packed.dtype}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"expected bool mask, got {mask.dtype}")
    if mask.ndim != 1:
        raise ValueError("mask must be 1D (one mask for the packed axis)")

    axis = axis if axis >= 0 else packed.ndim + axis
    if not (0 <= axis < packed.ndim):
        raise IndexError("axis out of range")

    # Move bytes axis to the end: [..., B]
    p = packed.movedim(axis, -1)
    B = p.shape[-1]
    d = mask.numel()
    if d > 8 * B:
        raise ValueError(f"mask length ({d}) exceeds available bits ({8*B}) in packed tensor")

    # Indices of kept bits (0..d-1)
    sel = torch.nonzero(mask, as_tuple=False).flatten()  # [k]
    k = sel.numel()

    # If nothing selected, return empty along that axis (bytes dimension becomes 0)
    if k == 0:
        empty = p[..., :0]  # [..., 0]
        return empty.movedim(-1, axis)

    # Map bit positions -> (byte index, bit offset)
    byte_idx = torch.div(sel, 8, rounding_mode='floor')        # [k]
    bit_off  = (sel % 8).to(torch.uint8)                       # [k]

    # Gather referenced bytes then extract bit offsets (little-endian)
    gathered = torch.index_select(p, dim=-1, index=byte_idx)   # [..., k]
    shifts = bit_off.view(*(1,)* (gathered.ndim - 1), k)       # [1,...,k]
    bits = ((gathered >> shifts) & 1).to(torch.uint8)          # [..., k]

    # Re-pack selected bits into bytes along the last dim
    pad = (-k) % 8
    if pad:
        bits = F.pad(bits, (0, pad), value=0)                  # [..., k+pad]
    blocks = bits.unfold(-1, 8, 8)                             # [..., Kbytes, 8]
    weights = (1 << torch.arange(8, device=bits.device, dtype=torch.uint8))  # [8]
    out = (blocks * weights).sum(-1, dtype=torch.uint8)        # [..., Kbytes]

    return out.movedim(-1, axis)


# Helpers for mask_select_packed_fast

class _BitLUTs(NamedTuple):
    popcnt: torch.Tensor           # [256] uint8
    pext:   torch.Tensor           # [256,256] uint8

_LUTS: dict[torch.device, _BitLUTs] = {}

def _get_luts(device: torch.device) -> _BitLUTs:
    if device in _LUTS:
        return _LUTS[device]
    pop = torch.tensor([bin(i).count("1") for i in range(256)],
                       dtype=torch.uint8, device=device)
    pext = torch.empty((256, 256), dtype=torch.uint8, device=device)
    with torch.no_grad():
        for m in range(256):
            pos = [b for b in range(8) if (m >> b) & 1]
            for x in range(256):
                v = 0
                for r, b in enumerate(pos):
                    v |= ((x >> b) & 1) << r
                pext[m, x] = v
    _LUTS[device] = _BitLUTs(popcnt=pop, pext=pext)
    return _LUTS[device]

# optional tiny cache for weights per device (avoids re-alloc each call)
_WEIGHTS: dict[torch.device, torch.Tensor] = {}

def _weights8(device: torch.device) -> torch.Tensor:
    w = _WEIGHTS.get(device)
    if w is None:
        w = (1 << torch.arange(8, device=device, dtype=torch.uint8))
        _WEIGHTS[device] = w
    return w

@torch.no_grad()  # type: ignore[misc]
def mask_select_packed_fast(packed: torch.ByteTensor, mask: torch.BoolTensor, axis: int = -1) -> torch.ByteTensor:
    if packed.dtype is not torch.uint8:
        raise TypeError("packed must be uint8")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must be bool")
    if mask.ndim != 1:
        raise ValueError("mask must be 1-D")

    axis = axis if axis >= 0 else packed.ndim + axis
    if not (0 <= axis < packed.ndim):
        raise IndexError("axis out of range")

    p = packed.movedim(axis, -1).contiguous()  # [..., B]
    B = p.shape[-1]
    device = p.device

    # ensure mask is on same device as packed
    mask = mask.to(device)

    d = mask.numel()
    if d > 8 * B:
        raise ValueError(f"mask length ({d}) exceeds available bits ({8*B})")

    # fast exits
    if d == 0 or not bool(mask.any()):
        return p[..., :0].movedim(-1, axis)
    if d == 8 * B and bool(mask.all()):
        return p.movedim(-1, axis)

    luts = _get_luts(device)

    # pack mask into bytes (little-endian)
    pad = (-d) % 8
    if pad:
        mask = F.pad(mask, (0, pad), value=False)
    m8 = mask.view(-1, 8).to(torch.uint8)                            # [B,8]
    mask_bytes = (m8 * _weights8(device)).sum(dim=1)                 # [B] uint8

    # popcount + prefix sum → start bit positions
    cnt = luts.popcnt[mask_bytes]                                    # [B] uint8
    if B == 0:
        return p[..., :0].movedim(-1, axis)

    start_bits = torch.cumsum(cnt.to(torch.int32), dim=0) - cnt.to(torch.int32)
    total_bits = int(start_bits[-1] + cnt[-1])
    outB = (total_bits + 7) // 8
    if outB == 0:
        return p[..., :0].movedim(-1, axis)

    start_byte = (start_bits // 8).to(torch.long)                    # [B]
    start_bit  = (start_bits % 8).to(torch.uint8)                    # [B]

    # byte-wise “pext”: index with LONG tensors (uint8 indexing is mask semantics)
    MB = mask_bytes.view(*([1] * (p.ndim - 1)), B).expand_as(p)      # [...,B]
    comp = luts.pext[MB.long(), p.long()]                            # [...,B] uint8

    # place each compressed byte into output (≤2 target bytes / source byte)
    sb   = start_byte.view(*([1] * (p.ndim - 1)), B).expand_as(p)    # [...,B] long
    sbit = start_bit.view(*([1] * (p.ndim - 1)), B).expand_as(p)     # [...,B] uint8

    comp16 = comp.to(torch.int16)
    low  = (comp16 << sbit.to(torch.int16))                          # [...,B]
    high = (comp16 >> (8 - sbit).to(torch.int16))                    # [...,B]

    out = torch.zeros((*p.shape[:-1], outB), dtype=torch.int16, device=device)

    # --- FIX: skip bytes where the mask popcount is 0 to avoid OOB at sb==outB ---
    activeB = (cnt != 0)                                             # [B] bool
    act = activeB.view(*([1] * (p.ndim - 1)), B).expand_as(p)        # [...,B] bool

    # Low scatter: clamp indices (extra safety), zero-out inactive contributions
    sb_clamped = sb.clamp_max(outB - 1)
    out.scatter_add_(-1, sb_clamped, torch.where(act, low, torch.zeros_like(low)))

    # High/ spill scatter: only when there is a spill and the next byte exists
    spill = act & (sbit != 0) & (sb < (outB - 1))
    if spill.any():
        out.scatter_add_(
            -1,
            (sb + 1).clamp_max(outB - 1),
            torch.where(spill, high, torch.zeros_like(high)),
        )

    # no overlaps by construction; addition == bitwise OR
    return out.to(torch.uint8).movedim(-1, axis)


mask_select_packed = mask_select_packed_fast


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float | int | bool = 0.0,
    padding_side: Literal["right", "left"] = "right",
    pin_memory: bool = False,
    pad_to_multiple_of: int = 1,
    min_length: int = 0,
) -> Tensor:

    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty list of sequences.")
    if pad_to_multiple_of < 1:
        raise ValueError(f"pad_to_multiple_of must be a positive integer. Got {pad_to_multiple_of}.")

    if not pin_memory and pad_to_multiple_of == 1 and min_length == 0:
        return _pad_sequence(sequences, batch_first, padding_value, padding_side)

    if padding_side != "right":
        raise NotImplementedError("pad_sequence with pin_memory=True requires padding_side='right'.")
    if not batch_first:
        raise NotImplementedError("pad_sequence with pin_memory=True requires batch_first=True.")

    for s in sequences:
        if pin_memory and s.device.type != "cpu":
            raise ValueError("All sequences must be on CPU when pin_memory=True.")
        if s.shape[1:] != sequences[0].shape[1:]:
            raise ValueError("All sequences must have the same shape except for the first dimension.")
        if s.dtype != sequences[0].dtype:
            raise ValueError("All sequences must have the same dtype.")
        if s.device != sequences[0].device:
            raise ValueError("All sequences must be on the same device.")

    batch_size = len(sequences)
    seq_length = max(min_length, math.ceil(max(s.shape[0] for s in sequences) / pad_to_multiple_of) * pad_to_multiple_of)
    other_dims = sequences[0].shape[1:]
    size = (batch_size, seq_length) + tuple(other_dims)

    padded = torch.full(size, fill_value=padding_value, dtype=sequences[0].dtype, device=sequences[0].device, pin_memory=pin_memory)
    for i, s in enumerate(sequences):
        s = s.contiguous() if not s.is_contiguous() else s
        padded[i, :s.shape[0]].copy_(s)
    return padded
