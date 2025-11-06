"""
Utilities for fast, bit-packed boolean arrays and tensors.
"""

from typing import NamedTuple
from typing import Optional
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

from src.utils import check_tensor


__all__ = [
    "pack_bool_tensor",
    "unpack_bit_tensor",
    "packbits",
    "unpackbits",
    "slice_bitpacked_tensor",
]


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


@torch.no_grad()
def pack_bool_tensor(b: Tensor) -> Tensor:
    """
    Packs a boolean tensor along the last dimension into an integer tensor.

    Useful for packing a (T, C) boolean tensor (C ≤ 64) into a (T,) integer tensor.

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
    x: Tensor = (b.to(torch.int64) * weights).sum(dim=-1)

    # Cast down to the minimal unsigned dtype (storage stays compact for transfer)
    out_dtype = smallest_unsigned_integer_dtype(d)
    return x.to(out_dtype)


@torch.no_grad()
def unpack_bit_tensor(x: Tensor, d: int, dtype: torch.dtype = torch.bool) -> Tensor:
    """
    Unpacks an integer tensor into a boolean (or other) tensor along a new last dimension.

    Useful for unpacking a (T,) integer tensor into a (T, C) boolean tensor (C ≤ 64).

    NOTE: unpack(pack(b)) == b, but pack(unpack(x)) != x.
    """
    if not (1 <= d <= 64):
        raise ValueError(f"Cannot unpack bit tensor with {d} channels.")
    if x.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        raise TypeError(f"x must be an unsigned integer tensor, got {x.dtype}")

    # Use torch.int64 workspace for safe shifts
    x = x.to(torch.int64)
    masks = (1 << torch.arange(d, device=x.device, dtype=torch.int64))
    z: Tensor = (x.unsqueeze(-1) & masks) != 0
    return z.to(dtype)


@torch.no_grad()
def packbits(x: Tensor, axis: int = -1) -> Tensor:
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
    out: Tensor = (y8.to(torch.uint8) * weights).sum(dim=-1, dtype=torch.uint8)    # [..., B]

    # Move bytes axis back to original position
    return out.movedim(-1, axis)


@torch.no_grad()
def unpackbits(x: Tensor, count: int = -1, axis: int = -1) -> Tensor:
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
    z: Tensor = bits_flat[..., :count]             # [..., count]

    # Move the expanded bit-axis back
    return z.movedim(-1, axis)


@torch.no_grad()
def slice_bitpacked_tensor(
    packed: Tensor,
    *,
    mask: Optional[Tensor] = None,
    idx: Optional[Tensor] = None,
    ranges: Optional[list[tuple[int, int]]] = None,
    bigchunks: bool = True,
    axis: int = -1,
) -> Tensor:
    """
    Slice a bit-packed tensor using indices that refer the unpacked bit positions.

    Offers three criteria for selecting: a boolean mask, integer indices, or a list of (start, end) ranges.

    NOTE: The backends were written by ChatGPT but tested by humans.

    Args:
        packed: bitpacked tensor to slice.
        mask: boolean mask indicating which bits to keep.
        idx: integer indices referring which bits to keep.
        ranges: list of (start, end) tuples which bits to keep.
        bigchunks: if True, uses backends optimized for large contiguous slices.
        axis: axis along which to slice (the packed bit axis).

    Returns:
        A bit-packed tensor containing only the selected bits along the given axis.
    """

    selectors = [mask is not None, idx is not None, ranges is not None]
    if sum(selectors) != 1:
        raise ValueError("exactly one of mask, idx, or ranges must be provided")

    if packed.dtype != torch.uint8:
        raise TypeError(f"expected packed uint8 tensor, got {packed.dtype}")

    if mask is not None:
        check_tensor(mask, (None,), torch.bool)
        if bigchunks:
            return _slice_bitpacked_tensor_from_mask_bigchunks(packed, mask, axis=axis)
        return _slice_bitpacked_tensor_from_mask_general(packed, mask, axis=axis)

    if idx is not None:
        # NOTE: these have not been implemented yet!
        check_tensor(idx, (None,), (torch.int32, torch.int64))
        if bigchunks:
            return _slice_bitpacked_tensor_from_idx_bigchunks(packed, idx, axis=axis)
        return _slice_bitpacked_tensor_from_idx_general(packed, idx, axis=axis)

    if ranges is not None:
        if not isinstance(ranges, list) or not all(isinstance(r, tuple) and len(r) == 2 for r in ranges):
            raise TypeError("ranges must be a list of (start, end) tuples")
        if bigchunks:
            return _slice_bitpacked_tensor_from_ranges_bigchunks(packed, ranges, axis=axis)
        try:
            return _slice_bitpacked_tensor_from_ranges_general(packed, ranges, axis=axis)
        except NotImplementedError:
            warnings.warn("The kernel _slice_bitpacked_tensor_from_ranges_general has not been implemented yet. Falling back to _slice_bitpacked_tensor_from_ranges_bigchunks.")
            return _slice_bitpacked_tensor_from_ranges_bigchunks(packed, ranges, axis=axis)

    raise RuntimeError(f"Could not determine slicing backend when {type(mask)=} {type(idx)=} and {type(ranges)=}.")


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


# tiny cache for weights per device (avoids re-alloc each call)
_WEIGHTS: dict[torch.device, torch.Tensor] = {}


def _weights8(device: torch.device) -> torch.Tensor:
    w = _WEIGHTS.get(device)
    if w is None:
        w = (1 << torch.arange(8, device=device, dtype=torch.uint8))
        _WEIGHTS[device] = w
    return w


@torch.no_grad()
def _slice_bitpacked_tensor_from_mask_general(packed: Tensor, mask: Tensor, axis: int = -1) -> Tensor:

    axis = axis if axis >= 0 else packed.ndim + axis
    if not (0 <= axis < packed.ndim):
        raise IndexError("axis out of range")

    # Bring target axis to the end as a view: [..., B_src]
    p = packed.movedim(axis, -1)                    # view, no copy
    B_src = p.shape[-1]
    dev = p.device

    # Pack mask into bytes (on same device as p to keep things simple)
    d = mask.numel()
    if d > 8 * B_src:
        raise ValueError(f"mask length ({d}) exceeds available bits ({8*B_src}) in packed tensor")

    mask = mask.to(dev)
    pad = (-d) % 8
    if pad:
        mask = torch.nn.functional.pad(mask, (0, pad), value=False)
    # Number of mask bytes we actually need to touch
    B_mask = mask.numel() // 8
    if B_mask == 0:
        # Return correctly-shaped empty packed tensor
        empty = p[..., :0]
        return empty.movedim(-1, axis)

    m8 = mask.view(B_mask, 8).to(torch.uint8)              # [B_mask, 8]
    weights = _weights8(dev)                                # [8] uint8, cached per device
    mask_bytes = (m8 * weights).sum(dim=1)                  # [B_mask] uint8

    # popcount & prefix
    luts = _get_luts(dev)
    cnt = luts.popcnt[mask_bytes]                           # [B_mask] uint8
    start_bits = torch.cumsum(cnt.to(torch.int32), 0) - cnt.to(torch.int32)
    total_bits = int(start_bits[-1] + cnt[-1])
    outB = (total_bits + 7) // 8
    if outB == 0:
        empty = p[..., :0]
        return empty.movedim(-1, axis)

    start_byte = (start_bits // 8).to(torch.long)           # [B_mask]
    start_bit  = (start_bits % 8).to(torch.uint8)           # [B_mask]

    # Work only on the bytes we actually need (first B_mask source bytes)
    pB = p[..., :B_mask]                                    # [..., B_mask]

    # Byte-wise parallel extract via LUT
    MB = mask_bytes.view(*([1] * (pB.ndim - 1)), B_mask).expand_as(pB)   # [..., B_mask]
    comp = luts.pext[MB.long(), pB.long()]                               # [..., B_mask] uint8

    # Place each compressed byte into output (≤2 target bytes / source byte)
    sb   = start_byte.view(*([1] * (pB.ndim - 1)), B_mask).expand_as(pB) # [..., B_mask] long
    sbit = start_bit.view(*([1] * (pB.ndim - 1)), B_mask).expand_as(pB)  # [..., B_mask] uint8

    comp16 = comp.to(torch.int16)
    low  = (comp16 << sbit.to(torch.int16))                               # [..., B_mask]
    high = (comp16 >> (8 - sbit).to(torch.int16))                         # [..., B_mask]

    out = torch.zeros((*p.shape[:-1], outB), dtype=torch.int16, device=dev)

    # Skip bytes with popcount==0 to avoid sb==outB OOB and needless work
    active = (cnt != 0)
    if bool(active.any()):
        act = active.view(*([1] * (pB.ndim - 1)), B_mask).expand_as(pB)

        # Low scatter
        sb_clamped = sb.clamp_max(outB - 1)
        out.scatter_add_(-1, sb_clamped, torch.where(act, low, torch.zeros_like(low)))

        # Spill scatter
        spill = act & (sbit != 0) & (sb < (outB - 1))
        if bool(spill.any()):
            out.scatter_add_(-1, (sb + 1).clamp_max(outB - 1),
                             torch.where(spill, high, torch.zeros_like(high)))

    return out.to(torch.uint8).movedim(-1, axis)


@torch.no_grad()
def _slice_bitpacked_tensor_from_mask_bigchunks(packed: Tensor, mask: Tensor, axis: int = -1) -> Tensor:

    axis = axis if axis >= 0 else packed.ndim + axis
    if not (0 <= axis < packed.ndim):
        raise IndexError("axis out of range")

    # Bring target axis to the end as a view: [..., B_src]
    p = packed.movedim(axis, -1)
    dev = p.device
    B_src = p.shape[-1]

    d = mask.numel()
    if d > 8 * B_src:
        raise ValueError(f"mask length ({d}) exceeds available bits ({8*B_src})")

    # -------- Run-length encode the mask on CPU --------
    if mask.device.type != "cpu":
        m = mask.cpu()
    else:
        m = mask
    if d == 0 or not bool(m.any()):
        # empty selection: make a correctly-shaped empty tensor
        empty = p[..., :0]
        return empty.movedim(-1, axis)

    # Edges: prepend/append 0 to get starts/ends
    v = m.to(torch.int8)
    edges = torch.diff(torch.cat([torch.tensor([0], dtype=torch.int8), v, torch.tensor([0], dtype=torch.int8)]))
    starts = (edges == 1).nonzero(as_tuple=False).flatten().tolist()   # inclusive bit indices
    ends   = (edges == -1).nonzero(as_tuple=False).flatten().tolist()  # exclusive bit indices
    assert len(starts) == len(ends)
    runs = [(int(s), int(e)) for s, e in zip(starts, ends)]
    total_bits = sum(e - s for s, e in runs)
    outB = (total_bits + 7) // 8

    # Allocate output on device, preserve original layout (axis restored at the end)
    out = torch.zeros((*p.shape[:-1], outB), dtype=torch.int16, device=dev)

    # Helper to pad a last-dim slice on the right with zeros to the given length
    def _pad_right(x: torch.Tensor, target_last_len: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur >= target_last_len:
            return x
        # pad tuple is (pad_left, pad_right) for last dimension
        return F.pad(x, (0, target_last_len - cur))

    # Assemble by streaming runs sequentially into `out`
    dest_bits = 0  # write pointer in bits within the output stream
    for (s, e) in runs:
        L = e - s                     # run length in bits
        if L <= 0:
            continue

        # Source byte window that covers the run
        src_byte0 = s // 8
        src_bitoff = s & 7
        # Number of aligned bytes we need to represent L bits
        n_aligned = (L + 7) // 8

        if src_bitoff == 0:
            # Byte-aligned: just slice exact bytes
            aligned = p[..., src_byte0 : src_byte0 + n_aligned].to(torch.int16)  # [..., n_aligned]
        else:
            # We need (offset + L) bits; cover with this many source bytes, +1 for lookahead
            n_src = (src_bitoff + L + 7) // 8
            src = p[..., src_byte0 : src_byte0 + n_src + 1]                       # [..., <= n_src+1]
            src = _pad_right(src, n_src + 1)                                      # ensure [..., n_src+1]
            s_lo = (src[..., :-1].to(torch.int16) >> src_bitoff)                  # [..., n_src]
            s_hi = (src[..., 1:].to(torch.int16) << (8 - src_bitoff)) & 0xFF      # [..., n_src]
            aligned = (s_lo | s_hi)[..., :n_aligned]                               # [..., n_aligned]

        # Mask off superfluous bits in the last byte of the run
        last_bits = L & 7
        if last_bits:
            mask_last = (1 << last_bits) - 1
            aligned[..., -1] &= mask_last

        # Destination placement
        dst_byte0 = dest_bits // 8
        dst_bitoff = dest_bits & 7

        if dst_bitoff == 0:
            # Pure byte copy into out[..., dst_byte0 : dst_byte0 + n_aligned]
            out[..., dst_byte0 : dst_byte0 + n_aligned] += aligned
        else:
            # Low part
            low = (aligned << dst_bitoff)                                   # [..., n_aligned]
            lo_end = min(dst_byte0 + n_aligned, outB)
            out[..., dst_byte0 : lo_end] += low[..., : (lo_end - dst_byte0)]

            # Spill/high part
            high = (aligned >> (8 - dst_bitoff))                             # [..., n_aligned]
            hi_start = dst_byte0 + 1
            hi_end = min(hi_start + n_aligned, outB)
            if hi_start < hi_end:
                out[..., hi_start : hi_end] += high[..., : (hi_end - hi_start)]

        dest_bits += L

    # Convert back to uint8 and restore axis
    return out.to(torch.uint8).movedim(-1, axis)


@torch.no_grad()
def _slice_bitpacked_tensor_from_idx_general(packed: Tensor, idx: Tensor, axis: int = -1) -> Tensor:
    raise NotImplementedError()


@torch.no_grad()
def _slice_bitpacked_tensor_from_idx_bigchunks(packed: Tensor, idx: Tensor, axis: int = -1) -> Tensor:
    raise NotImplementedError()


@torch.no_grad()
def _slice_bitpacked_tensor_from_ranges_general(packed: Tensor, ranges: list[tuple[int, int]], axis: int = -1) -> Tensor:
    raise NotImplementedError()


@torch.no_grad()
def _slice_bitpacked_tensor_from_ranges_bigchunks(packed: Tensor, ranges: list[tuple[int, int]], axis: int = -1) -> Tensor:
    axis = axis if axis >= 0 else packed.ndim + axis
    if not (0 <= axis < packed.ndim):
        raise IndexError("axis out of range")

    # Bring target axis to the end as a view: [..., B_src]
    p = packed.movedim(axis, -1)
    dev = p.device
    B_src = p.shape[-1]
    d_bits = 8 * B_src

    # Validate & clamp to available bits
    clean: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if lo < 0 or hi < 0:
            raise ValueError("ranges must be non-negative")
        if lo >= hi:
            continue
        if lo >= d_bits:
            continue
        hi = min(hi, d_bits)
        clean.append((lo, hi))

    if not clean:
        # produce correctly-shaped empty packed tensor
        empty = p[..., :0]
        return empty.movedim(-1, axis)

    # Optional ultra-fast path: if every run is byte-aligned and lengths are whole bytes
    all_byte_aligned = (
        clean[0][0] % 8 == 0 and
        all((lo % 8 == 0) and ((hi - lo) % 8 == 0) for lo, hi in clean)
    )
    if all_byte_aligned:
        pieces = [p[..., (lo // 8) : (lo // 8) + ((hi - lo) // 8)] for lo, hi in clean]
        out = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        return out.movedim(-1, axis)

    # General run path: stream bytes with at most one shift per run
    total_bits = sum(hi - lo for lo, hi in clean)
    outB = (total_bits + 7) // 8
    out = torch.zeros((*p.shape[:-1], outB), dtype=torch.int16, device=dev)  # sum==OR

    def _pad_right(x: torch.Tensor, target_last_len: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur >= target_last_len:
            return x
        return F.pad(x, (0, target_last_len - cur))

    dest_bits = 0
    for lo, hi in clean:
        L = hi - lo
        src_byte0 = lo >> 3
        src_bitoff = lo & 7
        n_aligned = (L + 7) >> 3  # bytes that hold L bits once aligned

        if src_bitoff == 0:
            aligned = p[..., src_byte0 : src_byte0 + n_aligned].to(torch.int16)
        else:
            # need a lookahead byte for spill
            n_src = (src_bitoff + L + 7) >> 3
            src = p[..., src_byte0 : src_byte0 + n_src + 1]
            src = _pad_right(src, n_src + 1)
            s_lo = (src[..., :-1].to(torch.int16) >> src_bitoff)              # [..., n_src]
            s_hi = (src[..., 1:].to(torch.int16) << (8 - src_bitoff)) & 0xFF  # [..., n_src]
            aligned = (s_lo | s_hi)[..., :n_aligned]                           # [..., n_aligned]

        # Trim extra bits in the last byte of this run
        last_bits = L & 7
        if last_bits:
            aligned[..., -1] &= (1 << last_bits) - 1

        dst_byte0 = dest_bits >> 3
        dst_bitoff = dest_bits & 7

        if dst_bitoff == 0:
            out[..., dst_byte0 : dst_byte0 + n_aligned] += aligned
        else:
            low = (aligned << dst_bitoff)
            lo_end = min(dst_byte0 + n_aligned, outB)
            out[..., dst_byte0 : lo_end] += low[..., : (lo_end - dst_byte0)]

            high = (aligned >> (8 - dst_bitoff))
            hi_start = dst_byte0 + 1
            hi_end = min(hi_start + n_aligned, outB)
            if hi_start < hi_end:
                out[..., hi_start : hi_end] += high[..., : (hi_end - hi_start)]

        dest_bits += L

    return out.to(torch.uint8).movedim(-1, axis)
