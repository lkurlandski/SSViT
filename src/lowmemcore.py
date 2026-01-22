"""
Core components for constant memory operations.
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
import math
import os
import sys
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Protocol
import warnings

import torch
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F
from torch import nn
from torch import Tensor

from src.utils import check_tensor
from src.utils import TensorError
from src.utils import pad_sequence


DISABLE_LOW_MEMORY_PATHS = os.environ.get("DISABLE_LOW_MEMORY_PATHS", "0") == "1"
if DISABLE_LOW_MEMORY_PATHS:
    print("[WARN] Low-memory paths are disabled via DISABLE_LOW_MEMORY_PATHS=1")


def functional_forward(z: Tensor, *, module: nn.Module, fp32: bool = False) -> Tensor:

    def check(components: list[Optional[Tensor]]) -> None:
        components = [w_b for w_b in components if w_b is not None]
        if any(w_b.dtype != torch.float32 for w_b in components):
            warnings.warn("Some weights and/or biases are not in float32, which is unexpected, when `fp32=True`.")

    if isinstance(module, nn.Conv1d):
        w = module.weight.detach()
        b = module.bias.detach() if module.bias is not None else None

        if fp32:
            z = z.to(torch.float32)
            check([w, b])

        z = F.conv1d(z, w, b, **_get_conv_kwds(module))

        return z

    if hasattr(module, "forward_functional"):
        z = module.forward_functional(z, fp32=fp32)  # type: ignore[operator]
        if not isinstance(z, Tensor):
            raise TensorError(f"forward_functional of {type(module)} did not return a Tensor.")
        return z

    raise NotImplementedError(f"functional_forward does not support {type(module)} yet.")


def _get_conv_kwds(conv: nn.Conv1d) -> dict[str, Any]:
    return {
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
        "groups": conv.groups,
    }


def _check_lowmem_config(kernel_size: int, stride: int, chunk_size: int, overlap: int, mode: Literal["raise", "warn", "ignore"] = "warn") -> bool:
    """
    Check low-memory configuration for potential issues and possibly raise/warn.

    Args:
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        chunk_size: Low-memory chunk size.
        overlap: Low-memory chunk overlap.
        mode: One of "raise", "warn", or "ignore".

    Returns:
        True if configuration is without concern.
    """
    if mode not in ("raise", "warn", "ignore"):
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'raise', 'warn', or 'ignore'.")

    status = True

    def process(msg: str) -> None:
        nonlocal status
        if mode == "raise":
            raise ValueError(msg)
        if mode == "warn":
            warnings.warn(msg)
        status = False

    if overlap < kernel_size / 2:
        msg = f"Overlap {overlap} is less than half the kernel size {kernel_size}. Pooling may be impacted windowing issues. Consider overlap >= kernel_size / 2."
        process(msg)

    if max(1, chunk_size - overlap) % stride != 0:
        msg = f"Chunk stepping {max(1, chunk_size - overlap)} is not aligned with stride {stride}. Pooling window positions may differ between chunks. Consider (chunk_size - overlap) % stride == 0."
        process(msg)

    if overlap % stride != 0:
        msg = f"Overlap {overlap} is not divisible by stride {stride}. Window positions near chunk edges may differ between chunks. Consider overlap % stride == 0."
        process(msg)

    return status


class LowMemoryNetworkMixin(nn.Module, ABC):
    """
    Mixin for neural networks supporting low-memory paths.
    """

    def forward(self, z: Optional[Tensor] = None, preprocess: Optional[PreprocessFn] = None, ts: Optional[Sequence[Tensor]] = None) -> Tensor:
        """
        Dispatches to either `forward_embeddings` or `forward_streaming` based on provided inputs.

        Args:
            z: input tensor of shape (B, T, E) (`forward_embeddings`).
            preprocess: callable to compute the input tensor (`forward_streaming`).
            ts: optional tensor arguments of shape (B, T, *) for `preprocess` (`forward_streaming`).

        Returns:
            Output tensor of shape (B, ...).
        """
        self.check_inputs(z, ts)

        if z is not None:
            z = self.forward_embeddings(z)
        elif preprocess is not None and ts is not None:
            if DISABLE_LOW_MEMORY_PATHS:
                z = self.forward_embeddings(preprocess(*ts))
            else:
                z = self.forward_streaming(preprocess=preprocess, ts=ts)
        else:
            raise ValueError("Either `z` or both `preprocess` and `ts` must be provided.")

        self.check_output(z)
        return z

    @abstractmethod
    def forward_embeddings(self, z: Tensor) -> Tensor:
        """Dense, memory intensive path."""

    @abstractmethod
    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        """Streaming, low-memory path."""

    def check_inputs(self, z: Optional[Tensor] = None, ts: Optional[Sequence[Tensor]] = None) -> None:
        """Validate the inputs to `forward`."""

    def check_output(self, z: Tensor) -> None:
        """Validate the output of `forward`."""


class LowMemoryPreprocessorMixin(nn.Module, ABC):
    """
    Mixin for neural networks that can be used before a low-memory network.
    """

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Forward method of the network."""

    @abstractmethod
    @torch.no_grad()
    def forward_functional(self, *args: Any, **kwds: Any) -> Any:
        """Fully functional interface to the network with aggressive gradient detachment."""


class PreprocessFn(Protocol):
    """
    Preprocessing function for MalConv backbones.
    """

    def __call__(self, *parts: Tensor) -> Tensor:
        """
        Processes temporal input tensors to produce temporal embeddings.

        Args:
            parts: input parts, each of shape (B, T, *).

        Returns:
            z: output tensor of shape (B, T, E).
        """
        ...


def _lowmem_patchwise_max_over_time_dispatched(
    *,
    preprocess: PreprocessFn,
    ts: Sequence[Tensor],                 # each (B,T,*)
    rf: int,
    stride: int,
    patch_size: int,
    num_patches: int,
    channels: int,
    activations_fn: Callable[[Tensor], Tensor],  # (M,E,L)->(M,C,L_out)
    patch_active: torch.Tensor,           # (B,N) bool
    patch_batch_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """
    Compute patchwise (max_vals, max_pos) for ONLY the active (b,j) patches.

    Returns:
      max_vals: (B,N,C) with zeros for inactive/invalid patches
      max_pos:  (B,N,C) with zeros for inactive/invalid patches

    Strategy:
      - For each active (b,j), slice the minimal input span needed for patch j:
          start = max(0, j*patch_size - (rf-1))
          end   = min(T, (j+1)*patch_size)
        (This contains all windows whose end can fall in patch j, but also includes some
         windows that end in patch j-1; those are filtered out by patch_idx==j.)
      - Batch a set of patch slices (micro-batch) into (M,Lmax,*) with padding using an
        extra all-zeros time step at index T.
      - Run preprocess + conv once per micro-batch.
      - Mask conv outputs to (a) valid L_out for each slice and (b) patch_idx==j.
      - Take max and argmax to produce per-(patch,channel) maxima and positions.
    """
    if not ts or any(t.shape[0] != ts[0].shape[0] for t in ts):
        raise ValueError("All tensors in `ts` must share batch dim.")
    B, T = ts[0].shape[:2]
    if T < rf:
        raise RuntimeError(f"Input sequence length {T} < receptive field {rf}")
    if patch_size <= 0:
        raise ValueError(f"{patch_size=} must be positive.")
    if stride <= 0:
        raise ValueError(f"{stride=} must be positive.")
    check_tensor(patch_active, (B, num_patches), torch.bool)
    patch_active = patch_active.to(device=ts[0].device)

    device = ts[0].device

    # Outputs default to zeros (inactive patches stay zero)
    max_vals = ts[0].new_zeros((B, num_patches, channels), dtype=torch.float32).to(device=device)
    max_pos  = torch.zeros((B, num_patches, channels), device=device, dtype=torch.long)

    # Active patch list: (M,2) = (b,j)
    bj = patch_active.nonzero(as_tuple=False)
    if bj.numel() == 0:
        return max_vals, max_pos

    # Precompute per-task slice bounds
    b_idx = bj[:, 0]
    j_idx = bj[:, 1]

    patch_start = (j_idx * patch_size).to(torch.long)                 # (M,)
    patch_end   = torch.minimum((j_idx + 1) * patch_size, torch.tensor(T, device=device)).to(torch.long)  # (M,)
    start = torch.maximum(patch_start - (rf - 1), torch.zeros_like(patch_start))  # (M,)
    end   = patch_end                                                 # (M,)
    lengths = (end - start).to(torch.long)                            # (M,)

    # Some patches can be too short to produce any conv output; we'll just leave zeros there.
    valid_task = lengths >= rf
    if not valid_task.any():
        return max_vals, max_pos

    b_idx = b_idx[valid_task]
    j_idx = j_idx[valid_task]
    start = start[valid_task]
    end   = end[valid_task]
    lengths = lengths[valid_task]

    M_total = int(b_idx.numel())

    # Append a single all-zero timestep at index T for padding.
    # This assumes your preprocess treats all-zero raw inputs as padding reasonably
    # (e.g., embedding padding_idx=0, or zero features -> neutral).
    ts_pad: list[Tensor] = []
    for t in ts:
        pad = t.new_zeros((B, 1) + t.shape[2:])
        ts_pad.append(torch.cat([t, pad], dim=1))  # (B,T+1,*)

    # Process in micro-batches of patches
    for off in range(0, M_total, patch_batch_size):
        sl = slice(off, min(off + patch_batch_size, M_total))

        b_mb = b_idx[sl]          # (m,)
        j_mb = j_idx[sl]          # (m,)
        s_mb = start[sl]          # (m,)
        e_mb = end[sl]            # (m,)  # noqa: F841
        L_mb = lengths[sl]        # (m,)
        m = int(b_mb.numel())

        Lmax = int(L_mb.max().item())
        if Lmax < rf:
            continue

        ar = torch.arange(Lmax, device=device, dtype=torch.long)              # (Lmax,)
        time_ix = s_mb.unsqueeze(1) + ar.unsqueeze(0)                         # (m,Lmax)
        valid_time = ar.unsqueeze(0) < L_mb.unsqueeze(1)                      # (m,Lmax)
        time_ix = torch.where(valid_time, time_ix, torch.full_like(time_ix, T))  # pad index T

        batch_ix = b_mb.unsqueeze(1).expand(m, Lmax)                          # (m,Lmax)

        # Gather + preprocess: (m,Lmax,*) -> (m,Lmax,E)
        wins: list[Tensor] = []
        for tpad in ts_pad:
            wins.append(tpad[batch_ix, time_ix, ...].contiguous())            # (m,Lmax,*)

        z = preprocess(*wins)                                                 # (m,Lmax,E)
        z = z.transpose(1, 2).contiguous()                                    # (m,E,Lmax)

        with torch.no_grad():
            g = activations_fn(z)                                             # (m,C,L_out_max)
            _, _, L_out_max = g.shape

            # Valid conv outputs per slice
            # L_out_i = floor((L_i - rf)/stride) + 1
            L_out_i = torch.div((L_mb - rf), stride, rounding_mode="floor") + 1  # (m,)
            idx = torch.arange(L_out_max, device=device, dtype=torch.long)       # (L_out_max,)
            valid_out = idx.unsqueeze(0) < L_out_i.unsqueeze(1)                  # (m,L_out_max)

            # Compute patch assignment of each conv output (global indices)
            # pos_end = (start + idx*stride) + (rf-1)
            pos_end = s_mb.unsqueeze(1) + idx.unsqueeze(0) * stride + (rf - 1)   # (m,L_out_max)
            patch_idx = torch.div(pos_end, patch_size, rounding_mode="floor").clamp(0, num_patches - 1)

            in_this_patch = patch_idx == j_mb.unsqueeze(1)                       # (m,L_out_max)
            keep = valid_out & in_this_patch                                    # (m,L_out_max)

            # Mask g so only outputs belonging to patch j remain
            neg_inf = torch.finfo(g.dtype).min
            g_masked = g.masked_fill(~keep.unsqueeze(1), neg_inf)                # (m,C,L_out_max)

            v, arg = g_masked.max(dim=-1)                                       # (m,C), (m,C)
            has_any = keep.any(dim=1)                                           # (m,)

            # If no valid outputs for this patch (should be rare), force to zero.
            if (~has_any).any():
                v = torch.where(has_any.unsqueeze(1), v, v.new_zeros((m, channels)))
                arg = torch.where(has_any.unsqueeze(1), arg, arg.new_zeros((m, channels), dtype=torch.long))

            pos = s_mb.unsqueeze(1) + arg * stride                              # (m,C)
            pos = pos.clamp(0, max(0, T - rf))

        # Write back into (B,N,C)
        max_vals[b_mb, j_mb, :] = v.to(max_vals.dtype)
        max_pos[b_mb, j_mb, :]  = pos

    return max_vals, max_pos


def _lowmem_patchwise_max_over_time_streaming(
    *,
    preprocess: PreprocessFn,
    ts: Sequence[Tensor],
    rf: int,
    first_stride: int,
    chunk_size: int,
    overlap: int,
    channels: int,
    num_patches: int,
    activations_fn: Callable[[Tensor], Tensor],  # (B,E,L)->(B,C,L_out)
    patch_active: Optional[torch.Tensor] = None,  # (B, N) bool; update maxima only where True
) -> tuple[Tensor, Tensor]:
    """
    Streaming version: never materializes (B,T,E). Returns
      max_vals: (B, N, C)
      max_pos:  (B, N, C)
    with memory O(B · N · C), independent of T.

    Patches are defined by *input*-space indices:
      patch_size = ceil(T / num_patches)
      patch j covers [j * patch_size, (j+1) * patch_size)
    """

    if not ts or any(t.shape[0] != ts[0].shape[0] for t in ts):
        raise ValueError("All tensors in `ts` must share batch dim.")
    B, T = ts[0].shape[0], ts[0].shape[1]
    if T < rf:
        raise RuntimeError(f"Input sequence length {T} < receptive field {rf}")

    patch_size = (T + num_patches - 1) // num_patches

    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    sentinel: Optional[float] = None

    max_vals: torch.Tensor
    max_pos: torch.Tensor

    step = max(1, chunk_size - overlap)
    start = 0

    if patch_active is not None:
        check_tensor(patch_active, (B, num_patches), torch.bool)
        patch_active = patch_active.to(device=ts[0].device)

    with torch.no_grad():
        while start < T:
            end = min(start + chunk_size, T)
            end_ext = min(end + overlap, T)

            # Preprocess slice of inputs: (B, L, *) -> (B, L, E)
            slices = [t[:, start:end_ext] for t in ts]
            z_chunk = preprocess(*slices)                   # (B, L, E)

            # On the first iteration of the loop, set up storages.
            if dtype is None or device is None or sentinel is None:
                dtype = z_chunk.dtype
                device = z_chunk.device
                sentinel = torch.finfo(dtype).min
                max_vals = torch.full((B, num_patches, channels), sentinel, device=device, dtype=dtype)
                max_pos = torch.zeros((B, num_patches, channels), device=device, dtype=torch.long)

            z_chunk = z_chunk.transpose(1, 2).contiguous()  # (B, E, L)

            if z_chunk.shape[-1] >= rf:
                g = activations_fn(z_chunk)                 # (B, C, L_out)
                B_, C_, L_out = g.shape
                assert B_ == B and C_ == channels

                idx = torch.arange(L_out, device=device)        # (L_out,)
                pos = start + idx * first_stride                # (L_out,)
                pos_end = pos + (rf - 1)                        # (L_out,)
                # Map each conv output position to a patch index
                patch_idx = torch.div(pos_end, patch_size, rounding_mode="floor")
                patch_idx.clamp_(0, num_patches - 1)

                # For each patch, update maxima
                for j in range(num_patches):
                    mask = (patch_idx == j)                    # (L_out,)
                    if not mask.any():
                        continue
                    # If nobody in the batch is active for this patch, skip work.
                    if patch_active is not None and not patch_active[:, j].any():
                        continue

                    g_j = g[..., mask]                         # (B, C, L_j)
                    v_j, idx_j_local = g_j.max(dim=-1)         # (B, C)

                    pos_candidates = pos[mask]                 # (L_j,)
                    # Map local argmax indices to global positions (B,C)
                    pos_j = pos_candidates[idx_j_local]        # (B, C)

                    cur_v = max_vals[:, j, :]                  # (B, C)
                    cur_p = max_pos[:, j, :]

                    upd = v_j > cur_v
                    if patch_active is not None:
                        upd = upd & patch_active[:, j].unsqueeze(1)
                    max_vals[:, j, :] = torch.where(upd, v_j, cur_v)
                    max_pos[:, j, :]  = torch.where(upd, pos_j, cur_p)

            if end == T:
                break
            start += step

    # If the sequence is shorter than N * P, we may wind up with empty patches.
    # These will have max_vals == sentinel; set them to zero to avoid blowing up downstream.
    assert sentinel is not None
    mask = (max_vals == sentinel)
    if mask.any():
        max_vals = max_vals.masked_fill(mask, 0.0)
        max_pos = max_pos.masked_fill(mask, 0)

    # Also, clamp the positions to valid window starts.
    max_pos.clamp_(0, max(0, T - rf))

    return max_vals, max_pos


def _scatter_g_to_BNC(
    *,
    g_all: torch.Tensor,                                     # (sum_U, C)
    meta: list[tuple[int, torch.Tensor, int]],
    batch_size: int,
    num_patches: int,
    channels: int,
) -> torch.Tensor:
    """
    Maps concatenated per-window activations back to (B, N, C) using meta.

    meta[b] = (start, inv_flat, U_b) where:
      - start:   offset into g_all for sample b
      - inv_flat: (N * C,) tensor mapping each (patch, channel) to a unique
                  window index in [0, U_b).
      - U_b:     number of unique windows for sample b.

    g_all[start:start + U_b] has shape (U_b, C). For each (patch j, channel c),
    we want g_all_b[ inv_flat[j*C + c], c ].
    """
    out = torch.zeros(
        (batch_size, num_patches, channels),
        device=g_all.device,
        dtype=g_all.dtype,
    )

    for b, (start, inv_flat, u_b) in enumerate(meta):
        if inv_flat.numel() != num_patches * channels:
            raise RuntimeError(
                f"Inverse indices length mismatch for batch {b}: "
                f"{inv_flat.numel()} vs {num_patches * channels}"
            )

        if u_b == 0:
            continue

        g_b = g_all[start:start + u_b]     # (U_b, C)
        inv = inv_flat.view(num_patches, channels)  # (N, C)

        active = inv >= 0
        row_idx = inv.clamp(min=0)
        col_idx = torch.arange(channels, device=g_all.device).view(1, -1).expand(num_patches, -1)
        gathered = g_b[row_idx, col_idx]                 # (N, C), junk where inv=-1
        out[b] = torch.where(active, gathered, out[b])

    return out


# -------------------------------------------------------------------------------- #
# MalConv Helpers
# -------------------------------------------------------------------------------- #

def _lowmem_max_over_time_streaming(
    *,
    preprocess: PreprocessFn,
    ts: Sequence[Tensor],
    rf: int,
    first_stride: int,
    chunk_size: int,
    overlap: int,
    channels: int,
    activations_fn: Callable[[torch.Tensor], torch.Tensor],  # (B,E,L)->(B,C,L_out)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming version: never materializes (B,T,E). Returns (max_vals, max_pos) with memory O(B·C).
    """
    if not ts or any(t.shape[0] != ts[0].shape[0] for t in ts):
        raise ValueError("All tensors in `ts` must share batch dim.")
    B, T = ts[0].shape[0], ts[0].shape[1]
    if T < rf:
        raise RuntimeError(f"Input sequence length {T} < receptive field {rf}")

    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    max_vals: torch.Tensor
    max_pos: torch.Tensor

    step = max(1, chunk_size - overlap)
    start = 0

    with torch.no_grad():
        while start < T:
            # Ensure windows crossing right edge are seen
            end = min(start + chunk_size, T)
            end_ext = min(end + overlap, T)

            # Preprocess slice of inputs
            slices = [t[:, start:end_ext] for t in ts]
            z_chunk = preprocess(*slices)                   # (B, L, E)
            if dtype is None or device is None:
                dtype = z_chunk.dtype
                device = z_chunk.device
                max_vals = torch.full((B, channels), torch.finfo(dtype).min, device=device, dtype=dtype)
                max_pos = torch.zeros((B, channels), device=device, dtype=torch.long)
            z_chunk = z_chunk.transpose(1, 2).contiguous()  # (B, E, L)

            if z_chunk.shape[-1] >= rf:
                g = activations_fn(z_chunk)                  # (B, C, L_out)
                v, idx = g.max(dim=-1)                       # (B, C)
                pos = start + idx * first_stride
                upd = v > max_vals
                max_vals = torch.where(upd, v, max_vals)
                max_pos  = torch.where(upd, pos, max_pos)

            if end == T:
                break
            start += step

    max_pos.clamp_(0, max(0, T - rf))
    return max_vals, max_pos


def _gather_wins_via_preprocess_batched(
    *,
    preprocess: PreprocessFn,
    ts: Sequence[Tensor],               # (B, T, *)
    positions: torch.Tensor,            # (B, C)
    rf: int,
    embedding_dim: int,
    mask: Optional[torch.Tensor] = None,  # (B, C) bool; only these entries participate
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor, int]]]:
    B, T = ts[0].shape[:2]
    device = ts[0].device

    if mask is not None:
        check_tensor(mask, (B, positions.shape[1]), torch.bool)
        mask = mask.to(device=device)

    posuniq_list, inv_list, u_counts = [], [], []
    for b in range(B):
        pos_b = positions[b]
        if mask is None:
            pos_sel = pos_b
            sel = None
        else:
            sel = mask[b]
            pos_sel = pos_b[sel]

        if pos_sel.numel() == 0:
            posuniq_b = pos_b.new_empty((0,), dtype=torch.long)
            inv_full = pos_b.new_full((pos_b.numel(),), -1, dtype=torch.long)
            U_b = 0
        else:
            posuniq_b, inv_sel = torch.unique(pos_sel, sorted=True, return_inverse=True)
            inv_full = pos_b.new_full((pos_b.numel(),), -1, dtype=torch.long)
            if sel is None:
                inv_full[:] = inv_sel.to(torch.long)
            else:
                inv_full[sel] = inv_sel.to(torch.long)
            U_b = int(posuniq_b.numel())

        posuniq_list.append(posuniq_b)
        inv_list.append(inv_full)
        u_counts.append(U_b)

    meta: list[tuple[int, torch.Tensor, int]] = []

    sum_U = int(sum(u_counts))
    if sum_U == 0:
        wins_empty = ts[0].new_empty((0, rf) + tuple(ts[0].shape[2:]))
        z = preprocess(*([wins_empty] * len(ts)))            # (0, rf, E)
        z = z.transpose(1, 2).contiguous()                 # (0, E, rf)
        meta = [(0, inv_list[b], 0) for b in range(B)]
        return z, meta

    arange_rf = torch.arange(rf, device=device, dtype=torch.int32)
    batch_ix_chunks, time_ix_chunks = [], []
    offset = 0

    for b, posuniq_b in enumerate(posuniq_list):
        U_b = int(posuniq_b.numel())
        if U_b == 0:
            meta.append((offset, inv_list[b].to(torch.long), 0))
            continue
        time_ix_b  = (posuniq_b.to(torch.int32).unsqueeze(1) + arange_rf.unsqueeze(0))  # (U_b, rf)
        batch_ix_b = torch.full_like(time_ix_b, b, dtype=torch.int32)                    # (U_b, rf)
        time_ix_chunks.append(time_ix_b)
        batch_ix_chunks.append(batch_ix_b)
        # NOTE: meta matches your original scatter: (start, inv, u_b)
        meta.append((offset, inv_list[b].to(torch.long), U_b))
        offset += U_b

    batch_ix = torch.cat(batch_ix_chunks, dim=0).long()  # (ΣU, rf)
    time_ix  = torch.cat(time_ix_chunks,  dim=0).long()  # (ΣU, rf)

    wins: list[Tensor] = []
    for t in ts:
        wins.append(t[batch_ix, time_ix, ...].contiguous())  # (ΣU, rf, *)

    # No inner autocast needed if your outer loop already set it
    z = preprocess(*wins)                                     # (ΣU, rf, E)
    if z.shape[-1] != embedding_dim:
        raise RuntimeError(f"preprocess returned wrong E: {z.shape[-1]} vs {embedding_dim}")

    wins_cat = z.transpose(1, 2).contiguous()                 # (ΣU, E, rf)
    return wins_cat, meta


def _scatter_g_to_BC(
    *,
    g_all: torch.Tensor,                                     # (sum_U, C)
    meta: list[tuple[int, torch.Tensor, int]],
    batch_size: int,
    channels: int,
) -> torch.Tensor:
    """
    Maps concatenated per-window activations back to (B, C) using meta.
    """
    out = torch.empty((batch_size, channels), device=g_all.device, dtype=g_all.dtype)
    for b, (start, inv, u_b) in enumerate(meta):
        if not inv.numel() == channels:
            raise RuntimeError(f"Inverse indices length mismatch: {inv.numel()} vs {channels}")
        g_b = g_all[start:start + u_b]                       # (U_b, C)
        out[b] = g_b[inv, torch.arange(channels, device=g_all.device)]
    return out
