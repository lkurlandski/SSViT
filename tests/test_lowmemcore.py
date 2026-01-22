"""
Tests.
"""

import math
from typing import Literal
from typing import Optional

import _pytest
import pytest
import torch
from torch import nn
from torch.nn import Embedding
from torch import Tensor

from src.lowmemcore import _lowmem_max_over_time_streaming
from src.lowmemcore import _lowmem_patchwise_max_over_time_streaming
from src.lowmemcore import _gather_wins_via_preprocess_batched
from src.lowmemcore import _scatter_g_to_BC
from src.lowmemcore import _scatter_g_to_BNC


def _identity_preprocess(*xs: torch.Tensor) -> torch.Tensor:
    """
    Works for both call sites:
      - streaming: preprocess(t[:, start:end_ext]) -> (B, L, E)
      - gather:    preprocess(wins)               -> (sum_U, rf, E)
    """
    assert len(xs) == 1
    return xs[0]


def _make_positive_conv(E: int, C: int, K: int, stride: int) -> nn.Conv1d:
    """
    Positive weights + positive input ensures padding/empty-patch behaviors are stable.
    """
    torch.manual_seed(0)
    conv = nn.Conv1d(E, C, kernel_size=K, stride=stride, bias=False)
    with torch.no_grad():
        conv.weight.data = conv.weight.data.abs_() + 0.01
    return conv


def _reference_max_over_time(
    *,
    z: torch.Tensor,      # (B,T,E)
    conv: nn.Conv1d,
    rf: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = conv(z.transpose(1, 2).contiguous())  # (B,C,L_out)
    v, idx = g.max(dim=-1)                    # (B,C)
    pos = idx.to(torch.long) * stride
    pos = pos.clamp(0, max(0, z.shape[1] - rf))
    return v, pos


def _reference_patchwise_max(
    *,
    z: torch.Tensor,          # (B,T,E)
    conv: nn.Conv1d,          # applied to (B,E,T)
    rf: int,
    stride: int,
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference for _lowmem_patchwise_max_over_time_streaming.

    It computes conv over the whole sequence, then assigns each conv output index to a patch:
        patch_idx = floor((pos_end) / patch_size), pos_end = pos + (rf-1)
    then takes max over time within each patch per channel.
    """
    B, T, E = z.shape
    C = conv.out_channels
    dtype = z.dtype
    device = z.device

    patch_size = (T + num_patches - 1) // num_patches

    g = conv(z.transpose(1, 2).contiguous())  # (B,C,L_out)
    _, _, L_out = g.shape

    pos = torch.arange(L_out, device=device, dtype=torch.long) * stride
    pos_end = pos + (rf - 1)

    patch_idx = torch.div(pos_end, patch_size, rounding_mode="floor").clamp(0, num_patches - 1)

    sentinel = torch.finfo(dtype).min
    max_vals = torch.full((B, num_patches, C), sentinel, device=device, dtype=dtype)
    max_pos = torch.zeros((B, num_patches, C), device=device, dtype=torch.long)

    for j in range(num_patches):
        mask = (patch_idx == j)
        if not mask.any():
            continue
        g_j = g[..., mask]                 # (B,C,Lj)
        v_j, idx_local = g_j.max(dim=-1)   # (B,C)

        pos_candidates = pos[mask]
        pos_j = pos_candidates[idx_local]  # (B,C)

        cur_v = max_vals[:, j, :]
        cur_p = max_pos[:, j, :]

        upd = v_j > cur_v
        max_vals[:, j, :] = torch.where(upd, v_j, cur_v)
        max_pos[:, j, :] = torch.where(upd, pos_j, cur_p)

    empty = (max_vals == sentinel)
    if empty.any():
        max_vals = max_vals.masked_fill(empty, 0.0)
        max_pos = max_pos.masked_fill(empty, 0)

    max_pos.clamp_(0, max(0, T - rf))
    return max_vals, max_pos


class TestLowmemMaxOverTimeStreaming:

    @pytest.mark.parametrize("chunk_size", [64, 96, 128, 192, 256])
    def test_matches_reference_safe_overlap(self, chunk_size: int) -> None:
        """
        With safe overlap (rf-1) and aligned stepping, streaming max should exactly match
        the full-materialize reference.
        """
        torch.manual_seed(10)

        B, T, E = 2, 512, 4
        C = 7
        rf = 16
        stride = 8

        overlap = rf // 2
        step = max(1, chunk_size - overlap)
        assert step % stride == 0, "Require aligned stepping to compare exact semantics"

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride)

        def activations_fn(x: torch.Tensor) -> torch.Tensor:
            return conv(x)  # type: ignore[no-any-return]

        max_vals, max_pos = _lowmem_max_over_time_streaming(
            preprocess=_identity_preprocess,
            ts=[z],
            rf=rf,
            first_stride=stride,
            chunk_size=chunk_size,
            overlap=overlap,
            channels=C,
            activations_fn=activations_fn,
        )

        ref_vals, ref_pos = _reference_max_over_time(z=z, conv=conv, rf=rf, stride=stride)

        torch.testing.assert_close(max_vals, ref_vals, rtol=0, atol=0)
        assert torch.equal(max_pos, ref_pos)

    def test_raises_on_short_input(self) -> None:
        torch.manual_seed(11)

        B, T, E = 2, 8, 4
        C = 5
        rf = 16
        stride = 8

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride)

        def activations_fn(x: torch.Tensor) -> torch.Tensor:
            return conv(x)  # type: ignore[no-any-return]

        with pytest.raises(RuntimeError):
            _lowmem_max_over_time_streaming(
                preprocess=_identity_preprocess,
                ts=[z],
                rf=rf,
                first_stride=stride,
                chunk_size=64,
                overlap=rf - 1,
                channels=C,
                activations_fn=activations_fn,
            )


class TestLowmemPatchwiseMaxOverTimeStreaming:

    @pytest.mark.parametrize("chunk_size,overlap", [(512, 0), (128, 64), (192, 64)])
    def test_matches_reference_when_aligned(self, chunk_size: int, overlap: int) -> None:
        """
        Ensures chunking doesn't change results *in an aligned configuration*:
          - stride == rf
          - chunk step is multiple of stride
        """
        torch.manual_seed(1)

        B, T, E = 2, 512, 4
        C = 6
        rf = stride = 8
        N = 16  # patch_size = ceil(512/16)=32, multiple of stride

        step = max(1, chunk_size - overlap)
        assert step % stride == 0, "Test requires aligned chunk stepping."

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride)

        def activations_fn(x: torch.Tensor) -> torch.Tensor:
            return conv(x)  # type: ignore[no-any-return]

        max_vals, max_pos = _lowmem_patchwise_max_over_time_streaming(
            preprocess=_identity_preprocess,
            ts=[z],
            rf=rf,
            first_stride=stride,
            chunk_size=chunk_size,
            overlap=overlap,
            channels=C,
            num_patches=N,
            activations_fn=activations_fn,
        )

        ref_vals, ref_pos = _reference_patchwise_max(z=z, conv=conv, rf=rf, stride=stride, num_patches=N)

        torch.testing.assert_close(max_vals, ref_vals, rtol=0, atol=0)
        assert torch.equal(max_pos, ref_pos)

    def test_empty_patches_zeroed(self) -> None:
        """
        Constructs a regime where many patches have no conv outputs mapped to them
        and verifies they become exactly zero.
        """
        torch.manual_seed(2)

        B, T, E = 2, 64, 4
        C = 5
        rf = stride = 8
        N = 32  # patch_size=2 << stride => many empty patches

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride)

        def activations_fn(x: torch.Tensor) -> torch.Tensor:
            return conv(x)  # type: ignore[no-any-return]

        max_vals, max_pos = _lowmem_patchwise_max_over_time_streaming(
            preprocess=_identity_preprocess,
            ts=[z],
            rf=rf,
            first_stride=stride,
            chunk_size=64,
            overlap=0,
            channels=C,
            num_patches=N,
            activations_fn=activations_fn,
        )

        ref_vals, ref_pos = _reference_patchwise_max(z=z, conv=conv, rf=rf, stride=stride, num_patches=N)

        torch.testing.assert_close(max_vals, ref_vals, rtol=0, atol=0)
        assert torch.equal(max_pos, ref_pos)
        assert (max_vals == 0).any()


class TestScatterGToBC:

    def test_roundtrip_matches_manual(self) -> None:
        """
        Validates gather->conv->scatter reproduces per-channel conv outputs implied by
        positions of shape (B, C).
        """
        torch.manual_seed(12)

        B, T, E = 2, 128, 4
        C = 6
        rf = 8

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride=rf)

        positions = torch.randint(0, T - rf + 1, (B, C), dtype=torch.long)
        positions[:, :2] = positions[:, 0:1]  # force duplicates

        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=_identity_preprocess,
            ts=[z],
            positions=positions,
            rf=rf,
            embedding_dim=E,
        )

        g_all = conv(wins_cat).squeeze(-1)  # (sum_U, C)

        out = _scatter_g_to_BC(
            g_all=g_all,
            meta=meta,
            batch_size=B,
            channels=C,
        )

        out_manual = torch.zeros((B, C), dtype=out.dtype)
        for b in range(B):
            for c in range(C):
                pos = int(positions[b, c].item())
                w = z[b, pos : pos + rf, :].transpose(0, 1).unsqueeze(0)  # (1,E,rf)
                y = conv(w).squeeze(-1).squeeze(0)                        # (C,)
                out_manual[b, c] = y[c]

        torch.testing.assert_close(out, out_manual, rtol=0, atol=1e-6)

    def test_raises_on_inv_length_mismatch(self) -> None:
        torch.manual_seed(13)

        B, C = 2, 4
        g_all = torch.randn(3, C)

        bad_meta = [
            (0, torch.tensor([0, 1, 2, 0, 1], dtype=torch.long), 3),
            (0, torch.tensor([0, 1, 2, 3, 0], dtype=torch.long), 3),
        ]

        with pytest.raises(RuntimeError):
            _scatter_g_to_BC(
                g_all=g_all,
                meta=bad_meta,
                batch_size=B,
                channels=C,
            )


class TestScatterGToBNC:

    def test_roundtrip_matches_manual(self) -> None:
        """
        Validates gather->conv->scatter reproduces per-(patch,channel) conv outputs implied by
        flattened positions of shape (B, N*C).
        """
        torch.manual_seed(3)

        B, T, E = 2, 128, 4
        rf = 8
        N = 4
        C = 6
        K = N * C

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride=rf)

        positions = torch.randint(0, T - rf + 1, (B, K), dtype=torch.long)
        positions[:, :10] = positions[:, :10] // 4 * 4  # add duplicates

        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=_identity_preprocess,
            ts=[z],
            positions=positions,
            rf=rf,
            embedding_dim=E,
        )

        g_all = conv(wins_cat).squeeze(-1)

        out = _scatter_g_to_BNC(
            g_all=g_all,
            meta=meta,
            batch_size=B,
            num_patches=N,
            channels=C,
        )

        out_manual = torch.zeros((B, N, C), dtype=out.dtype)
        for b in range(B):
            for n in range(N):
                for c in range(C):
                    pos = int(positions[b, n * C + c].item())
                    w = z[b, pos : pos + rf, :].transpose(0, 1).unsqueeze(0)  # (1,E,rf)
                    y = conv(w).squeeze(-1).squeeze(0)                        # (C,)
                    out_manual[b, n, c] = y[c]

        torch.testing.assert_close(out, out_manual, rtol=0, atol=1e-6)

    def test_gather_empty_positions_returns_empty(self) -> None:
        torch.manual_seed(4)

        B, T, E = 2, 128, 4
        rf = 8
        z = torch.rand(B, T, E, dtype=torch.float32)

        positions = torch.empty((B, 0), dtype=torch.long)

        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=_identity_preprocess,
            ts=[z],
            positions=positions,
            rf=rf,
            embedding_dim=E,
        )

        assert wins_cat.shape == (0, E, rf)
        assert len(meta) == B

    def test_gather_mask_produces_zeros_when_available(self) -> None:
        """
        Verify:
        (1) masked-out entries are exactly zero after scatter
        (2) masked-in entries match manual conv (within float32 tolerance)
        """
        torch.manual_seed(6)

        B, T, E = 2, 128, 4
        rf = 8
        N = 4
        C = 6
        K = N * C

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride=rf)

        positions = torch.randint(0, T - rf + 1, (B, K), dtype=torch.long)

        mask = torch.zeros((B, K), dtype=torch.bool)
        mask[:, : K // 2] = True

        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=_identity_preprocess,
            ts=[z],
            positions=positions,
            rf=rf,
            embedding_dim=E,
            mask=mask,
        )

        g_all = conv(wins_cat).squeeze(-1)

        out = _scatter_g_to_BNC(
            g_all=g_all,
            meta=meta,
            batch_size=B,
            num_patches=N,
            channels=C,
        )

        # (1) Masked-out entries should be *exactly* zero.
        out_flat = out.reshape(B, K)
        assert torch.all(out_flat[~mask] == 0)

        # (2) Masked-in entries should match manual conv (float32 tolerance).
        out_manual = torch.zeros((B, N, C), dtype=out.dtype)
        for b in range(B):
            for n in range(N):
                for c in range(C):
                    flat = n * C + c
                    if not mask[b, flat]:
                        continue
                    pos = int(positions[b, flat].item())
                    w = z[b, pos : pos + rf, :].transpose(0, 1).unsqueeze(0)
                    y = conv(w).squeeze(-1).squeeze(0)
                    out_manual[b, n, c] = y[c]

        torch.testing.assert_close(out, out_manual, rtol=0, atol=1e-6)
