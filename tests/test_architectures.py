"""
Tests.
"""

import inspect
import math
from typing import Literal
from typing import Optional

import _pytest
import pytest
import torch
from torch import nn
from torch.nn import Embedding
from torch import Tensor

from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import SinusoidalPositionalEncoding
from src.architectures import PatchEncoder
from src.architectures import ViT
from src.architectures import MalConvBase
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import MalConvClassifier
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier
from src.architectures import PatchPositionalityEncoder
from src.architectures import _lowmem_max_over_time_streaming
from src.architectures import _lowmem_patchwise_max_over_time_streaming
from src.architectures import _gather_wins_via_preprocess_batched
from src.architectures import _scatter_g_to_BC
from src.architectures import _scatter_g_to_BNC
from src.utils import seed_everything


seed_everything(0)


def _zero_grads(*modules: nn.Module) -> None:
    for m in modules:
        if m is None:
            continue
        if isinstance(m, nn.Module):
            for p in m.parameters(recurse=True):
                if p is not None and p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()


def _clear_grads(*modules: nn.Module) -> None:
    for m in modules:
        if m is None:
            continue
        for p in m.parameters(recurse=True):
            p.grad = None


class TestClassificationHead:
    B = 4

    @pytest.mark.parametrize("input_size", [8, 16, 32])
    @pytest.mark.parametrize("num_classes", [2, 4, 8])
    @pytest.mark.parametrize("hidden_size", [-1, 8, 16])
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_forward(self, input_size: int, num_classes: int, hidden_size: int, num_layers: int) -> None:
        if num_layers > 1 and hidden_size <= 0:
            with pytest.raises(ValueError):
                ClassifificationHead(input_size, num_classes, hidden_size, num_layers)
            return

        model = ClassifificationHead(input_size, num_classes, hidden_size, num_layers)
        z = torch.randn(self.B, input_size)
        z = model.forward(z)
        assert z.shape == (self.B, num_classes)


class TestSinusoidalPositionalEncoding:

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257])
    @pytest.mark.parametrize("embedding_dim", [2, 4])
    def test_forward(self, batch_size: int, seq_length: int, embedding_dim: int) -> None:
        model = SinusoidalPositionalEncoding(embedding_dim=embedding_dim)
        x = torch.rand((batch_size, seq_length, embedding_dim))
        z = model.forward(x)
        assert z.shape == (batch_size, seq_length, embedding_dim)

    @pytest.mark.parametrize("embedding_dim", [-1, 0, 1, 3])
    def test_invalid_configuration(self, embedding_dim: int) -> None:
        with pytest.raises(ValueError):
            SinusoidalPositionalEncoding(embedding_dim=embedding_dim)


class TestPatchEncoder:

    @pytest.mark.parametrize("seq_length", list(range(258)))
    @pytest.mark.parametrize("patch_size", [0, 1, 2, 3, 4, 5, 11, 17, 53, 255, 256, 257, 258, None])
    @pytest.mark.parametrize("num_patches", [0, 1, 2, 3, 4, 5, 11, 17, 53, 255, 256, 257, 258, None])
    def test_patch_dims(self, seq_length: int, patch_size: Optional[int], num_patches: Optional[int]) -> None:

        if seq_length <= 0:
            with pytest.raises(ValueError):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return
        
        if patch_size is not None and num_patches is not None:
            with pytest.raises(ValueError):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        if patch_size is None and num_patches is None:
            with pytest.raises(ValueError):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        if patch_size is not None and patch_size <= 0:
            with pytest.raises(ValueError):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        if patch_size is not None and patch_size > seq_length:
            with pytest.warns(UserWarning):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        if num_patches is not None and num_patches <= 0:
            with pytest.raises(ValueError):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        if num_patches is not None and num_patches > seq_length:
            with pytest.warns(UserWarning):
                PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)
            return

        P, N = PatchEncoder.compute_patch_dims(seq_length, patch_size, num_patches)

        assert isinstance(P, int) and isinstance(N, int)
        assert P > 0 and N > 0
        assert P <= seq_length and N <= seq_length
        assert seq_length <= P * N
        # assert P * N <= seq_length + P

        if patch_size is not None:
            assert P == patch_size
            assert N == (seq_length + patch_size - 1) // patch_size
            assert (N - 1) * P < seq_length
            if seq_length % patch_size == 0:
                assert P * N == seq_length

        if num_patches is not None:
            assert N <= num_patches
            assert P == (seq_length + num_patches - 1) // num_patches
            assert (P - 1) * N < seq_length
            if seq_length % num_patches == 0:
                assert P * N == seq_length

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257])
    @pytest.mark.parametrize("in_channels", [3, 5])
    @pytest.mark.parametrize("out_channels", [7, 11])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2, 3])
    @pytest.mark.parametrize("patch_size", [1, 2, 3, 4, 5, 255, 256, 257, 258, None])
    @pytest.mark.parametrize("num_patches", [1, 2, 3, 4, 5, 255, 256, 257, 258, None])
    def test_forward(self, batch_size: int, seq_length: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, patch_size: Optional[int], num_patches: Optional[int]) -> None:
        if patch_size is not None and num_patches is not None:
            with pytest.raises(ValueError):
                PatchEncoder(in_channels, out_channels, num_patches, patch_size, kernel_size=kernel_size, stride=stride)
            return

        if patch_size is None and num_patches is None:
            with pytest.raises(ValueError):
                PatchEncoder(in_channels, out_channels, num_patches, patch_size, kernel_size=kernel_size, stride=stride)
            return

        if patch_size is not None and patch_size < kernel_size:
            with pytest.raises(ValueError):
                PatchEncoder(in_channels, out_channels, num_patches, patch_size, kernel_size=kernel_size, stride=stride)
            return


        model = PatchEncoder(in_channels, out_channels, num_patches, patch_size, kernel_size=kernel_size, stride=stride)
        z = torch.rand(batch_size, seq_length, in_channels)

        if seq_length < model.min_length:
            with pytest.raises(RuntimeError):
                model.forward(z)
            return

        z = model.forward(z)
        num_patches = num_patches if num_patches is not None else math.ceil(seq_length / patch_size)  # type: ignore[operator]
        assert z.shape[0] == batch_size
        assert 0 < z.shape[1] <= num_patches
        assert z.shape[2] == out_channels


class TestFiLM:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("guide_dim", [4, 8, 16])
    @pytest.mark.parametrize("embedding_dim", [4, 8, 16])
    @pytest.mark.parametrize("hidden_size", [8, 16, 32])
    def test_forward(self, batch_size: int, seq_length: int, guide_dim: int, embedding_dim: int, hidden_size: int) -> None:
        net = FiLM(guide_dim, embedding_dim, hidden_size)
        x = torch.rand((batch_size, seq_length, embedding_dim))
        g = torch.rand((batch_size, seq_length, guide_dim))
        z = net.forward(x, g)
        assert z.shape == (batch_size, seq_length, embedding_dim)


class TestMalConvClassifier:

    @pytest.mark.parametrize("cls", [MalConv, MalConvLowMem, MalConvGCG])
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("guide_dim", [0, 3])
    def test_forward(self, cls: type[MalConv | MalConvLowMem | MalConvGCG], batch_size: int, seq_length: int, guide_dim: int, vocab_size: int = 11, channels: int = 7, num_classes: int = 2) -> None:
        embedding = torch.nn.Embedding(vocab_size, 8)
        filmer: FiLM | FiLMNoP
        if guide_dim == 0:
            filmer = FiLMNoP(guide_dim, embedding_dim=8, hidden_size=3)
        else:
            filmer = FiLM(guide_dim, embedding_dim=8, hidden_size=3)
        backbone = cls(embedding_dim=8, channels=channels, kernel_size=3, stride=3)
        head = ClassifificationHead(7, num_classes=2)
        net = MalConvClassifier(embedding, filmer, backbone, head)

        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        if guide_dim == 0:
            g = None
        else:
            g = torch.rand((batch_size, seq_length, guide_dim))

        too_short = x.shape[1] < net.backbone.min_length

        if too_short:
            with pytest.raises(RuntimeError):
                net.forward(x, g)
            return

        z = net.forward(x, g)
        assert z.shape == (batch_size, num_classes)

    @pytest.mark.parametrize("cls", [MalConv, MalConvLowMem, MalConvGCG])
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("guide_dim", [0, 3])
    def test_gradients(self, cls: type[MalConv | MalConvLowMem | MalConvGCG], batch_size: int, seq_length: int, guide_dim: int, vocab_size: int = 11, channels: int = 7, num_classes: int = 2) -> None:
        embedding = torch.nn.Embedding(vocab_size, 8)
        filmer: FiLM | FiLMNoP
        if guide_dim == 0:
            filmer = FiLMNoP(guide_dim, embedding_dim=8, hidden_size=3)
        else:
            filmer = FiLM(guide_dim, embedding_dim=8, hidden_size=3)
        backbone = cls(embedding_dim=8, channels=channels, kernel_size=3, stride=3)
        head = ClassifificationHead(7, num_classes=2)
        net = MalConvClassifier(embedding, filmer, backbone, head)

        net.train()

        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        if guide_dim == 0:
            g = None
        else:
            g = torch.rand((batch_size, seq_length, guide_dim))

        too_short = x.shape[1] < net.backbone.min_length
        if too_short:
            return

        # Case 1. Eval mode without gradient
        net.eval()
        _clear_grads(net)
        with torch.no_grad():
            z1: Tensor = net(x, g)
            loss = z1.sum()
        assert z1.shape == (batch_size, num_classes)
        assert all(param.grad is None for param in net.parameters())

        # Case 2: Eval mode with gradient
        # This is the quirky case. We expect the default malconv to have grads everywhere,
        # because we didn't disable them. However, the low-memory models will have grads severed.
        net.eval()
        _clear_grads(net)
        z2: Tensor = net(x, g)
        loss = z2.sum()
        loss.backward()  # type: ignore[no-untyped-call]
        assert z2.shape == (batch_size, num_classes)
        if cls is MalConv:
            assert all(param.grad is not None for param in net.parameters())
        else:
            assert not all(param.grad is None for param in net.parameters())
            assert any(param.grad is None for param in net.parameters())

        # Case 4: Train mode with gradient
        net.train()
        _clear_grads(net)
        z3: Tensor = net(x, g)
        loss = z3.sum()
        loss.backward()  # type: ignore[no-untyped-call]
        assert z3.shape == (batch_size, num_classes)
        assert all(param.grad is not None for param in net.parameters())

        # Case 4: Train mode without gradient
        # The low-memory models will raise an exception here due to internal checks.
        if cls is MalConv:
            net.train()
            _clear_grads(net)
            with torch.no_grad():
                z4: Tensor = net(x, g)
                loss = z3.sum()
            assert z4.shape == (batch_size, num_classes)
            assert all(param.grad is None for param in net.parameters())
        else:
            z4 = z3.clone()

        assert torch.allclose(z1, z2, atol=1e-7), f"Max diff: {torch.max(torch.abs(z1.flatten() - z2.flatten()))}"
        assert torch.allclose(z1, z3, atol=1e-1), f"Max diff: {torch.max(torch.abs(z1.flatten() - z3.flatten()))}"
        assert torch.allclose(z1, z4, atol=1e-1), f"Max diff: {torch.max(torch.abs(z1.flatten() - z4.flatten()))}"
        assert torch.allclose(z3, z4, atol=1e-7), f"Max diff: {torch.max(torch.abs(z3.flatten() - z4.flatten()))}"


class TestHierarchicalMalConvClassifier:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("add_none", [False, True])
    def test_forward(self, batch_size: int, seq_length: int, num_structures: int, add_none: bool, vocab_size: int = 11, guide_dim: int = 3, channels: int = 7, num_classes: int = 2) -> None:
        embeddings = [torch.nn.Embedding(vocab_size, embedding_dim=8 + i) for i in range(num_structures)]
        filmers = [FiLM(guide_dim, embedding_dim=8 + i, hidden_size=3) for i in range(num_structures)]
        backbones = [MalConv(embedding_dim=8 + i, channels=channels, kernel_size=3, stride=3) for i in range(num_structures)]
        head = ClassifificationHead(channels, num_classes=num_classes)
        net = HierarchicalMalConvClassifier(embeddings, filmers, backbones, head)

        x: list[Optional[Tensor]] = [torch.randint(0, vocab_size, (batch_size, seq_length + i)) for i in range(num_structures)]
        g: list[Optional[Tensor]] = [torch.rand((batch_size, seq_length + i, guide_dim)) for i in range(num_structures)]

        if add_none:
            x[0] = None
            g[0] = None

        if add_none and num_structures == 1:
            with pytest.raises(ValueError):
                net.forward(x, g)
            return

        too_short = False
        for i in range(num_structures):
            if x[i] is not None and x[i].shape[1] < net.min_lengths[i]:  # type: ignore[union-attr]
                too_short = True
                break

        if too_short:
            with pytest.raises(RuntimeError):
                net.forward(x, g)
            return

        z = net.forward(x, g)
        assert z.shape == (batch_size, num_classes)


class TestViT:

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257])
    @pytest.mark.parametrize("embedding_dim", [2, 4])
    @pytest.mark.parametrize("d_model", [16, 32])
    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    def test_forward(self, batch_size: int, seq_length: int, embedding_dim: int, d_model: int, nhead: int, pooling: Literal["cls", "mean"]) -> None:
        model = ViT(embedding_dim, d_model, nhead, num_layers=1, posencoder="fixed", pooling=pooling)
        z = torch.rand(batch_size, seq_length, embedding_dim)
        z = model.forward(z)
        assert z.shape == (batch_size, d_model)


class TestVitClassifier:
    ...


class TestHierarchicalViTClassifier:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("add_none", [False, True])
    def test_forward(self, batch_size: int, seq_length: int, num_structures: int, add_none: bool, vocab_size: int = 11, guide_dim: int = 3, d_model: int = 8, num_classes: int = 2) -> None:
        embeddings = [torch.nn.Embedding(vocab_size, embedding_dim=8 + 2 * i) for i in range(num_structures)]
        filmers = [FiLM(guide_dim, embedding_dim=8 + 2 * i, hidden_size=3) for i in range(num_structures)]
        patchers = [PatchEncoder(in_channels=8 + 2 * i, out_channels=d_model, kernel_size=2, stride=3, patch_size=None, num_patches=2) for i in range(num_structures)]
        patchposencoders = [PatchPositionalityEncoder(d_model, hidden_size=8) for _ in range(num_structures)]
        backbone = ViT(embedding_dim=d_model, d_model=d_model, nhead=1, num_layers=1, posencoder="fixed", pooling="cls")
        head = ClassifificationHead(d_model, num_classes=num_classes)
        net = HierarchicalViTClassifier(embeddings, filmers, patchers, patchposencoders, backbone, head)

        x: list[Optional[Tensor]] = [torch.randint(0, vocab_size, (batch_size, seq_length + i)) for i in range(num_structures)]
        g: list[Optional[Tensor]] = [torch.rand((batch_size, seq_length + i, guide_dim)) for i in range(num_structures)]

        if add_none:
            x[0] = None
            g[0] = None

        if add_none and num_structures == 1:
            with pytest.raises(ValueError):
                net.forward(x, g)
            return

        too_short = False
        for i in range(num_structures):
            if x[i] is not None and x[i].shape[1] < net.min_lengths[i]:  # type: ignore[union-attr]
                too_short = True
                break

        if too_short:
            with pytest.raises(RuntimeError):
                net.forward(x, g)
            return

        z = net.forward(x, g)
        assert z.shape == (batch_size, num_classes)


# ============================================================================
# Tests for the low-memory streaming functions.
# ============================================================================


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

    def test_patch_active_masks_patches_when_available(self) -> None:
        """
        Forward-compatible: after patch_active=(B,N) is added, verify inactive patches are zero.
        """
        sig = inspect.signature(_lowmem_patchwise_max_over_time_streaming)
        if "patch_active" not in sig.parameters:
            pytest.skip("patch_active not implemented yet")

        torch.manual_seed(5)

        B, T, E = 2, 256, 4
        C = 5
        rf = stride = 8
        N = 8

        z = torch.rand(B, T, E, dtype=torch.float32)
        conv = _make_positive_conv(E, C, rf, stride)

        def activations_fn(x: torch.Tensor) -> torch.Tensor:
            return conv(x)  # type: ignore[no-any-return]

        full_vals, _ = _lowmem_patchwise_max_over_time_streaming(  # type: ignore[call-arg]
            preprocess=_identity_preprocess,
            ts=[z],
            rf=rf,
            first_stride=stride,
            chunk_size=128,
            overlap=64,
            channels=C,
            num_patches=N,
            activations_fn=activations_fn,
            patch_active=None,
        )

        patch_active = torch.zeros((B, N), dtype=torch.bool)
        patch_active[:, ::2] = True

        masked_vals, _ = _lowmem_patchwise_max_over_time_streaming(  # type: ignore[call-arg]
            preprocess=_identity_preprocess,
            ts=[z],
            rf=rf,
            first_stride=stride,
            chunk_size=128,
            overlap=64,
            channels=C,
            num_patches=N,
            activations_fn=activations_fn,
            patch_active=patch_active,
        )

        assert torch.all(masked_vals[:, 1::2, :] == 0)
        torch.testing.assert_close(masked_vals[:, ::2, :], full_vals[:, ::2, :], rtol=0, atol=0)


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
        Forward-compatible: after mask=(B,K) is added and scatter handles inv=-1, verify inactive
        entries are zeros.
        """
        sig = inspect.signature(_gather_wins_via_preprocess_batched)
        if "mask" not in sig.parameters:
            pytest.skip("mask not implemented yet")

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

        wins_cat, meta = _gather_wins_via_preprocess_batched(  # type: ignore[call-arg]
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

        torch.testing.assert_close(out, out_manual, rtol=0, atol=0)
