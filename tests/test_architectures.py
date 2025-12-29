"""
Tests.
"""

from functools import partial
from itertools import product
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
from src.architectures import _lowmem_patchwise_max_over_time_streaming
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
        patchers = [PatchEncoder(in_channels=8 + 2 * i, out_channels=d_model, kernel_size=2, stride=3, patch_size=3, num_patches=None) for i in range(num_structures)]
        patchposencoders = [PatchPositionalityEncoder(d_model, hidden_size=8) for _ in range(num_structures)]
        backbone = ViT(embedding_dim=d_model, d_model=d_model, nhead=1, num_layers=1, pooling="cls")
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
