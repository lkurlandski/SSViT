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

from src.architectures import Identity
from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import SinusoidalPositionalEncoding
from src.architectures import PatchEncoderBase
from src.architectures import PatchEncoder
from src.architectures import PatchEncoderLowMem
from src.architectures import PatchEncoderLowMemSwitchMoE
from src.architectures import ViT
from src.architectures import MalConvBase
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import MalConvClassifier
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import StructuralMalConvClassifier
from src.architectures import ViTClassifier
from src.architectures import HierarchicalViTClassifier
from src.architectures import StructuralViTClassifier
from src.architectures import PatchPositionalityEncoder
from src.architectures import DWBlock
from src.architectures import AdaptiveAtnPooling1d
from src.architectures import DWCSequenceEncoder
from src.architectures import _reconcile_per_structure_patches_static
from src.architectures import _reconcile_per_structure_patches_dynamic
from src.data import _unpack_order_to_tensors
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


class TestPatchEncoderLowMem:

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("in_channels", [3, 5])
    @pytest.mark.parametrize("out_channels", [7, 11])
    @pytest.mark.parametrize("num_patches", [None, 1, 3])
    @pytest.mark.parametrize("patch_size", [None, 1, 3])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("chunk_size", [16, 32, 64])
    @pytest.mark.parametrize("overlap", [None, 0, 1])
    @pytest.mark.parametrize("fp32", [True, False])
    def test_forward(
        self,
        batch_size: int,
        seq_length: int,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        kernel_size: int,
        stride: int,
        chunk_size: int,
        overlap: Optional[int],
        fp32: bool,
    ) -> None:

        def init() -> PatchEncoderLowMem:
            return PatchEncoderLowMem(
                in_channels,
                out_channels,
                num_patches,
                patch_size,
                kernel_size=kernel_size,
                stride=stride,
                chunk_size=chunk_size,
                overlap=overlap,
                fp32=fp32,
            )

        if num_patches is None:
            with pytest.raises(ValueError):
                net = init()
            return

        if patch_size is not None:
            with pytest.raises(ValueError):
                net = init()
            return

        net = init()

        z = torch.rand(batch_size, seq_length, in_channels)

        if seq_length < net.min_length:
            with pytest.raises(RuntimeError):
                net.forward(z)
            return

        z = net.forward(z)

        assert z.shape[0] == batch_size
        assert z.shape[1] == num_patches
        assert z.shape[2] == out_channels


class TestPatchEncoderLowMemSwitchMoE:

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("num_patches", [1, 3])
    @pytest.mark.parametrize("num_experts", [1, 2, 3])
    @pytest.mark.parametrize("probe_dim", [4, 8])
    @pytest.mark.parametrize("router_hidden", [8, 16])
    @pytest.mark.parametrize("router_temperature", [1e-6, 1.0])
    @pytest.mark.parametrize("router_noise_std", [0.0, 0.1])
    @pytest.mark.parametrize("router_top_k", [1, 2, 3])
    def test_forward(
        self,
        batch_size: int,
        seq_length: int,
        num_patches: int,
        num_experts: int,
        probe_dim: int,
        router_hidden: int,
        router_temperature: float,
        router_noise_std: float,
        router_top_k: int,
    ) -> None:

        in_channels = 8
        out_channels = 16
        num_patches = 4
        patch_size = None
        kernel_size = 3
        stride = 2
        chunk_size = 32
        overlap = None
        fp32 = True

        def init() -> PatchEncoderLowMemSwitchMoE:
            return PatchEncoderLowMemSwitchMoE(
                in_channels,
                out_channels,
                num_patches,
                patch_size,
                kernel_size=kernel_size,
                stride=stride,
                chunk_size=chunk_size,
                overlap=overlap,
                fp32=fp32,
                num_experts=num_experts,
                probe_dim=probe_dim,
                router_hidden=router_hidden,
                router_temperature=router_temperature,
                router_noise_std=router_noise_std,
                router_top_k=router_top_k,
            )

        if router_top_k > num_experts:
            with pytest.raises(ValueError):
                init()
            return

        net = init()

        z = torch.rand(batch_size, seq_length, in_channels)

        if seq_length < net.min_length:
            with pytest.raises(RuntimeError):
                net.forward(z)
            return

        z = net.forward(z)

        assert z.shape[0] == batch_size
        assert z.shape[1] == num_patches
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


class TestViTClassifier:

    @pytest.mark.parametrize("num_experts", [1, 2, 3, 5])
    def test_patcher_stats(self, num_experts: int) -> None:
        vocab_size = 11
        embedding_dim = 8
        patcher_channels = 16
        d_model = 16
        embedding = Embedding(vocab_size, embedding_dim)
        filmer = FiLMNoP(0, embedding_dim, 0)
        norm = nn.LayerNorm(patcher_channels)
        patchposencoder = Identity()
        backbone = ViT(patcher_channels, d_model, posencoder="fixed")
        head = ClassifificationHead(d_model, num_classes=2)

        x = torch.randint(0, vocab_size, (2, 1024))
        g = None

        patcher = PatchEncoder(embedding_dim, patcher_channels, 4, None)
        model = ViTClassifier(embedding, filmer, patcher, norm, patchposencoder, backbone, head)
        assert model.last_aux_loss is None
        assert model.last_entropy is None
        assert model.last_usage is None

        model(x, g)
        assert model.last_aux_loss is None
        assert model.last_entropy is None
        assert model.last_usage is None

        patcher = PatchEncoderLowMemSwitchMoE(embedding_dim, patcher_channels, 4, None, num_experts=num_experts)
        model = ViTClassifier(embedding, filmer, patcher, norm, patchposencoder, backbone, head)
        assert model.last_aux_loss is not None and model.last_aux_loss.shape == () and (model.last_aux_loss == 0).all()
        assert model.last_entropy is not None and model.last_entropy.shape == () and (model.last_entropy == 0).all()
        assert model.last_usage is not None and model.last_usage.shape == (num_experts,) and (model.last_usage == 0).all()

        model(x, g)
        assert model.last_aux_loss is not None and model.last_aux_loss.shape == () and (model.last_aux_loss >= 0).all()
        assert model.last_entropy is not None and model.last_entropy.shape == () and (model.last_entropy >= 0).all()
        assert model.last_usage is not None and model.last_usage.shape == (num_experts,) and (model.last_usage >= 0).all()


class TestHierarchicalViTClassifier:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("add_none", [False, True])
    def test_forward(self, batch_size: int, seq_length: int, num_structures: int, add_none: bool, vocab_size: int = 11, guide_dim: int = 3, d_model: int = 8, num_classes: int = 2) -> None:
        embeddings = [torch.nn.Embedding(vocab_size, embedding_dim=8 + 2 * i) for i in range(num_structures)]
        filmers = [FiLM(guide_dim, embedding_dim=8 + 2 * i, hidden_size=3) for i in range(num_structures)]
        patchers = [PatchEncoder(in_channels=8 + 2 * i, out_channels=d_model, kernel_size=2, stride=3, patch_size=None, num_patches=2) for i in range(num_structures)]
        norms = [nn.LayerNorm(p.out_channels) for p in patchers]
        patchposencoders = [PatchPositionalityEncoder(d_model, hidden_size=8) for _ in range(num_structures)]
        backbone = ViT(embedding_dim=d_model, d_model=d_model, nhead=1, num_layers=1, posencoder="fixed", pooling="cls")
        head = ClassifificationHead(d_model, num_classes=num_classes)
        net = HierarchicalViTClassifier(embeddings, filmers, patchers, norms, patchposencoders, backbone, head)

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

    @pytest.mark.skip(reason="Not implemented yet")
    def test_patcher_stats(self, num_experts: int) -> None:
        ...


class TestStructuralViTClassifier:

    @pytest.mark.skip(reason="Not implemented yet")
    def test_forward(self) -> None:
        ...

    @pytest.mark.parametrize("num_patches", [1, 4])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 5])
    @pytest.mark.parametrize("num_experts", [1, 2, 3, 5])
    def test_patcher_stats(self, num_patches: int, num_structures: int, num_experts: int) -> None:
        batch_size = 2
        vocab_size = 11
        embedding_dim = 8
        patcher_channels = 16
        d_model = 16
        embeddings = [Embedding(vocab_size, embedding_dim) for _ in range(num_structures)]
        filmers = [FiLMNoP(0, embedding_dim, 0) for _ in range(num_structures)]
        norms = [nn.LayerNorm(patcher_channels) for _ in range(num_structures)]
        patchposencoder = [Identity() for _ in range(num_structures)]
        backbone = ViT(patcher_channels, d_model, posencoder="fixed")
        head = ClassifificationHead(d_model, num_classes=2)

        x = [torch.randint(0, vocab_size, (batch_size, 1024)) for _ in range(num_structures)]
        g = [None for _ in range(num_structures)]
        order = [[(structure_index, local_index) for structure_index in range(num_structures)] for local_index in range(batch_size)]
        order_struct, order_local = _unpack_order_to_tensors(order)
        struct_sizes = torch.tensor([batch_size for _ in range(num_structures)])

        patchers = [PatchEncoder(embedding_dim, patcher_channels, num_patches, None) for _ in range(num_structures)]
        model = StructuralViTClassifier(embeddings, filmers, patchers, norms, patchposencoder, backbone, head)
        assert model.last_aux_loss is None
        assert model.last_entropy is None
        assert model.last_usage is None

        model(x, g, order, order_struct, order_local, struct_sizes)
        assert model.last_aux_loss is None
        assert model.last_entropy is None
        assert model.last_usage is None

        patchers = [PatchEncoderLowMemSwitchMoE(embedding_dim, patcher_channels, num_patches, None, num_experts=num_experts) for _ in range(num_structures)]
        model = StructuralViTClassifier(embeddings, filmers, patchers, norms, patchposencoder, backbone, head)

        assert model.last_aux_loss is not None and model.last_aux_loss.shape == (num_structures,) and (model.last_aux_loss == 0).all()
        assert model.last_entropy is not None and model.last_entropy.shape == (num_structures,) and (model.last_entropy == 0).all()
        assert model.last_usage is not None and model.last_usage.shape == (num_structures, num_experts) and (model.last_usage == 0).all()

        model(x, g, order, order_struct, order_local, struct_sizes)
        assert model.last_aux_loss is not None and model.last_aux_loss.shape == (num_structures,) and (model.last_aux_loss >= 0).all()
        assert model.last_entropy is not None and model.last_entropy.shape == (num_structures,) and (model.last_entropy >= 0).all()
        assert model.last_usage is not None and model.last_usage.shape == (num_structures, num_experts) and (model.last_usage >= 0).all()


class TestDWBlock:

    def test_forward_shape_preserves_length(self) -> None:
        torch.manual_seed(0)
        m = DWBlock(dim=4, k=7)
        x = torch.randn(2, 4, 17)
        y = m(x)
        assert y.shape == x.shape

    def test_forward_shortest(self) -> None:
        m = DWBlock(dim=4, k=7)
        x = torch.randn(1, 4, m.min_length)
        y = m(x)
        assert y.shape == x.shape


class TestAdaptiveAtnPooling1d:

    def test_min_length(self) -> None:
        m = AdaptiveAtnPooling1d(dim=4)
        assert m.min_length == 1


    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        m = AdaptiveAtnPooling1d(dim=4)
        x = torch.randn(3, 4, 11)
        y = m(x)
        assert y.shape == (3, 4)


    def test_masked_uniform_matches_mean(self) -> None:
        # Force logits == 0 everywhere => uniform attention over unmasked positions
        m = AdaptiveAtnPooling1d(dim=2)
        with torch.no_grad():
            m.w.weight.zero_()
            m.w.bias.zero_()

        # (B=1, C=2, T=4)
        x = torch.tensor([[
            [1.0, 2.0, 100.0, 200.0],
            [3.0, 4.0, 300.0, 400.0],
        ]])  # (1,2,4)

        key_padding_mask = torch.tensor([[False, False, True, True]])
        y = m(x, key_padding_mask=key_padding_mask)

        expected = torch.tensor([[ (1.0 + 2.0)/2.0, (3.0 + 4.0)/2.0 ]])
        assert torch.allclose(y, expected, atol=1e-6)


    def test_all_masked_returns_zero(self) -> None:
        m = AdaptiveAtnPooling1d(dim=2)
        with torch.no_grad():
            m.w.weight.zero_()
            m.w.bias.zero_()

        x = torch.randn(1, 2, 5)
        key_padding_mask = torch.ones(1, 5, dtype=torch.bool)
        y = m(x, key_padding_mask=key_padding_mask)

        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


class TestDWCSequenceEncoder:

    def test_min_length_default(self) -> None:
        enc = DWCSequenceEncoder(in_channels=8, out_channels=16, depth=2, pooling="avg")
        assert enc.min_length == 64

    def test_min_length_padding_zero(self) -> None:
        enc = DWCSequenceEncoder(in_channels=8, out_channels=16, depth=2, kernel_size=7, pooling="avg")
        assert enc.min_length == 7

    def test_forward_shortest_ok(self) -> None:
        enc = DWCSequenceEncoder(in_channels=8, out_channels=16, depth=2, kernel_size=7, pooling="max")
        x = torch.randn(2, 8, enc.min_length)
        y = enc(x)
        assert y.shape == (2, 16)

    def test_forward_too_short_raises(self) -> None:
        enc = DWCSequenceEncoder(in_channels=8, out_channels=16, depth=2, kernel_size=7, pooling="max")
        x = torch.randn(1, 8, enc.min_length - 1)
        with pytest.raises(RuntimeError):
            _ = enc(x)

    def test_forward_atn_with_mask(self) -> None:
        torch.manual_seed(0)
        enc = DWCSequenceEncoder(
            in_channels=8,
            out_channels=16,
            depth=3,
            pooling="atn",
            stride=8,
            kernel_size=7,
        )
        x = torch.randn(4, 8, 128)
        key_padding_mask = torch.zeros(4, 128, dtype=torch.bool)
        key_padding_mask[:, 96:] = True

        y = enc(x, key_padding_mask=key_padding_mask)
        assert y.shape == (4, 16)

    def test_checkpointing_backward(self) -> None:
        torch.manual_seed(0)
        enc = DWCSequenceEncoder(
            in_channels=8,
            out_channels=16,
            depth=4,
            pooling="atn",
            checkpoint_segments=2,
        )
        enc.train()

        x = torch.randn(2, 8, 256, requires_grad=True)
        key_padding_mask = torch.zeros(2, 256, dtype=torch.bool)

        y = enc(x, key_padding_mask=key_padding_mask)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestReconcilePerStructurePatches:

    @staticmethod
    def _create_inputs() -> tuple[list[Tensor], list[list[tuple[int, int]]]]:
        zs = [
            torch.rand((2, 1, 8)),  # (B_0, 1, C)
            torch.rand((3, 1, 8)),  # (B_1, 1, C)
            torch.rand((1, 1, 8)),  # (B_2, 1, C)
        ]

        order = [
            [(2, 0), (0, 1)],
            [(0, 0), (1, 1), (1, 0), (1, 2)],
        ]

        return zs, order

    @staticmethod
    def _check_outputs(zs: list[Tensor], z: Tensor, k: Tensor, *, length: int = 4) -> None:
        assert z.shape == (2, length, 8)
        assert k.shape == (2, length)

        assert z.dtype == torch.float
        assert k.dtype == torch.bool

        assert torch.allclose(z[0, 0, :], zs[2][0, 0, :])
        assert torch.allclose(z[0, 1, :], zs[0][1, 0, :])
        assert torch.allclose(z[1, 0, :], zs[0][0, 0, :])
        assert torch.allclose(z[1, 1, :], zs[1][1, 0, :])
        assert torch.allclose(z[1, 2, :], zs[1][0, 0, :])

        assert not k[0, 0].item()
        assert not k[0, 1].item()
        assert k[0, 2].item()
        assert k[0, 3].item()
        assert not k[1, 0].item()
        assert not k[1, 1].item()
        assert not k[1, 2].item()
        assert not k[1, 3].item()

    def test_dynamic(self) -> None:
        zs, order = self._create_inputs()
        z, k = _reconcile_per_structure_patches_dynamic(zs, order)
        self._check_outputs(zs, z, k)

    def test_static(self) -> None:
        zs, order = self._create_inputs()
        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)

        z, k = _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)
        self._check_outputs(zs, z, k, length=4)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_static_multitoken(self) -> None:
        zs = [
            torch.arange(2*1*4, dtype=torch.float32).view(2, 1, 4),   # (B0=2, L0=1, C=4)
            torch.arange(3*3*4, dtype=torch.float32).view(3, 3, 4),   # (B1=3, L1=3, C=4)  <-- multi-token
        ]
        order = [
            [(1, 2), (0, 1)],  # sample0: struct1(local2, L=3 tokens), struct0(local1, L=1) => len 4
            [(0, 0)],          # sample1: struct0(local0, L=1) => len 1
        ]
        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)

        z, k = _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)

        # sample 0 should start with zs[1][2, :, :] then zs[0][1, :, :]
        assert torch.allclose(z[0, 0:3], zs[1][2])      # 3 tokens
        assert torch.allclose(z[0, 3:4], zs[0][1])      # 1 token
        assert k[0, :4].eq(False).all()

    def test_gradients_match_dynamic(self) -> None:
        torch.manual_seed(0)
        zs, order = self._create_inputs()
        zs_dyn = [z.clone().requires_grad_(True) for z in zs]
        zs_sta = [z.clone().requires_grad_(True) for z in zs]

        z_d, _ = _reconcile_per_structure_patches_dynamic(zs_dyn, order)
        loss_d = (z_d ** 2).sum()
        loss_d.backward()  # type: ignore[no-untyped-call]

        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)
        z_s, _ = _reconcile_per_structure_patches_static(zs_sta, order_struct, order_local, struct_sizes)
        loss_s = (z_s ** 2).sum()
        loss_s.backward()  # type: ignore[no-untyped-call]

        for a, b in zip(zs_dyn, zs_sta):
            assert a.grad is not None and b.grad is not None
            assert torch.allclose(a.grad, b.grad)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_static_multitoken_structure(self) -> None:
        # Make one structure emit multiple tokens per item (L_s > 1).
        torch.manual_seed(0)
        zs = [
            torch.rand((2, 1, 4)),  # struct 0: L0=1
            torch.rand((3, 3, 4)),  # struct 1: L1=3  <-- multi-token
            torch.rand((1, 2, 4)),  # struct 2: L2=2
        ]
        order = [
            [(1, 2), (0, 1)],  # sample0: struct1(local2, 3 toks) then struct0(local1, 1 tok) => len 4
            [(2, 0)],          # sample1: struct2(local0, 2 toks) => len 2
        ]
        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)

        # Pack-to-batch-max => M = max(4, 2) = 4
        z, k = _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)

        assert z.shape == (2, 4, 4)
        assert k.shape == (2, 4)

        # sample0: first 3 tokens from zs[1][2], then 1 token from zs[0][1]
        assert torch.allclose(z[0, 0:3], zs[1][2, 0:3])
        assert torch.allclose(z[0, 3:4], zs[0][1, 0:1])
        assert k[0, :4].eq(False).all()

        # sample1: first 2 tokens from zs[2][0], last 2 padded
        assert torch.allclose(z[1, 0:2], zs[2][0, 0:2])
        assert k[1, 0].item() is False
        assert k[1, 1].item() is False
        assert k[1, 2].item() is True
        assert k[1, 3].item() is True

    def test_empty_sample_behavior(self) -> None:
        # One sample has no structures.
        zs = [
            torch.rand((2, 1, 8)),
            torch.rand((2, 1, 8)),
        ]
        order = [
            [],                 # empty
            [(0, 1), (1, 0)],    # non-empty
        ]

        # Dynamic currently raises for empty sample; keep that expectation.
        with pytest.raises(RuntimeError):
            _reconcile_per_structure_patches_dynamic(zs, order)

        # Static: depending on your chosen semantics, it might:
        # - raise (matching dynamic), OR
        # - return an all-padding row for that sample.
        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)
        try:
            z, k = _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)
        except RuntimeError:
            return  # acceptable if you decided to match dynamic semantics

        # If it didn't raise, assert sample0 is entirely padding.
        assert z.shape == (2, 4, 8)
        assert k.shape == (2, 4)
        assert k[0].all().item() is True  # all padding for empty sample

    def test_cpu_cuda_parity(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(0)
        zs, order = self._create_inputs()
        order_struct, order_local = _unpack_order_to_tensors(order, num=4, policy="raise")
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long)

        # CPU reference
        z_cpu, k_cpu = _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)

        # CUDA run
        zs_cuda = [z.cuda() for z in zs]
        order_struct_cuda = order_struct.cuda()
        order_local_cuda = order_local.cuda()
        struct_sizes = struct_sizes.cuda()

        z_gpu, k_gpu = _reconcile_per_structure_patches_static(zs_cuda, order_struct_cuda, order_local_cuda, struct_sizes)

        assert torch.allclose(z_cpu, z_gpu.cpu())
        assert torch.equal(k_cpu, k_gpu.cpu())
