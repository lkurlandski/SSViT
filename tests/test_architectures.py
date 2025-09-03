"""
Tests.
"""

import math
from typing import Literal
from typing import Optional

import pytest
import torch

from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import SinusoidalPositionalEncoding
from src.architectures import PatchEncoder
from src.architectures import ViT
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier


# TODO: clean this up.
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

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("patch_size", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("in_channels", [3, 5, 7])
    def test_split_patches(self, batch_size: int, seq_length: int, patch_size: int, in_channels: int) -> None:
        z = torch.rand(batch_size, seq_length, in_channels)
        patches = PatchEncoder.split_patches(z, patch_size)
        assert patches.shape[0] == batch_size
        assert patches.shape[1] == math.ceil(seq_length / patch_size)
        assert patches.shape[2] == min(patch_size, seq_length)
        assert patches.shape[3] == in_channels

    @pytest.mark.parametrize("seq_length", list(range(258)))
    @pytest.mark.parametrize("patch_size", [0, 1, 2, 3, 4, 5, 11, 17, 53, 255, 256, 257, 258, None])
    @pytest.mark.parametrize("num_patches", [0, 1, 2, 3, 4, 5, 11, 17, 53, 255, 256, 257, 258, None])
    def test_patch_dims(self, seq_length: int, patch_size: Optional[int], num_patches: Optional[int]) -> None:

        if seq_length <= 0:
            with pytest.raises(ValueError):
                PatchEncoder.patch_dims(seq_length, patch_size, num_patches)
            return
        
        if patch_size is not None and num_patches is not None:
            with pytest.raises(ValueError):
                PatchEncoder.patch_dims(seq_length, patch_size, num_patches)
            return

        if patch_size is None and num_patches is None:
            with pytest.raises(ValueError):
                PatchEncoder.patch_dims(seq_length, patch_size, num_patches)
            return

        if patch_size is not None and (patch_size <= 0 or patch_size > seq_length):
            with pytest.raises(ValueError):
                PatchEncoder.patch_dims(seq_length, patch_size, num_patches)
            return

        if num_patches is not None and (num_patches <= 0 or num_patches > seq_length):
            with pytest.raises(ValueError):
                PatchEncoder.patch_dims(seq_length, patch_size, num_patches)
            return

        P, N = PatchEncoder.patch_dims(seq_length, patch_size, num_patches)

        assert isinstance(P, int) and isinstance(N, int)
        assert P > 0 and N > 0
        assert P <= seq_length and N <= seq_length
        assert seq_length <= P * N <= seq_length + P

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
                PatchEncoder(in_channels, out_channels, kernel_size, stride, patch_size=patch_size, num_patches=num_patches)
            return

        if patch_size is None and num_patches is None:
            with pytest.raises(ValueError):
                PatchEncoder(in_channels, out_channels, kernel_size, stride, patch_size=patch_size, num_patches=num_patches)
            return

        if patch_size is not None and patch_size < kernel_size:
            with pytest.raises(ValueError):
                PatchEncoder(in_channels, out_channels, kernel_size, stride, patch_size=patch_size, num_patches=num_patches)
            return


        model = PatchEncoder(in_channels, out_channels, kernel_size, stride, patch_size=patch_size, num_patches=num_patches)
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


class TestViT:

    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257])
    @pytest.mark.parametrize("embedding_dim", [2, 4])
    @pytest.mark.parametrize("d_model", [16, 32])
    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    def test_forward(self, batch_size: int, seq_length: int, embedding_dim: int, d_model: int, nhead: int, pooling: Literal["cls", "mean"]) -> None:
        model = ViT(embedding_dim, d_model, nhead, num_layers=1, pooling=pooling)
        z = torch.rand(batch_size, seq_length, embedding_dim)
        z = model.forward(z)
        assert z.shape == (batch_size, d_model)


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


class TestMalConvs:

    @pytest.mark.parametrize("arch", ["MalConv", "MalConvLowMem", "MalConvGCG"])
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("embedding_dim", [4, 8, 16])
    @pytest.mark.parametrize("channels", [8, 16, 32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("chunk_size", [None, 11, 17])
    @pytest.mark.parametrize("overlap", [None, 3, 5])
    def test_forward(
        self,
        arch: Literal["MalConv", "MalConvLowMem", "MalConvGCG"],
        batch_size: int,
        seq_length: int,
        embedding_dim: int,
        channels: int,
        kernel_size: int,
        stride: int,
        chunk_size: Optional[int],
        overlap: Optional[int],
    ) -> None:
        if arch == "MalConv":
            if chunk_size is not None or overlap is not None:
                return
            net = MalConv(embedding_dim, channels, kernel_size, stride)
        elif arch == "MalConvLowMem":
            if chunk_size is None:
                return
            net = MalConvLowMem(embedding_dim, channels, kernel_size, stride, chunk_size=chunk_size, overlap=overlap)
        elif arch == "MalConvGCG":
            if chunk_size is None:
                return
            net = MalConvGCG(embedding_dim, channels, kernel_size, stride, chunk_size=chunk_size, overlap=overlap)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        x = torch.rand((batch_size, seq_length, embedding_dim))
        if seq_length < net.min_length:
            with pytest.raises(RuntimeError):
                net.forward(x)
            return
        z = net.forward(x)
        assert z.shape == (batch_size, channels)


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

        x = [torch.randint(0, vocab_size, (batch_size, seq_length + i)) for i in range(num_structures)]
        g = [torch.rand((batch_size, seq_length + i, guide_dim)) for i in range(num_structures)]

        if add_none:
            x[0] = None
            g[0] = None

        if add_none and num_structures == 1:
            with pytest.raises(ValueError):
                net.forward(x, g)
            return

        too_short = False
        for i in range(num_structures):
            if x[i] is not None and x[i].shape[1] < net.min_lengths[i]:
                too_short = True
                break

        if too_short:
            with pytest.raises(RuntimeError):
                net.forward(x, g)
            return

        z = net.forward(x, g)
        assert z.shape == (batch_size, num_classes)


class TestHierarchicalViTClassifier:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("add_none", [False, True])
    def test_forward(self, batch_size: int, seq_length: int, num_structures: int, add_none: bool, vocab_size: int = 11, guide_dim: int = 3, d_model: int = 8, num_classes: int = 2) -> None:
        embeddings = [torch.nn.Embedding(vocab_size, embedding_dim=8 + 2 * i) for i in range(num_structures)]
        filmers = [FiLM(guide_dim, embedding_dim=8 + 2 * i, hidden_size=3) for i in range(num_structures)]
        patchers = [PatchEncoder(in_channels=8 + 2 * i, out_channels=d_model, kernel_size=2, stride=3, patch_size=3, num_patches=None) for i in range(num_structures)]
        backbone = ViT(embedding_dim=d_model, d_model=d_model, nhead=1, num_layers=1, pooling="cls")
        head = ClassifificationHead(d_model, num_classes=num_classes)
        net = HierarchicalViTClassifier(embeddings, filmers, patchers, backbone, head)

        x = [torch.randint(0, vocab_size, (batch_size, seq_length + i)) for i in range(num_structures)]
        g = [torch.rand((batch_size, seq_length + i, guide_dim)) for i in range(num_structures)]

        if add_none:
            x[0] = None
            g[0] = None

        if add_none and num_structures == 1:
            with pytest.raises(ValueError):
                net.forward(x, g)
            return

        too_short = False
        for i in range(num_structures):
            if x[i] is not None and x[i].shape[1] < net.min_lengths[i]:
                too_short = True
                break

        if too_short:
            with pytest.raises(RuntimeError):
                net.forward(x, g)
            return

        z = net.forward(x, g)
        assert z.shape == (batch_size, num_classes)
