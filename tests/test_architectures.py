"""
Tests.
"""

import math

import pytest
import torch

from src.utils import TensorError
from src.architectures import ClassifificationHead
from src.architectures import MultiChannelDiscreteEmbedding
from src.architectures import MultiChannelDiscreteSequenceVisionTransformer
from src.architectures import MultiChannelMalConv
from src.architectures import SequenceEmbeddingEncoder


class TestClassificationHead:
    B = 4

    @pytest.mark.parametrize("input_size", [8, 16, 32])
    @pytest.mark.parametrize("num_classes", [2, 4, 8])
    @pytest.mark.parametrize("hidden_size", [-1, 8, 16])
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_forward(self, input_size: int, num_classes: int, hidden_size: int, num_layers: int):
        if num_layers > 1 and hidden_size <= 0:
            with pytest.raises(ValueError):
                ClassifificationHead(input_size, num_classes, hidden_size, num_layers)
            return

        model = ClassifificationHead(input_size, num_classes, hidden_size, num_layers)
        z = torch.randn(self.B, input_size)
        z = model.forward(z)
        assert z.shape == (self.B, num_classes)


class TestMultiChannelDiscreteEmbedding:
    B = 4
    T = 256

    @pytest.mark.parametrize("num_embedding", [[8], [16, 32], [8, 16, 32]])
    @pytest.mark.parametrize("embedding_dim", [[4], [8, 16], [4, 8, 16]])
    def test_forward(self, num_embedding: list[int], embedding_dim: list[int]):
        if len(num_embedding) != len(embedding_dim):
            with pytest.raises(ValueError):
                MultiChannelDiscreteEmbedding(num_embedding, embedding_dim)
            return

        model = MultiChannelDiscreteEmbedding(num_embedding, embedding_dim)
        x = [torch.randint(0, v, (self.B, self.T), dtype=torch.long) for v in num_embedding]
        z = model.forward(*x)
        assert z.shape == (self.B, self.T, sum(embedding_dim))

        if len(num_embedding) > 1:
            x = [torch.randint(0, v, (self.B, self.T - i), dtype=torch.long) for i, v in enumerate(num_embedding)]
            with pytest.raises(TensorError):
                model.forward(*x)


class TestSequenceEmbeddingEncoder:
    B = 4

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("patch_size", [1, 2, 3, 11, 17, 53])
    @pytest.mark.parametrize("in_channels", [3, 5, 7])
    def test_split_patches(self, batch_size: int, seq_length: int, patch_size: int, in_channels: int):
        z = torch.rand(batch_size, seq_length, in_channels)
        patches = SequenceEmbeddingEncoder._split_patches(z, patch_size)
        assert patches.shape[0] == batch_size
        assert patches.shape[1] == math.ceil(seq_length / patch_size)
        assert patches.shape[2] == min(patch_size, seq_length)
        assert patches.shape[3] == in_channels

    def _test_forward(
        self, seq_length: int, patch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        model = SequenceEmbeddingEncoder(patch_size, in_channels, out_channels, kernel_size, stride)
        z = torch.rand(self.B, seq_length, in_channels)
        z = model.forward(z)
        assert z.shape[0] == self.B
        assert z.shape[1] == math.ceil(seq_length / patch_size)
        assert z.shape[2] == out_channels

    @pytest.mark.parametrize("seq_length", [11, 17, 53])
    @pytest.mark.parametrize("patch_size", [11, 17, 53])
    @pytest.mark.parametrize("in_channels", [3, 5, 7])
    @pytest.mark.parametrize("out_channels", [8, 16, 32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_forward(
        self, seq_length: int, patch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        self._test_forward(seq_length, patch_size, in_channels, out_channels, kernel_size, stride)

    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 63])
    def test_forward_seq_length_too_short(self, seq_length: int):
        # Any sequence length less than the kernel size should raise an error.
        with pytest.raises(RuntimeError):
            self._test_forward(seq_length, patch_size=64, in_channels=16, out_channels=12, kernel_size=64, stride=4)

    @pytest.mark.parametrize("patch_size", [1, 2, 3, 11, 17, 63])
    def test_forward_patch_size_too_small(self, patch_size: int):
        # Any patch size less than the kernel size should raise an error.
        with pytest.raises(ValueError):
            self._test_forward(
                seq_length=1024, patch_size=patch_size, in_channels=16, out_channels=12, kernel_size=64, stride=4
            )


class TestMultiChannelDiscreteSequenceVisionTransformer:
    B = 2

    def _test_forward(
        self,
        seq_length: int,
        num_embeddings: list[int],
        embedding_dim: list[int],
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_classes: int,
    ):
        model = MultiChannelDiscreteSequenceVisionTransformer(
            num_embeddings, embedding_dim, patch_size, d_model, nhead, num_layers, num_classes
        )
        x = [torch.randint(0, v, (self.B, seq_length), dtype=torch.long) for v in num_embeddings]
        y = model.forward(*x)
        assert y.shape == (self.B, num_classes)

    @pytest.mark.parametrize("seq_length", [359, 512])
    @pytest.mark.parametrize("num_embeddings", [[8], [16, 32], [8, 16, 32]])
    @pytest.mark.parametrize("embedding_dim", [[4], [8, 16], [4, 8, 16]])
    @pytest.mark.parametrize("patch_size", [255, 256, 257])
    @pytest.mark.parametrize("d_model", [16, 32])
    @pytest.mark.parametrize("nhead", [1, 2])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("num_classes", [2, 4, 8])
    def test_forward(
        self,
        seq_length: int,
        num_embeddings: list[int],
        embedding_dim: list[int],
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_classes: int,
    ):
        if len(num_embeddings) != len(embedding_dim):
            return
        self._test_forward(
            seq_length, num_embeddings, embedding_dim, patch_size, d_model, nhead, num_layers, num_classes
        )

    def test_forward_patch_size_too_small(self):
        with pytest.raises(ValueError):
            self._test_forward(
                seq_length=1024,
                num_embeddings=[8],
                embedding_dim=[8],
                patch_size=63,
                d_model=64,
                nhead=1,
                num_layers=1,
                num_classes=1,
            )


class TestMultiChannelMalConv:
    B = 2

    def _test_forward(
        self,
        seq_length: int,
        num_embeddings: list[int],
        embedding_dim: list[int],
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_classes: int,
    ):
        model = MultiChannelMalConv(num_embeddings, embedding_dim, out_channels, kernel_size, stride, num_classes)
        x = [torch.randint(0, v, (self.B, seq_length), dtype=torch.long) for v in num_embeddings]
        y = model.forward(*x)
        assert y.shape == (self.B, num_classes)

    @pytest.mark.parametrize("seq_length", [17, 53, 256])
    @pytest.mark.parametrize("num_embeddings", [[8], [16, 32], [8, 16, 32]])
    @pytest.mark.parametrize("embedding_dim", [[4], [8, 16], [4, 8, 16]])
    @pytest.mark.parametrize("out_channels", [16, 32])
    @pytest.mark.parametrize("kernel_size", [3, 5])
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("num_classes", [1, 2, 3])
    def test_forward(
        self,
        seq_length: int,
        num_embeddings: list[int],
        embedding_dim: list[int],
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_classes: int,
    ):
        if len(num_embeddings) != len(embedding_dim):
            return
        self._test_forward(seq_length, num_embeddings, embedding_dim, out_channels, kernel_size, stride, num_classes)

    @pytest.mark.parametrize("seq_length", [1, 2, 3, 11, 17, 63])
    def test_forward_seq_length_too_short(self, seq_length: int):
        # Any sequence length less than the kernel size should raise an error.
        with pytest.raises(RuntimeError):
            self._test_forward(
                seq_length,
                num_embeddings=[8],
                embedding_dim=[8],
                out_channels=64,
                kernel_size=64,
                stride=1,
                num_classes=1,
            )
