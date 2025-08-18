"""
Neural architectures and their components.

Notations:
    B: Batch size
    T: Sequence length
    N: Number of patches
    P: Patch size
    E: Embedding dimension
    C: Number of channels
    S: Post-conv length
    H: Hidden dimension
    M: Number of classes
"""

import math
import sys
from typing import Protocol

import torch
from torch import nn
from torch import Tensor
from torch import IntTensor
from torch import FloatTensor
from torch.nn.utils.rnn import pad_sequence

from src.utils import TensorError
from src.utils import check_tensor


class ClassifificationHead(nn.Module):  # type: ignore[misc]
    """
    Classification head for a neural network.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = -1,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        if num_layers > 1 and hidden_size <= 0:
            raise ValueError("Hidden size must be positive for multi-layer heads.")

        def create_hidden_layer(in_size: int) -> list[nn.Module]:
            return [
                nn.Linear(in_size, hidden_size),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
            ]

        self.layers = nn.Sequential()
        for i in range(num_layers - 1):
            layers = create_hidden_layer(input_size if i == 0 else hidden_size)
            for layer in layers:
                self.layers.append(layer)
        self.layers.append(nn.Linear(hidden_size if num_layers > 1 else input_size, num_classes))

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor of shape (B, H).
        Returns:
            Output tensor of shape (B, M).
        """
        check_tensor(x, (None, self.input_size), torch.float)

        z = self.layers.forward(x)

        return z


class MultiChannelDiscreteSequenceClassifier(Protocol):
    """
    Multi-channel classifier for discrete sequence inputs.
    """

    def __init__(self, num_embeddings: list[int], embedding_dim: list[int], num_classes: int) -> None: ...

    def forward(self, *x: IntTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor(s) of shape (B, T).
        Returns:
            Output tensor of shape (B, C).
        """
        ...


class LatentPooler(Protocol):
    """
    Pooling layer for latent representations.
    """

    def forward(self, z: FloatTensor) -> FloatTensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, E).
        """
        ...


class MeanLatentPooler(nn.Module):  # type: ignore[misc]
    """
    Mean pooling layer for latent representations.
    """

    def forward(self, z: FloatTensor) -> FloatTensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, E).
        """
        check_tensor(z, (None, None, None), torch.float)
        return z.mean(dim=1)


class MultiChannelDiscreteEmbedding(nn.Module):  # type: ignore[misc]
    """
    Multi-channel embedding layer for discrete sequence inputs.
    """

    def __init__(self, num_embeddings: list[int], embedding_dim: list[int]) -> None:
        super().__init__()

        self.embedding = nn.ModuleList()
        for num_embeddings_, embedding_dim_ in zip(num_embeddings, embedding_dim, strict=True):
            embedding = nn.Embedding(num_embeddings_, embedding_dim_)
            self.embedding.append(embedding)

    @property
    def max_length(self) -> int:
        return sys.maxsize

    @property
    def min_length(self) -> int:
        return 0

    def forward(self, *x: IntTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor(s) of shape (B, T).
        Returns:
            Output tensor of shape (B, T, E).
        """
        for x_ in x:
            check_tensor(x_, tuple(x[0].shape), torch.int64)

        z = torch.cat([self.embedding[i].forward(x[i]) for i in range(len(x))], dim=-1)

        return z


class SequenceEmbeddingEncoder(nn.Module):  # type: ignore[misc]
    """
    Patch embedding layer for discrete sequence inputs.
    """

    def __init__(
        self, patch_size: int, in_channels: int, out_channels: int, kernel_size: int = 64, stride: int = 64
    ) -> None:
        super().__init__()

        if patch_size < kernel_size:
            raise ValueError("Patch size must be greater than or equal to kernel size.")

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    @property
    def max_length(self) -> int:
        return sys.maxsize

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def forward(self, z: FloatTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, C).
        """
        check_tensor(z, (None, None, None), torch.float)

        B = z.shape[0]
        T = z.shape[1]
        E = z.shape[2]
        P = min(self.patch_size, T)
        N = math.ceil(T / P)
        C = self.out_channels
        S = math.floor((T - self.kernel_size) / self.stride + 1)

        if T < self.min_length:
            raise RuntimeError(f"Input sequence length {T} is less than the minimum required length {self.min_length}.")

        z = self.split_patches(z)  # (B,  N, P, E)
        z = z.reshape(B * N, P, E)  # (BN, P, E)
        z = z.permute(0, 2, 1)  # (BN, E, P)
        z = self.conv.forward(z)  # (BN, C, S)
        z = self.pool.forward(z)  # (BN, C, 1)
        z = z.squeeze(-1)  # (BN, C)
        z = z.reshape(B, N, C)  # (B,  N, C)

        return z

    @staticmethod
    def _split_patches(z: FloatTensor, patch_size: int) -> FloatTensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, P, E).
        """
        check_tensor(z, (None, None, None), torch.float)

        if z.shape[1] <= patch_size:
            return z.unsqueeze(1)

        patches = torch.split(z.permute(1, 0, 2), patch_size)  # N x (P, B, E)
        patches = pad_sequence(patches, batch_first=True)  # (N, P, B, E)
        patches = patches.permute(2, 0, 1, 3)  # (B, N, P, E)

        return patches

    def split_patches(self, z: FloatTensor) -> FloatTensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, P, E).
        """
        check_tensor(z, (None, None, self.in_channels), torch.float)
        return self._split_patches(z, self.patch_size)


class MultiChannelDiscreteSequenceVisionTransformer(nn.Module):  # type: ignore[misc]
    """
    Vision Transformer for discrete sequential inputs.
    """

    def __init__(
        self,
        num_embeddings: list[int],
        embedding_dim: list[int],
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.embedding = MultiChannelDiscreteEmbedding(num_embeddings, embedding_dim)
        self.encoder = SequenceEmbeddingEncoder(patch_size, sum(embedding_dim), d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, activation="gelu", batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers, norm=nn.RMSNorm(d_model))
        self.pool = MeanLatentPooler()
        self.head = ClassifificationHead(d_model, num_classes)

        self.num_embeddings = num_embeddings
        self.embedding_dim = sum(embedding_dim)
        self.patch_size = patch_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes

    @property
    def max_length(self) -> int:
        mods = [self.embedding, self.encoder, self.backbone, self.head]
        return min(getattr(mod, "max_length", 0) for mod in [mods])

    @property
    def min_length(self) -> int:
        mods = [self.embedding, self.encoder, self.backbone, self.head]
        return max(getattr(mod, "min_length", 0) for mod in [mods])

    def forward(self, *x: IntTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor(s) of shape (B, T).
        Returns:
            Output tensor of shape (B, M).
        """
        for x_ in x:
            check_tensor(x_, tuple(x[0].shape), torch.int64)

        print(x[0].shape)
        z = self.embedding.forward(*x)  # (B, T, E)
        z = self.encoder.forward(z)  # (B, N, H)
        z = self.backbone.forward(z)  # (B, N, H)
        z = self.pool.forward(z)  # (B, H)
        z = self.head.forward(z)  # (B, M)

        return z


class MultiChannelMalConv(nn.Module):  # type: ignore[misc]
    """
    MalConv model for discrete sequential inputs.
    """

    def __init__(
        self,
        num_embeddings: list[int],
        embedding_dim: list[int],
        out_channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.embedding = MultiChannelDiscreteEmbedding(num_embeddings, embedding_dim)
        self.conv_1 = nn.Conv1d(sum(embedding_dim), out_channels, kernel_size, stride)
        self.conv_2 = nn.Conv1d(sum(embedding_dim), out_channels, kernel_size, stride)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.head = ClassifificationHead(out_channels, num_classes)

        self.num_embeddings = num_embeddings
        self.embedding_dim = sum(embedding_dim)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_classes = num_classes

    @property
    def max_length(self) -> int:
        return sys.maxsize

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def forward(self, *x: IntTensor) -> FloatTensor:
        """
        Args:
            x: Input tensor of shape (B, T).
        Returns:
            Output tensor of shape (B, M).
        """
        for x_ in x:
            check_tensor(x_, tuple(x[0].shape), torch.int64)

        B = x[0].shape[0]
        T = x[0].shape[1]
        E = self.embedding_dim
        C = self.out_channels
        S = math.floor((T - self.kernel_size) / self.stride + 1)

        z = self.embedding.forward(*x)  # [B, L, E]
        z = z.transpose(1, 2)  # [B, E, L]
        c_1 = self.conv_1.forward(z)  # [B, C, S - 1]
        c_2 = self.conv_2.forward(z)  # [B, C, S - 1]
        g = c_1 * self.sigmoid.forward(c_2)  # [B, C, S - 1]
        z = self.pool.forward(g)  # [B, C, 1]
        z = z.squeeze(-1)  # [B, C]
        z = self.head.forward(z)  # [B, M]

        return z
