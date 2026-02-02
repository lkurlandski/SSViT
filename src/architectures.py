"""
Neural architectures and their components.

Notations (general):
    B: Batch size
    T: Sequence length
    E: Embedding dimension
    G: Guide dimension
    M: Number of classes

Notations (MalConv):
    C: Number of channels
    K: Kernel size
    W: Stride
    S: Post-conv length, i.e., ⌊(T - K) / W + 1)⌋

Notations (ViT):
    N: Number of patches
    P: Patch size
    D: Hidden dimension
    H: Num attention heads
    L: Number of layers
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
from inspect import signature
from itertools import chain
import math
import os
import sys
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Protocol
import warnings

import torch
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn
from torch import Tensor

from src.lowmemcore import DISABLE_LOW_MEMORY_PATHS
from src.lowmemcore import functional_forward
from src.lowmemcore import PreprocessFn
from src.lowmemcore import _get_conv_kwds
from src.lowmemcore import _check_lowmem_config
from src.lowmemcore import _lowmem_max_over_time_streaming
from src.lowmemcore import _lowmem_patchwise_max_over_time_streaming
from src.lowmemcore import _lowmem_patchwise_max_over_time_dispatched
from src.lowmemcore import _gather_wins_via_preprocess_batched
from src.lowmemcore import _scatter_g_to_BNC
from src.lowmemcore import _scatter_g_to_BC
from src.utils import check_tensor
from src.utils import TensorError
from src.utils import pad_sequence


# mypy: disable-error-code=no-any-return


STRUCTURAL_CLASSIFIER_USE_DYNAMIC_RECONCILE = os.environ.get("STRUCTURAL_CLASSIFIER_USE_DYNAMIC_RECONCILE", "0") == "1"
if STRUCTURAL_CLASSIFIER_USE_DYNAMIC_RECONCILE:
    warnings.warn("Using dynamic reconcile for structural classifier. This may have performance implications.")


NORMS: dict[str, type[nn.Module]] = {
    "layer": nn.LayerNorm,
    "rms": nn.RMSNorm,
}

ACTVS: dict[str, Callable[[Tensor], Tensor]] = {
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "relu": F.relu,
    "silu": F.silu,
    "tanh": F.tanh,
}

FLOATS = (torch.bfloat16, torch.float16, torch.float32)  # Commonly used for training and evaluating models.
INTEGERS = (torch.int32, torch.int64)                    # Used for indices compatible with nn.Embedding.


def get_model_input_lengths(model: nn.Module) -> tuple[int, int]:
    lo = 0
    hi = sys.maxsize

    for m in model.modules():
        if isinstance(l := getattr(m, "max_length", None), int):
            hi = min(hi, l)
        if isinstance(l := getattr(m, "min_length", None), int):
            lo = max(lo, l)

    return lo, hi


def _ddp_keepalive(out: Tensor, parameters: Iterable[nn.Parameter]) -> Tensor:
    """
    Ensure all parameters participate in the autograd graph every step.
    This avoids DDP 'unused parameter' errors when some trunks have no inputs.

    Usage:
    >>> out = _ddp_keepalive(out, self.net.parameters()) if self.training else out
    """
    if not torch.is_grad_enabled():
        return out

    views: list[Tensor] = []
    for p in parameters:
        if p.requires_grad:
            views.append(p.view(-1).select(0, 0))

    if not views:
        return out

    dummy = torch.stack(views).sum() * 0.0
    return out + dummy


class Identity(nn.Module):
    """
    Identity layer with a bit more functionality.

    Instead of just accepting one arg to `forward`, it accepts an arbitrary number of args,
    and furthermore will autocast floating point inputs if `autocast=True`.
    """

    def __init__(self, *args: Any, autocast: bool = False, **kwds: Any) -> None:
        super().__init__()
        self.autocast = autocast
        if self.autocast:
            warnings.warn(f"Using {self.__class__.__name__}(autocast=True) is almost certainly a poor decision and this feature will eventually be removed.")

    def forward(self, x: Tensor, *args: Any, **kwds: Any) -> Tensor:
        if self.autocast and torch.is_floating_point(x) and torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype(x.device.type))
        return x


# -------------------------------------------------------------------------------- #
# Other
# -------------------------------------------------------------------------------- #

class ClassifificationHead(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, D).
        Returns:
            Output tensor of shape (B, M).
        """
        check_tensor(x, (None, self.input_size), FLOATS)

        z = self.layers(x)

        return z


class DWBlock(nn.Module):
    """
    Depthwise convolutional block.

    Notes:
        S: stride, default 1
        X: expansion factor, default 4
        C: dimensionality
        K: kernel size

    Compute ~ T / S x (C x K + 2 x C^2 x X)
            ~ T x C^2 x X  # C x K << C^2 x X
    """

    def __init__(self, dim: int, k: int = 7, expansion: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.dw = nn.Conv1d(dim, dim, kernel_size=k, padding=k//2, groups=dim)
        self.pw1 = nn.Conv1d(dim, dim * expansion, kernel_size=1)
        self.pw2 = nn.Conv1d(dim * expansion, dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, dim)
        self.drop = nn.Dropout(drop)

    @property
    def min_length(self) -> int:
        k = int(self.dw.kernel_size[0])
        p = int(self.dw.padding[0])
        return max(1, k - 2 * p)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, C, T).
        Returns:
            y: Output embeddings of shape (B, C, T).
        """
        check_tensor(x, (None, self.dim, None), FLOATS)
        B, C, T = x.shape
        y = self.dw(x)    # (B, C, T)
        y = self.norm(y)  # (B, C, T)
        y = self.pw1(y)   # (B, C*expansion, T)
        y = F.gelu(y)     # (B, C*expansion, T)
        y = self.drop(y)  # (B, C*expansion, T)
        y = self.pw2(y)   # (B, C, T)
        y = x + y         # (B, C, T)
        check_tensor(y, (B, C, T), FLOATS)
        return y


class AdaptiveAtnPooling1d(nn.Module):
    """
    Learned query attention pooling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(dim, 1)

    @property
    def min_length(self) -> int:
        return 1

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, C, T).
            key_padding_mask: Optional mask of shape (B, T) indicating padding positions.
                These must correspond to the sequence length of the input.
        Returns:
            y: Output embeddings of shape (B, C).
        """
        check_tensor(x, (None, self.dim, None), FLOATS)
        if key_padding_mask is not None:
            check_tensor(key_padding_mask, (x.shape[0], x.shape[2]), torch.bool)
        B, C, T = x.shape
        logits: Tensor

        x_t = x.transpose(1, 2)                      # (B, T, C)
        logits = self.w(x_t).squeeze(-1)             # (B, T)
        if key_padding_mask is not None:
            neg = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(key_padding_mask, neg)

        a = torch.softmax(logits, dim=-1)            # (B, T)
        if key_padding_mask is not None:
            a = a.masked_fill(key_padding_mask, 0.0)
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        y = torch.sum(x_t * a.unsqueeze(-1), dim=1)  # (B, C)
        check_tensor(y, (B, C), FLOATS)
        return y


class DWCSequenceEncoder(nn.Module):
    """
    Sequence encoder using depthwise convolutions.

    See: Liu et al. "A ConvNet for the 2020s" CVPR 2022.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        kernel_size: int = 64,
        stride: int = 64,
        pooling: Literal["max", "avg", "atn"] = "atn",
        checkpoint_segments: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = pooling
        self.checkpoint_segments = min(checkpoint_segments, depth) if checkpoint_segments > -1 else depth

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.GELU(),
            nn.GroupNorm(1, out_channels),
        )
        self.blocks = nn.Sequential(*[DWBlock(out_channels) for _ in range(depth)])
        self.pool: nn.Module
        if pooling == "max":
            self.pool = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        elif pooling == "avg":
            self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif pooling == "atn":
            self.pool = AdaptiveAtnPooling1d(out_channels)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}.")

    @property
    def min_length(self) -> int:
        return max(1, self.kernel_size)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, E, T).
            key_padding_mask: Optional mask of shape (B, T) indicating padding positions.
                These must correspond to the sequence length of the input.
        Returns:
            y: Output embeddings of shape (B, C).
        """
        check_tensor(x, (None, self.in_channels, None), FLOATS)
        if key_padding_mask is not None:
            check_tensor(key_padding_mask, (x.shape[0], x.shape[2]), torch.bool)
        B, E, T = x.shape
        C = self.out_channels

        x = self.stem(x)    # (B, C, T')
        # The Norm at the end of the stem will upcast to float32, which then propagates
        # in every subsequent block, leading to a large number of dtype juggling and copies.
        # To elimiinate this, all we need to do is downcast back to the original dtype here.
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype(x.device.type))
        if self._should_checkpoint() and x.requires_grad:
            x = checkpoint_sequential(self.blocks, self.checkpoint_segments, x, use_reentrant=False)  # type: ignore[no-untyped-call]
        else:
            x = self.blocks(x)  # (B, C, T')

        if self.pooling == "atn":
            if key_padding_mask is not None:
                key_padding_mask = self._downsample_key_padding_mask(key_padding_mask, x.shape[2])
            x = self.pool(x, key_padding_mask)  # (B, C)
        else:
            x = self.pool(x)                    # (B, C)

        check_tensor(x, (B, C), FLOATS)
        return x

    def _should_checkpoint(self) -> bool:
        return self.checkpoint_segments > 0 and self.training and torch.is_grad_enabled()

    def _downsample_key_padding_mask(self, key_padding_mask: Tensor, T_out: int) -> Tensor:
        """
        Downsample (B, T) padding mask to (B, T_out) using the stem's (k, s, p).
        A latent token is padding iff its entire receptive-field window is padding.
        """
        # Get valid mask, then downsample via max-pooling
        valid = (~key_padding_mask).to(dtype=torch.float32).unsqueeze(1)  # (B, 1, T)
        valid_ds = F.max_pool1d(
            valid,
            kernel_size=self.kernel_size,
            stride=self.stride,
        ).squeeze(1)

        # True where all-padding
        kp_ds = (valid_ds == 0)

        # Match the stem output length exactly
        if kp_ds.shape[1] > T_out:
            kp_ds = kp_ds[:, :T_out]
        else:
            kp_ds = F.pad(kp_ds, (0, T_out - kp_ds.shape[1]), value=True)

        return kp_ds

# -------------------------------------------------------------------------------- #
# FiLM
# -------------------------------------------------------------------------------- #

class FiLM(nn.Module):
    """
    Feature-wise linear modulation (FiLM) layer.

    See: Perez "Film: Visual reasoning with a general conditioning layer" AAAI 2018.
    """

    def __init__(self, guide_dim: int, embedding_dim: int, hidden_size: int, fp32: bool = False):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(guide_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * embedding_dim)
        )

        self.guide_dim = guide_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.fp32 = fp32

        if self.fp32:
            raise NotImplementedError()

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, T, E).
            g: FiLM conditioning vector of shape (B, T, G).

        Returns:
            z: FiLM modulated embeddings of shape (B, T, E).
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        check_tensor(g, (x.shape[0], x.shape[1], self.guide_dim), FLOATS)
        if g.requires_grad:
            warnings.warn(f"FiLM conditioning vector `g` has `requires_grad={g.requires_grad}`.")

        film: Tensor = self.mlp(g)               # (B, T, 2E)
        gamma, beta = film.chunk(2, dim=-1)      # (B, T, E), (B, T, E)
        z = x.to(gamma.dtype) * gamma + beta     # (B, T, E)

        check_tensor(z, (x.shape[0], x.shape[1], self.embedding_dim), FLOATS)
        return z

    @torch.no_grad()
    def forward_functional(self, x: Tensor, g: Tensor) -> Tensor:
        """
        Functional version for low-memory streaming scans.
        """
        lin1: nn.Linear = self.mlp[0]  # type: ignore[assignment]
        lin2: nn.Linear = self.mlp[2]  # type: ignore[assignment]
        if not isinstance(lin1, nn.Linear) or not isinstance(lin2, nn.Linear):
            raise RuntimeError("FiLM.forward_functional only supports the standard MLP structure.")

        w1 = lin1.weight.detach()
        b1 = lin1.bias.detach() if lin1.bias is not None else None
        w2 = lin2.weight.detach()
        b2 = lin2.bias.detach() if lin2.bias is not None else None

        film = F.linear(F.relu(F.linear(g, w1, b1)), w2, b2)
        gamma, beta = film.chunk(2, dim=-1)
        z = x.to(gamma.dtype) * gamma + beta

        check_tensor(z, (x.shape[0], x.shape[1], self.embedding_dim), FLOATS)
        return z


class FiLMNoP(nn.Module):
    """
    No-op FiLM layer that does nothing but check the inputs.
    """

    def __init__(self, guide_dim: int, embedding_dim: int, hidden_size: int, autocast: bool = False):
        super().__init__()

        self.guide_dim = guide_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.autocast = autocast
        if self.autocast:
            warnings.warn(f"Using {self.__class__.__name__}(autocast=True) is almost certainly a poor decision and this feature will eventually be removed.")

    def forward(self, x: Tensor, g: Literal[None]) -> Tensor:
        check_tensor(x, (None, None, None), FLOATS)
        if g is not None:
            raise ValueError(f"Expected g to be None, got {type(g)} instead.")

        if self.autocast and torch.is_floating_point(x) and torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype(x.device.type))

        return x

    @torch.no_grad()
    def forward_functional(self, x: Tensor, g: Literal[None]) -> Tensor:
        """
        Functional version for low-memory streaming scans.
        """
        if g is not None:
            raise ValueError(f"Expected g to be None, got {type(g)} instead.")

        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype(x.device.type))

        return x

# -------------------------------------------------------------------------------- #
# Positional Encodings
# -------------------------------------------------------------------------------- #

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    See: Vaswani "Attention is all you need" NeurIPS 2017.

    Source: https://github.com/tatp22/multidim-positional-encoding
    """

    inv_freq: Tensor

    def __init__(self, embedding_dim: int, fp32: bool = False) -> None:
        super().__init__()

        if embedding_dim % 2 != 0 or embedding_dim <= 0:
            raise ValueError(f"The embedding dimension must a positive be even number. Got {embedding_dim} instead.")

        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim))
        self.register_buffer("inv_freq", inv_freq)

        self.embedding_dim = embedding_dim
        self.fp32 = fp32

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, E).

        Returns:
            z: Positional encoded tensor of shape (B, T, E).
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)

        if x.dtype != self.inv_freq.dtype and not self.fp32:
            self.inv_freq = self.inv_freq.to(x.dtype)

        pos = torch.arange(x.shape[1], device=x.device, dtype=self.inv_freq.dtype)  # (T,)
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)                       # (T, E/2)

        p = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)                     # (T, E/2, 2)
        p = p.flatten(-2, -1)                                                       # (T, E)
        p = p.repeat(x.shape[0], 1, 1)                                              # (B, T, E)
        p = p.to(x.dtype)

        z = p + x

        check_tensor(z, (x.shape[0], x.shape[1], self.embedding_dim), FLOATS)

        return z


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.

    See: Devlin "BERT: Pre-training of deep bidirectional transformers for language understanding" NAACL 2019.
    """

    def __init__(self, embedding_dim: int, max_len: int, fp32: bool = False) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(f"The embedding dimension must be positive. Got {embedding_dim} instead.")
        if max_len <= 0:
            raise ValueError(f"`max_len` must be positive. Got {max_len} instead.")

        self.embedding = nn.Embedding(max_len, embedding_dim)

        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.fp32 = fp32

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, E).

        Returns:
            z: Positional encoded tensor of shape (B, T, E).
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)

        B, T, E = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds maximum supported length {self.max_len}.")

        idx = torch.arange(T, device=x.device, dtype=torch.int32)  # (T,)

        p: Tensor = self.embedding(idx)                            # (T, E)
        p = p.to(torch.float32 if self.fp32 else x.dtype)          # (T, E)
        p = p.unsqueeze(0).expand(B, T, E)                         # (B, T, E)

        z = x + p                                                  # (B, T, E)

        check_tensor(z, (B, T, self.embedding_dim), FLOATS)
        return z


# -------------------------------------------------------------------------------- #
# Patch Encoders
# -------------------------------------------------------------------------------- #

class PatchEncoderBase(nn.Module, ABC):
    """
    Breaks a sequence into patches and encodes them.

    Args:
        in_channels: Input dimension.
        out_channels: Output dimension.
        num_patches: Number of patches. If specified, patch_size must be None.
        patch_size: Size of each patch. If specified, num_patches must be None.

    Returns:
        Output tensor of shape (B, N, C).

    Notes:
        Exactly one of `num_patches` or `patch_size` must be specified. If `patch_size` is
        specified, the number of patches will vary dynamically based on the input sequence length.
        If `num_patches` is specified, the patch size will vary dynamically based on the input
        sequence length. In either case, a minimum input sequence length is required, `min_length`.
    """

    def __init__(self, in_channels: int, out_channels: int, num_patches: Optional[int], patch_size: Optional[int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._num_patches = num_patches
        self._patch_size = patch_size

        if (num_patches is None) == (patch_size is None):
            raise ValueError(f"{self.__class__.__name__} requires `num_patches` is None xor `patch_size` is None.")

    @property
    def num_patches(self) -> Optional[int]:
        """Accessor for _num_patches (supports type override)."""
        return self._num_patches

    @property
    def patch_size(self) -> Optional[int]:
        """Accessor for _patch_size (supports type override)."""
        return self._patch_size

    @property
    def min_length(self) -> int:
        """Minimum input sequence length required."""
        return 1

    def resolve_patch_dims(self, seq_length: int) -> tuple[int, int]:
        """
        Determine (patch_size, num_patches) for a given sequence length.
        """
        return self.compute_patch_dims(seq_length, self.patch_size, self.num_patches)

    @staticmethod
    def compute_patch_dims(seq_length: int, patch_size: Optional[int], num_patches: Optional[int]) -> tuple[int, int]:
        """
        Determine (patch_size, num_patches) for a given sequence length.
        """
        if seq_length <= 0:
            raise ValueError(f"{seq_length=} must be positive.")
        if bool(patch_size is None) == bool(num_patches is None):
            raise ValueError(f"Exactly one of {patch_size=} or {num_patches=} must be specified.")

        T = seq_length

        # Fixed patch_size, dynamic num_patches.
        if patch_size is not None:
            if patch_size <= 0:
                raise ValueError(f"{patch_size=} must be positive.")
            if patch_size > T:
                warnings.warn(f"{patch_size=} greater than sequence length {T}.")
            P = patch_size
            N = math.ceil(T / P)
            return P, N

        # Fixed num_patches, dynamic patch_size.
        if num_patches is not None:
            if num_patches <= 0:
                raise ValueError(f"{num_patches=} must be positive.")
            if num_patches > T:
                warnings.warn(f"{num_patches=} greater than sequence length {T}.")
            N = num_patches
            P = math.ceil(T / N)
            return P, N

        raise RuntimeError("Unreachable.")

    @staticmethod
    def pad_to_multiple(z: Tensor, multiple: int) -> Tensor:
        """
        Right-pad z along the temporal dimension to a multiple of `multiple`.

        Args:
            z: (B, T, E)
            multiple: positive integer

        Returns:
            Output tensor of shape (B, T', E) with T' % multiple == 0 and T' >= T.
        """
        if multiple <= 0:
            raise ValueError(f"`multiple` must be positive. Got {multiple}.")

        B, T, E = z.shape
        remainder = T % multiple
        if remainder == 0:
            return z

        pad_len = multiple - remainder
        pad = z.new_zeros(B, pad_len, E)
        return torch.cat([z, pad], dim=1)

    @staticmethod
    def pad_to_length(z: Tensor, length: int) -> Tensor:
        """
        Right-pad z along the temporal dimension to a length of `length`.

        Args:
            z: (B, T, E)
            length: positive integer (T')

        Returns:
            Output tensor of shape (B, T', E) with T' >= T.
        """
        if length <= 0:
            raise ValueError(f"`length` must be positive. Got {length}.")

        B, T, E = z.shape
        if T > length:
            raise ValueError(f"`length` must be at least the input sequence length. Got {length} < {T}.")
        if T == length:
            return z

        pad_len = length - T
        pad = z.new_zeros(B, pad_len, E)
        return torch.cat([z, pad], dim=1)

    def forward(self, z: Optional[Tensor] = None, preprocess: Optional[PreprocessFn] = None, ts: Optional[Sequence[Tensor]] = None) -> Tensor:
        """
        Args:
            z: input tensor of shape (B, T, E). If provided, will invoke the `forward_embeddings` method.
            preprocess: callable to compute the input tensor. If provided, will invoke the `forward_streaming` method.
            ts: arguments to `preprocess`, each of shape (B, T, *). If provided, will invoke the `forward_streaming` method.

        Returns:
            Output tensor of shape (B, N, C).
        """
        if z is not None:
            check_tensor(z, (None, None, self.in_channels), FLOATS)
            if z.shape[1] < self.min_length:
                raise RuntimeError(f"Input sequence length {z.shape[1]} is less than the minimum required length {self.min_length}.")
            z = self.forward_embeddings(z)

        elif preprocess is not None and ts is not None:
            for t in ts:
                if t.shape[0] != ts[0].shape[0]:
                    raise TensorError(t, (ts[0].shape[0], ts[0].shape[1], None), None)
                if t.shape[1] != ts[0].shape[1]:
                    raise TensorError(t, (ts[0].shape[0], ts[0].shape[1], None), None)
                if t.shape[1] < self.min_length:
                    raise RuntimeError(f"Input sequence length {t.shape[1]} is less than the minimum required length {self.min_length}.")
            if DISABLE_LOW_MEMORY_PATHS:
                z = self.forward_embeddings(preprocess(*ts))
            else:
                z = self.forward_streaming(preprocess=preprocess, ts=ts)

        else:
            raise ValueError("Either `z` or both `preprocess` and `ts` must be provided.")

        check_tensor(z, (z.shape[0], self.num_patches, self.out_channels), FLOATS)
        return z

    @abstractmethod
    def forward_embeddings(self, z: Tensor) -> Tensor:
        ...

    @abstractmethod
    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        ...


class PatchEncoder(PatchEncoderBase):
    """
    Breaks a sequence into patches and encodes them via convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        *,
        kernel_size: int = 64,
        stride: int = 64,
        pooling: Literal["max", "avg", "atn"] = "max",
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if bool(patch_size is None) == bool(num_patches is None):
            raise ValueError(f"Exactly one of patch_size or num_patches must be specified. Got {patch_size=} and {num_patches=}.")

        if patch_size is not None and patch_size < kernel_size:
            raise ValueError(f"Patch size must be greater than or equal to kernel size. Got {patch_size=} and {kernel_size=}.")

        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.pool: nn.AdaptiveMaxPool1d | nn.AdaptiveAvgPool1d | AdaptiveAtnPooling1d
        if pooling == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "atn":
            self.pool = AdaptiveAtnPooling1d(out_channels)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}.")

    @property
    def min_length(self) -> int:
        if self.patch_size is not None:
            return 1
        if self.num_patches is not None:
            return self.num_patches * self.kernel_size
        raise RuntimeError("This should never happen.")

    def forward_embeddings(self, z: Tensor) -> Tensor:
        B, T, E = z.shape
        P, N = self.resolve_patch_dims(T)
        C = self.out_channels

        z = PatchEncoderBase.pad_to_length(z, N * P)  # (B, NP, E)
        z = z.view(B, N, P, E)        # (B,  N, P, E)
        z = z.reshape(B * N, P, E)    # (BN, P, E)
        z = z.permute(0, 2, 1)        # (BN, E, P)
        z = self.conv(z)              # (BN, C, S)
        z = self.pool(z)              # (BN, C, 1)
        z = z.squeeze(-1)             # (BN, C)
        z = z.reshape(B, N, C)        # (B,  N, C)

        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        return self.forward_embeddings(preprocess(*ts))


class ConvPatchEncoder(PatchEncoderBase):
    """
    Breaks a sequence into fixed-sized patches and encodes them via convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if num_patches is not None or patch_size is None:
            raise ValueError(f"{self.__class__.__name__} requires `num_patches` is None and `patch_size` is not None.")

        self.kernel_size = patch_size
        self.stride = patch_size

        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def num_patches(self) -> None:
        assert self._num_patches is None
        return self._num_patches

    @property
    def patch_size(self) -> int:
        assert self._patch_size is not None
        return self._patch_size

    @property
    def min_length(self) -> int:
        return 1

    def forward_embeddings(self, z: Tensor) -> Tensor:
        B, T, E = z.shape
        P, N = self.resolve_patch_dims(T)

        z = PatchEncoderBase.pad_to_multiple(z, P)  # (B, T', E)
        z = z.permute(0, 2, 1)      # (B, E, T')
        z = self.proj(z)            # (B, C, N)
        z = z.permute(0, 2, 1)      # (B, N, C)

        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        return self.forward_embeddings(preprocess(*ts))


class HierarchicalConvPatchEncoder(PatchEncoderBase):
    """
    Two-stage hierarchical convolutional patch encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        *,
        hidden_channels: Optional[int] = None,
        s1: Optional[int] = None,
        s2: Optional[int] = None,
        activation: Optional[str] = "leaky_relu",
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if num_patches is not None or patch_size is None:
            raise ValueError(f"{self.__class__.__name__} requires `num_patches` is None and `patch_size` is not None.")

        if s1 is None and s2 is None:
            factors = HierarchicalConvPatchEncoder.compute_factors(patch_size)
            s1, s2 = min(factors, key=lambda x: abs(x[0] - x[1]))
            if patch_size != s1 * s2:
                raise RuntimeError(f"`s1` and `s2` must be a factorization of `patch_size`. Got {patch_size=} {s1=} {s2=} {s1*s2=}.")
        elif s1 is None and s2 is not None:
            if patch_size % s2 != 0:
                raise ValueError(f"`patch_size` must be divisible by `s2`. Got {patch_size=} {s1=} {s2=}.")
            s1 = patch_size // s2
        elif s1 is not None and s2 is None:
            if patch_size % s1 != 0:
                raise ValueError(f"`patch_size` must be divisible by `s1`. Got {patch_size=} {s1=} {s2=}.")
            s2 = patch_size // s1
        elif s1 is not None and s2 is not None:
            if patch_size != s1 * s2:
                raise ValueError(f"`s1` and `s2` must be a factorization of `patch_size`. Got {patch_size=} {s1=} {s2=} {s1*s2=}.")

        hidden_channels = hidden_channels if hidden_channels is not None else out_channels

        assert s1 is not None
        assert s2 is not None
        assert hidden_channels is not None
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=s1, stride=s1)
        self.conv_2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=s2, stride=s2)
        self.actvfn = ACTVS[activation] if activation is not None else nn.Identity()

    @property
    def num_patches(self) -> None:
        assert self._num_patches is None
        return self._num_patches

    @property
    def patch_size(self) -> int:
        assert self._patch_size is not None
        return self._patch_size

    @property
    def min_length(self) -> int:
        return 1

    @staticmethod
    def compute_factors(n: int) -> list[tuple[int, int]]:
        """
        Finds all factors of a given positive integer.
        """
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i, int(n / i)))
        return factors

    def forward_embeddings(self, z: Tensor) -> Tensor:
        B, T, E = z.shape
        P, N = self.resolve_patch_dims(T)

        z = PatchEncoderBase.pad_to_multiple(z, P)  # (B, T', E)
        z = z.permute(0, 2, 1)  # (B, E, T')
        z = self.conv_1(z)      # (B, H, T'')
        z = self.actvfn(z)      # (B, H, T'')
        z = self.conv_2(z)      # (B, C, N)
        z = z.permute(0, 2, 1)  # (B, N, C)

        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        return self.forward_embeddings(preprocess(*ts))


class PatchEncoderLowMem(PatchEncoderBase):
    """
    Breaks a sequence into a fixed number of patches and encodes them via constant memory convolution.

    See: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017.
    See: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022.
    See: Jang et al., "Categorial Reparameterization with Gumbel-Softmax", ICLR 2017.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        *,
        kernel_size: int = 64,
        stride: int = 64,
        chunk_size: int = 2 ** 16,
        overlap: Optional[int] = None,
        fp32: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if num_patches is None or patch_size is not None:
            raise ValueError(f"{self.__class__.__name__} requires `num_patches` is not None and `patch_size` is None.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = kernel_size // 2 if overlap is None else overlap
        self.fp32 = fp32

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        if self.overlap < self.kernel_size / 2:
            warnings.warn(f"Overlap {self.overlap} is less than half the kernel size {self.kernel_size}. Pooling may be impacted windowing issues.")

    @property
    def num_patches(self) -> int:
        assert self._num_patches is not None
        return self._num_patches

    @property
    def patch_size(self) -> None:
        assert self._patch_size is None
        return self._patch_size

    @property
    def min_length(self) -> int:
        return self.num_patches * self.kernel_size

    def forward_embeddings(self, z: Tensor) -> Tensor:
        B, T, E = z.shape
        P, N = self.resolve_patch_dims(T)
        C = self.out_channels

        z = PatchEncoderBase.pad_to_length(z, N * P)  # (B, NP, E)
        z = z.view(B, N, P, E)        # (B,  N, P, E)
        z = z.reshape(B * N, P, E)    # (BN, P, E)
        z = z.permute(0, 2, 1)        # (BN, E, P)

        g: Tensor = self.conv(z)                  # (BN, C, S)
        g, _ = g.max(dim=-1)                      # (BN, C)
        g = g.view(B, N, C)                       # (B, N, C)

        return g

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        B, T = ts[0].shape[0], ts[0].shape[1]
        E, C = self.in_channels, self.out_channels
        P, N = self.resolve_patch_dims(T)

        max_vals, pos = _lowmem_patchwise_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.kernel_size,
            first_stride=self.stride,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            channels=C,
            num_patches=self.num_patches,
            activations_fn=partial(functional_forward, module=self.conv, fp32=self.fp32),
        )

        if not self.training:
            return max_vals

        positions_flat = pos.view(B, N * C)
        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=preprocess,
            ts=ts,
            positions=positions_flat,
            rf=self.kernel_size,
            embedding_dim=E,
        )

        g_all: Tensor = self.conv(wins_cat)
        g_all = g_all.squeeze(-1)

        z = _scatter_g_to_BNC(
            g_all=g_all,
            meta=meta,
            batch_size=B,
            num_patches=self.num_patches,
            channels=C,
        )

        return z


class DWCPatchEncoder(PatchEncoderBase):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        *,
        depth: int,
        kernel_size: int = 64,
        stride: int = 64,
        pooling: Literal["max", "avg", "atn"] = "atn",
        checkpoint_segments: int = 0,
    ) -> None:
        """
        Args:
            checkpoint_segments: Number of segments for gradient checkpointing.
                If `-1`, will checkpoint the entire convolutional stack.
                If `0`,  will disable checkpointing entirely.
                If `N`,  will pass to DWCSequenceEncoder, which will checkpoint `N` blocks.
        """
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if patch_size is not None or num_patches != 1:
            raise ValueError(f"{self.__class__.__name__} requires `num_patches`=1 and `patch_size` is None.")

        self.checkpoint_segments = checkpoint_segments

        self.conv = DWCSequenceEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            kernel_size=kernel_size,
            stride=stride,
            pooling=pooling,
            checkpoint_segments=0 if checkpoint_segments == -1 else checkpoint_segments,
        )

    @property
    def min_length(self) -> int:
        return self.conv.min_length

    def forward_embeddings(self, z: Tensor) -> Tensor:
        z = z.permute(0, 2, 1)
        z = z.contiguous()
        if self.checkpoint_segments == -1:
            z = checkpoint(self.conv, z, use_reentrant=False)
        else:
            z = self.conv(z)
        z = z.unsqueeze(1)
        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        return self.forward_embeddings(preprocess(*ts))


class PatchEncoderLowMemSwitchMoE(PatchEncoderBase):
    """
    MoE patch encoder with constant-memory.

    See: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017.
    See: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022.
    See: Jang et al., "Categorial Reparameterization with Gumbel-Softmax", ICLR 2017.

    Args:
        in_channels: Input dimension.
        out_channels: Output dimension.
        num_patches: Number of patches. Must be specified.
        patch_size: Size of each patch. Must be None.
        kernel_size: Convolution kernel size (experts).
        stride: Convolution stride (experts).
        chunk_size: Low-memory streaming chunk size (experts and probe).
        overlap: Low-memory streaming overlap size (experts).
        fp32: Low-memory streaming precision mode (for experts and probe).
        num_experts: Number of experts in the Mixture-of-Experts.
        probe_dim: Dimension of the probe features used for routing.
        probe_kernel_size: Convolution kernel size (probe).
        probe_stride: Convolution stride (probe).
        probe_overlap: Low-memory streaming overlap size (probe).
        router_hidden: Hidden dimension of the routing MLP.
        router_temperature: Softmax temperature for routing.
        router_noise_std: Standard deviation of Gaussian noise added to the router logits during training.
        router_mode: Routing mode, either "ste" (straight-through estimator) or "soft" (soft routing).
        patch_batch_size: Batch size for patch processing during training. If None, uses input batch size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_patches: Optional[int],
        patch_size: Optional[int],
        *,
        kernel_size: int = 64,
        stride: int = 64,
        chunk_size: int = 2 ** 16,
        overlap: Optional[int] = None,
        fp32: bool = True,
        # MoE
        num_experts: int = 1,
        probe_dim: int = 16,
        probe_kernel_size: int = 256,
        probe_stride: int = 256,
        probe_overlap: Optional[int] = None,
        router_hidden: int = 256,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.0,
        router_mode: Literal["ste", "soft"] = "soft",
        router_top_k: int = 1,
        patch_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if num_patches is None or patch_size is not None:
            raise ValueError(f"{self.__class__.__name__} requires `num_patches` is not None and `patch_size` is None.")

        if num_experts <= 0:
            raise ValueError(f"{num_experts=} must be positive.")
        if probe_dim <= 0:
            raise ValueError(f"{probe_dim=} must be positive.")
        if router_hidden <= 0:
            raise ValueError(f"{router_hidden=} must be positive.")
        if router_temperature <= 0:
            raise ValueError(f"{router_temperature=} must be positive.")
        if router_noise_std < 0:
            raise ValueError(f"{router_noise_std=} must be non-negative.")
        if router_mode not in {"ste", "soft"}:
            raise ValueError(f"{router_mode=} must be one of 'ste' or 'soft'.")
        if router_top_k <= 0 or router_top_k > num_experts:
            raise ValueError(f"{router_top_k=} must be in the range [1, {num_experts}].")
        if router_mode == "ste" and router_top_k != 1:
            raise ValueError(f"Using `router_mode='ste'` with `router_top_k={router_top_k}` is not supported.")

        if num_experts == 1:
            warnings.warn(f"{self.__class__.__name__} instantiated with a single expert. Consider using `PatchEncoderLowMem` instead.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = kernel_size // 2 if overlap is None else overlap
        self.fp32 = fp32

        self.num_experts = num_experts
        self.probe_dim = probe_dim
        self.probe_kernel_size = probe_kernel_size
        self.probe_stride = probe_stride
        self.router_hidden = router_hidden
        self.probe_overlap = probe_kernel_size // 2 if probe_overlap is None else probe_overlap
        self.router_temperature = float(router_temperature)
        self.router_noise_std = float(router_noise_std)
        self.router_mode = router_mode
        self.patch_batch_size = patch_batch_size
        self.router_top_k = router_top_k

        self.probe = nn.Conv1d(in_channels, probe_dim, probe_kernel_size, probe_stride)
        self.router = nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, stride)
            for _ in range(num_experts)
        ])

        # NOTE: last_aux_loss will have `.requires_grad is True` but `.grad is None` during training.
        # After backward, `.grad` will be populated with the gradient value. This is normal.
        self.last_aux_loss = torch.zeros(())           # ()   Most recent auxiliary loss for load balancing.
        self.last_usage = torch.zeros((num_experts,))  # (E,) Most recent expert usage statistic.
        self.last_entropy = torch.zeros(())            # ()   Most recent routing entropy.

        _check_lowmem_config(kernel_size, stride, chunk_size, self.overlap)
        _check_lowmem_config(probe_kernel_size, probe_stride, chunk_size, self.probe_overlap)

    def __repr__(self) -> str:
        s: str = super().__repr__()  # type: ignore[no-untyped-call]
        add = (
            "("
            f"num_experts={self.num_experts}, "
            f"router_temperature={self.router_temperature}, "
            f"router_noise_std={self.router_noise_std}, "
            f"router_mode='{self.router_mode}', "
            f"router_top_k={self.router_top_k}"
            ")"
        )
        sub = f"{self.__class__.__name__}("
        s = s.replace(sub, sub + add)
        return s

    @property
    def conv(self) -> nn.Conv1d:
        return self.experts[0]  # type: ignore[return-value]

    @property
    def num_patches(self) -> int:
        assert self._num_patches is not None
        return self._num_patches

    @property
    def patch_size(self) -> None:
        assert self._patch_size is None
        return self._patch_size

    @property
    def min_length(self) -> int:
        return self.num_patches * max(self.kernel_size, self.probe_kernel_size)

    def _route(self, probe_feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Route patches to experts based on probe features.

        Args:
            probe_feat: Tensor of shape (B, N, D).

        Returns:
            dispatch: Tensor (bool) of shape (B, N, E) indicating which experts receive each patch.
            gates: Tensor (float) of shape (B, N, E) gating weights used to scale expert outputs.
            aux:  Tensor (float) of shape (,) indicating the load balancing auxillary loss.
        """
        assert self.router is not None

        B, N, D = probe_feat.shape

        x = probe_feat.reshape(B * N, D)

        # Get the logits for each expert.
        logits: Tensor = self.router(x)               # (BN, E)
        logits = logits.view(B, N, self.num_experts)  # (B, N, E)

        # Add noise for exploration (training only).
        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        # Compute probabilities for gate scaling and auxillary loss.
        probs = torch.softmax(logits / self.router_temperature, dim=-1)  # (B, N, E)

        # TODO: Add a better comment here explaining the two routing modes.
        if self.router_mode == "soft":
            gates = probs                                              # (B, N, E)
            _, topk_idx = probs.topk(self.router_top_k, dim=-1)        # (B, N, K)
            dispatch = probs.new_zeros(probs.shape, dtype=torch.bool)  # (B, N, E)
            dispatch.scatter_(-1, topk_idx, True)                      # (B, N, E)
        elif self.router_mode == "ste":
            if self.training:
                gates = F.gumbel_softmax(logits, tau=self.router_temperature, hard=True, dim=-1)   # (B, N, E)
            else:
                gates = F.one_hot(torch.argmax(logits, dim=-1), self.num_experts).to(probs.dtype)  # (B, N, E)
            dispatch = gates > 0                                                                   # (B, N, E)
        else:
            raise RuntimeError("Unreachable.")

        # Compute load balancing auxillary loss, which encourages even expert usage.
        assign = dispatch.to(probs.dtype)  # (B, N, E)
        f = assign.mean(dim=(0, 1))  # (E,)
        p = probs.mean(dim=(0, 1))   # (E,)
        aux = self.num_experts * (f * p).sum()

        # Logging.
        self.last_aux_loss = aux
        self.last_entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean().detach()
        self.last_usage = dispatch.float().mean(dim=(0, 1)).detach()

        return dispatch, gates, aux

    def forward_embeddings(self, z: Tensor) -> Tensor:

        B, T, E = z.shape
        P, N = self.resolve_patch_dims(T)
        C = self.out_channels

        # Prepare patches.
        z = PatchEncoderBase.pad_to_length(z, N * P)  # (B, NP, E)
        z = z.view(B, N, P, E)                        # (B,  N, P, E)
        z = z.reshape(B * N, P, E)                    # (BN, P, E)
        z = z.permute(0, 2, 1)                        # (BN, E, P)

        # Probe features.
        r: Tensor = self.probe(z)         # (BN, D, S_probe)
        r = r.max(dim=-1)[0]              # (BN, D)
        r = r.view(B, N, self.probe_dim)  # (B, N, D)

        # Route to experts.
        dispatch, gates, aux = self._route(r)

        dispatch_flat = dispatch.view(B * N, self.num_experts)    # (BN, E)
        gates_flat    = gates.view(B * N, self.num_experts)       # (BN, E)

        # Expert passes.
        out = z.new_zeros((B * N, C))
        g_e: Tensor
        for e, expert in enumerate(self.experts):
            # Gather the patches for expert e.
            idx = dispatch_flat[:, e].nonzero(as_tuple=False).squeeze(-1)  # (n_e,)
            if idx.numel() == 0:
                continue
            z_e = z.index_select(0, idx)                            # (n_e, E, P)
            # Run expert and max-pool over time to get one vector per patch.
            g_e = expert(z_e)                                       # (n_e, C, S)
            g_e, _ = g_e.max(dim=-1)                                # (n_e, C)
            # Scale expert output by its gate and accumulate into the output.
            w = gates_flat.index_select(0, idx)[:, e].unsqueeze(-1).to(dtype=g_e.dtype)  # (n_e, 1)
            g_e = g_e * w
            out.index_add_(0, idx, g_e)

        # Reshape and return.
        out = out.view(B, N, C)  # (B, N, C)

        # Ensure every expert parameter participates in the autograd graph.
        out = self._ddp_keepalive(out)

        return out

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:

        B, T = ts[0].shape[0], ts[0].shape[1]
        E, C = self.in_channels, self.out_channels
        P, N = self.resolve_patch_dims(T)

        patch_batch_size = self.patch_batch_size if self.patch_batch_size is not None else B

        g_all: Tensor

        # Probe features.
        max_vals, pos = _lowmem_patchwise_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.probe_kernel_size,
            first_stride=self.probe_stride,
            chunk_size=self.chunk_size,
            overlap=self.probe_overlap,
            channels=self.probe_dim,
            num_patches=self.num_patches,
            activations_fn=partial(functional_forward, module=self.probe, fp32=self.fp32),
        )
        if not self.training:
            r = max_vals
            del max_vals
            del pos
        else:
            positions_flat = pos.view(B, N * self.probe_dim)
            wins_cat, meta = _gather_wins_via_preprocess_batched(
                preprocess=preprocess,
                ts=ts,
                positions=positions_flat,
                rf=self.probe_kernel_size,
                embedding_dim=E,
                mask=None,
            )
            g_all = self.probe(wins_cat)
            g_all = g_all.squeeze(-1)
            r = _scatter_g_to_BNC(
                g_all=g_all,
                meta=meta,
                batch_size=B,
                num_patches=self.num_patches,
                channels=self.probe_dim,
            )
            del max_vals
            del pos
            del positions_flat
            del wins_cat
            del meta
            del g_all

        # Route to experts.
        dispatch, gates, aux = self._route(r)

        # Pre-compute the experts' involvement once, to reduce CUDA synchronization in the expert loop.
        should_dispatch = dispatch.reshape(-1, dispatch.shape[-1]).any(dim=0).tolist()

        # Expert passes.
        out = r.new_zeros((B, N, C))
        for e, expert in enumerate(self.experts):
            active = dispatch[..., e]                                      # (B, N)
            mask = active.unsqueeze(-1).expand(B, N, C).reshape(B, N * C)  # (B, NC)
            if not should_dispatch[e]:
                continue

            max_vals, pos = _lowmem_patchwise_max_over_time_dispatched(
                preprocess=preprocess,
                ts=ts,
                rf=self.kernel_size,
                stride=self.stride,
                patch_size=P,
                num_patches=self.num_patches,
                channels=C,
                activations_fn=partial(functional_forward, module=expert, fp32=self.fp32),
                patch_active=active,
                patch_batch_size=patch_batch_size,
            )

            if not self.training:
                max_vals = self._clean_max_vals(max_vals, active, e)
                w = gates[..., e].unsqueeze(-1).to(dtype=max_vals.dtype)
                out = out + max_vals * w
                continue

            positions_flat = pos.view(B, N * C)
            wins_cat, meta = _gather_wins_via_preprocess_batched(
                preprocess=preprocess,
                ts=ts,
                positions=positions_flat,
                rf=self.kernel_size,
                embedding_dim=E,
                mask=mask,
            )

            g_all = expert(wins_cat)
            g_all = g_all.squeeze(-1)

            z = _scatter_g_to_BNC(
                g_all=g_all,
                meta=meta,
                batch_size=B,
                num_patches=self.num_patches,
                channels=C,
            )

            # Gate factor.
            w = gates[..., e].unsqueeze(-1).to(dtype=z.dtype)
            out = out + z * w

        # Ensure every expert parameter participates in the autograd graph.
        out = self._ddp_keepalive(out)

        return out

    def _clean_max_vals(self, max_vals: Tensor, active: Tensor, e: int) -> Tensor:
        # Find any inactive patch that has a nonzero value in any channel.
        # This should never happen and indicates a bug in the low-memory max computation.
        inactive = ~active
        if inactive.any():
            bad = (max_vals.abs() > 0).any(dim=-1) & inactive
            if bad.any():
                num_bad = int(bad.sum().item())
                max_leak = float(max_vals[inactive].abs().max().item())
                warnings.warn(f"Non-zero inactive patches ({num_bad}) with maximal leakage of {max_leak} detected in expert {e}. Zeroing them out.")
            max_vals = max_vals * active.unsqueeze(-1).to(max_vals.dtype)
        return max_vals

    def _ddp_keepalive(self, out: torch.Tensor) -> torch.Tensor:
        """
        Ensure all expert parameters participate in the autograd graph every step.
        This avoids DDP 'unused parameter' errors when routing skips experts.
        """
        warnings.warn(f"{self.__class__.__name__}::_ddp_keepalive uses non-vectorized operations and should be re-implemented with the optimized _ddp_keepalive function.")
        if (not self.training) or (not torch.is_grad_enabled()):
            return out

        dummy = out.sum() * 0.0  # scalar connected to the graph
        for expert in self.experts:
            if not isinstance(expert, nn.Conv1d):
                raise NotImplementedError("Only Conv1d experts are supported for DDP keepalive.")
            dummy = dummy + expert.weight.view(-1)[0] * 0.0
            if expert.bias is not None:
                dummy = dummy + expert.bias.view(-1)[0] * 0.0
        return out + dummy


class PatchPositionalityEncoder(nn.Module):
    """
    Injects approximate byte-level positional information into patch embeddings.

    If `max_length` is provided, both relative and absolute positional information
    is used. If not, only relative positional information is used.
    """

    def __init__(self, in_channels: int, max_length: Optional[int] = None, hidden_size: int = 64) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"`hidden_size` must be positive. Got {hidden_size}.")

        self.in_channels = in_channels
        self.max_length = max_length
        self.num_features = 2 if max_length is not None else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_channels),
        )

    def forward(self, z: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            z: input tensor of shape (B, N, C).
            lengths: input lengths of shape (B,).

        Returns:
            output tensor of shape (B, N, C).
        """
        check_tensor(z, (None, None, self.in_channels), FLOATS)

        B, N, C = z.shape

        if self.max_length is not None:
            if lengths is None:
                raise ValueError(f"`lengths` must be provided when `max_length` is set to {self.max_length}.")
            check_tensor(lengths, (B,), INTEGERS)

        # Relative center of each patch.
        patch_idx = torch.arange(N, device=z.device, dtype=z.dtype)  # (N,)
        rel_center = (patch_idx + 0.5) / N                           # (N,)
        rel_center = rel_center.unsqueeze(0).expand(B, N)            # (B, N)

        # Absolute center of each patch.
        if self.max_length is not None:
            assert lengths is not None
            lengths = lengths.to(z.dtype).clamp(min=1.0)
            center_bytes = rel_center * lengths.unsqueeze(1)  # (B, N)
            center_bytes = (center_bytes / float(self.max_length)).clamp(0.0, 1.0)

        # Combine features and produce bias.
        features_ = [rel_center]
        if self.max_length is not None:
            features_.append(center_bytes)
        features = torch.stack(features_, dim=-1)  # (B, N, F)
        bias = self.mlp(features.to(z.dtype))      # (B, N, C)

        z = z + bias                               # (B, N, C)

        check_tensor(z, (B, N, self.in_channels), FLOATS)

        return z

# -------------------------------------------------------------------------------- #
# ViT
# -------------------------------------------------------------------------------- #

class ViT(nn.Module):
    """
    Vision Transformer.

    See: Dosovitskiy "An image is worth 16x16 words: Transformers for image recognition at scale" ICLR 2021.
    """

    cls_token: Optional[nn.Parameter]

    def __init__(
        self,
        embedding_dim: int = 8,
        d_model: int = 256,
        nhead: int = 1,
        dim_feedforward: int = -1,
        num_layers: int = 1,
        posencoder: Literal["fixed", "learned", "none"] = "learned",
        max_len: Optional[int] = None,
        activation: str = "gelu",
        pooling: Literal["mean", "cls"] = "cls",  # NOTE: we use `avg` pooling elsewhere, so this might be confusing...
        seq_lengths: Optional[tuple[int, ...]] = None,
        batch_sizes: Optional[tuple[int, ...]] = None,
    ) -> None:
        super().__init__()

        dim_feedforward = 4 * d_model if dim_feedforward == -1 else dim_feedforward

        self.proj = nn.Linear(embedding_dim, d_model)
        self.posencoder: nn.Identity | SinusoidalPositionalEncoding | LearnedPositionalEncoding
        if posencoder == "fixed":
            self.posencoder = SinusoidalPositionalEncoding(d_model)
        elif posencoder == "learned":
            if max_len is None:
                raise ValueError("`max_len` must be specified when using learned positional encoding.")
            self.posencoder = LearnedPositionalEncoding(d_model, max_len + (1 if pooling == "cls" else 0))
        elif posencoder == "none":
            self.posencoder = nn.Identity()
        else:
            raise ValueError(f"Unknown positional encoding type: {posencoder}. Supported types are 'fixed', 'learned', and 'none'.")
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation=ACTVS[activation], batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=layer.norm_first is False)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        elif pooling == "mean":
            self.cls_token = None
        else:
            raise ValueError(f"Unknown pooling method: {pooling}. Supported methods are 'mean' and 'cls'.")

        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.pooling = pooling

        # Attributes to fix tensor shapes for compilation.
        self.seq_lengths = seq_lengths
        if seq_lengths is not None:
            if len(seq_lengths) == 0:
                raise ValueError("If `seq_lengths` is provided, it must be a non-empty tuple of integers.")
            if max_len is None:
                raise ValueError("If `seq_lengths` is provided, `max_len` must also be provided.")
            if any(l > max_len or l < 1 for l in seq_lengths):
                raise ValueError(f"All `seq_lengths` must be postiive integers less than or equal to `max_len` ({max_len}).")
            if max(seq_lengths) != max_len:
                raise ValueError(f"The maximum value in `seq_lengths` ({max(seq_lengths)}) must equal `max_len` ({max_len}).")
            if self.pooling == "cls":
                if any((l + 1) % 8 != 0 for l in seq_lengths or []):
                    warnings.warn(
                        "For performance, we recommend all tensor dimensions be multiples of 8. "
                        "When using `pooling='cls'`, a [CLS] token is prepended to the sequence. "
                        "So we recommend all `seq_lengths` be one less than a multiple of 8."
                        f"The provided `seq_lengths` are: {seq_lengths}."
                    )
            elif self.pooling == "mean":
                if any((l % 8) != 0 for l in seq_lengths or []):
                    warnings.warn(
                        "For performance, we recommend all tensor dimensions be multiples of 8. "
                        "So we recommend all `seq_lengths` be a multiple of 8."
                        f"The provided `seq_lengths` are: {seq_lengths}."
                    )

        self.batch_sizes = batch_sizes
        if batch_sizes is not None:
            if len(batch_sizes) == 0:
                raise ValueError("If `batch_sizes` is provided, it must be a non-empty tuple of integers.")
            if any(b <= 0 for b in batch_sizes):
                raise ValueError("All `batch_sizes` must be positive integers.")
            if any(b % 8 != 0 for b in batch_sizes):
                warnings.warn(
                    "For performance, we recommend all tensor dimensions be multiples of 8. "
                    "So we recommend all `batch_sizes` be a multiple of 8."
                    f"The provided `batch_sizes` are: {batch_sizes}."
                )

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, T, E).
            key_padding_mask: Optional mask of shape (B, T) indicating padding positions.
        Returns:
            z: Hidden representation of shape (B, D).
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if key_padding_mask is not None:
            check_tensor(key_padding_mask, (x.shape[0], x.shape[1]), torch.bool)

        z: Tensor = x

        B = x.shape[0]
        T = z.shape[1]

        # Possibly pad along the sequence dimension to a fixed size.
        T_pad = T
        if self.seq_lengths is not None:
            if all(l < T for l in self.seq_lengths):
                raise ValueError(f"Input sequence length {T} is larger than all configured sequence lengths {self.seq_lengths}.")
            T_pad = min(l for l in self.seq_lengths if l >= T)
            z = F.pad(z, (0, 0, 0, T_pad - T))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, T_pad - T), value=True)
            else:
                key_padding_mask = F.pad(torch.zeros((B, T), dtype=torch.bool, device=z.device), (0, T_pad - T), value=True)

        # Possibly pad along the batch dimension to a fixed size.
        B_pad = B
        if self.batch_sizes is not None:
            if all(b < B for b in self.batch_sizes):
                raise ValueError(f"Input batch size {B} is larger than all configured batch sizes {self.batch_sizes}.")
            B_pad = min(b for b in self.batch_sizes if b >= B)
            z = F.pad(z, (0, 0, 0, 0, 0, B_pad - B))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 0, 0, B_pad - B), value=True)

        # Project to model dimension.
        z = self.proj(z)                                   # (B, T, D)

        # Add cls token if needed.
        if self.pooling == "cls":
            assert self.cls_token is not None
            t = self.cls_token.expand(z.shape[0], -1, -1)  # (B, 1, D)
            z = torch.cat((t.to(z.dtype), z), dim=1)       # (B, T, D)
            if key_padding_mask is not None:
                t = torch.zeros((key_padding_mask.shape[0], 1), dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat((t, key_padding_mask), dim=1)  # (B, T+1)
            # T := T + 1

        z = self.posencoder(z)                                          # (B, T, D)
        z = z.contiguous()
        z = self.transformer(z, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        # Pooling (B, T, D) -> (B, D)
        if self.pooling == "cls":
            z = z[:, 0, :]
        else:
            if key_padding_mask is None:
                z = z.mean(dim=1)
            else:
                valid = (~key_padding_mask).to(z.dtype)
                denom = valid.sum(dim=1).clamp_min(1.0)
                z = (z * valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

        # Remove padding if added.
        z = z[:B, :]  # (B, D)

        check_tensor(z, (x.shape[0], self.d_model), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(0, self.transformer.num_layers):
            fully_shard(self.transformer.layers[i], **kwds)
        fully_shard(self, **kwds)

# -------------------------------------------------------------------------------- #
# MalConv
# -------------------------------------------------------------------------------- #


def _check_grad_connected(outputs: Tensor, inputs: Tensor | list[Tensor]) -> list[bool]:
    """
    Check if a tensor is connected to the gradients of the convolutional weights.
    """
    if not torch.is_grad_enabled():
        raise RuntimeError("Gradients are not enabled.")
    inputs = [inputs] if isinstance(inputs, Tensor) else inputs
    if not outputs.requires_grad:
        return [False for _ in inputs]
    grads = torch.autograd.grad(outputs, inputs, retain_graph=True, allow_unused=True)
    return [grad is not None for grad in grads]


class GatedConvolution(nn.Module):
    """
    Gated Convolution.

    See: Dauphin "Language modeling with gated convolutional networks" ICML 2017.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv_2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Input tensor of shape (B, E, T).
        Returns:
            Output tensor of shape (B, C, S).
        """
        check_tensor(z, (None, self.in_channels, None), FLOATS)
        c_1: Tensor = self.conv_1(z) # (B, C, S - 1)
        c_2: Tensor = self.conv_2(z) # (B, C, S - 1)
        g = c_1 * self.sigmoid(c_2)  # (B, C, S - 1)
        check_tensor(g, (z.shape[0], self.out_channels, None), FLOATS)
        return g

    @torch.no_grad()
    def forward_functional(self, z: Tensor, fp32: bool = False) -> Tensor:
        """
        Functional version of `GatedConvolution.forward` without auto-grad.

        It is unclear whether or not using `torch.no_grad()` at the top-level
        definition is strictly nessecary or whether using `torch.no_grad()` inside
        the function as a context manager (with boolean on/off control) would suffice.
        Since we only this this function for the low-memory computation when we need
        it detached from auto-grad, the former is acceptable for now.
        """
        check_tensor(z, (None, self.conv_1.in_channels, None), FLOATS)

        w1 = self.conv_1.weight.detach()
        b1 = self.conv_1.bias.detach() if self.conv_1.bias is not None else None
        w2 = self.conv_2.weight.detach()
        b2 = self.conv_2.bias.detach() if self.conv_2.bias is not None else None

        if fp32:
            z = z.to(torch.float32)
            if any(w_b.dtype != torch.float32 for w_b in [w1, w2, b1, b2] if w_b is not None):
                warnings.warn("Some weights and/or biases are not in float32, which is unexpected, when `fp32=True`.")

        c_1 = F.conv1d(z, w1, b1, **_get_conv_kwds(self.conv_1))
        c_2 = F.conv1d(z, w2, b2, **_get_conv_kwds(self.conv_2))
        g = c_1 * torch.sigmoid(c_2)

        check_tensor(g, (z.shape[0], self.conv_1.out_channels, None), FLOATS)
        return g

    def check_grad_connected(self, x: Tensor) -> tuple[bool, bool]:
        return tuple(_check_grad_connected(x, [self.conv_1.weight, self.conv_2.weight]))  # type: ignore[return-value]


class MalConvBase(nn.Module, ABC):
    """
    Base class of MalConv backbones.

    See: Raff "Malware detection by eating a whole EXE." AICS 2018.
    See: Raff "Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection" AAAI 2021.
    """

    embedding_dim: int
    channels: int
    kernel_size: int
    stride: int
    chunk_size: int
    overlap: int
    fp32: bool

    def __init__(
        self,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        chunk_size: int = 2 ** 16,
        overlap: Optional[int] = None,
        fp32: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = self.kernel_size // 2 if overlap is None else overlap
        self.fp32 = fp32

        if self.overlap < self.kernel_size / 2:
            warnings.warn(f"Overlap {self.overlap} is less than half the kernel size {self.kernel_size}. Pooling may be impacted windowing issues.")

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def forward(self, z: Optional[Tensor] = None, preprocess: Optional[PreprocessFn] = None, ts: Optional[Sequence[Tensor]] = None) -> Tensor:
        """
        Args:
            z: input tensor of shape (B, T, E). If provided, will invoke the `forward_embeddings` method.
            preprocess: callable to compute the input tensor. If provided, will invoke the `forward_streaming` method.
            ts: arguments to `preprocess`, each of shape (B, T, *). If provided, will invoke the `forward_streaming` method.

        Returns:
            (B, C) hidden representation.
        """
        if z is not None:
            check_tensor(z, (None, None, self.embedding_dim), FLOATS)
            if z.shape[1] < self.min_length:
                raise RuntimeError(f"Input sequence length {z.shape[1]} is less than the minimum required length {self.min_length}.")
            z = self.forward_embeddings(z)

        elif preprocess is not None and ts is not None:
            for t in ts:
                if t.shape[0] != ts[0].shape[0]:
                    raise TensorError(t, (ts[0].shape[0], ts[0].shape[1], None), None)
                if t.shape[1] != ts[0].shape[1]:
                    raise TensorError(t, (ts[0].shape[0], ts[0].shape[1], None), None)
                if t.shape[1] < self.min_length:
                    raise RuntimeError(f"Input sequence length {t.shape[1]} is less than the minimum required length {self.min_length}.")
            if DISABLE_LOW_MEMORY_PATHS:
                z = self.forward_embeddings(preprocess(*ts))
            else:
                z = self.forward_streaming(preprocess=preprocess, ts=ts)

        else:
            raise ValueError("Either `z` or both `preprocess` and `ts` must be provided.")

        check_tensor(z, (z.shape[0], self.channels), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard(self, **kwds)

    @abstractmethod
    def forward_embeddings(self, z: Tensor) -> Tensor:
        ...

    @abstractmethod
    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        ...


class MalConv(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.gconv = GatedConvolution(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward_embeddings(self, z: Tensor) -> Tensor:
        z = z.transpose(1, 2)        # (B, E, T)
        g = self.gconv(z)            # (B, C, S - 1)
        z = self.pool(g)             # (B, C, 1)
        z = z.squeeze(-1)            # (B, C)
        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        z = preprocess(*ts)                # (B, T, E)
        z = self.forward_embeddings(z)     # (B, C)
        return z


class MalConvLowMem(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.gconv = GatedConvolution(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward_embeddings(self, z: Tensor) -> Tensor:
        z = z.transpose(1, 2)        # (B, E, T)
        g = self.gconv(z)            # (B, C, S - 1)
        z = self.pool(g)             # (B, C, 1)
        z = z.squeeze(-1)            # (B, C)
        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:

        # Gather the max-pooled activations across the temporal dimension streaming without autograd.
        # It is critical here that we pass a separate, purely functional `activations_fn` with autograd disabled.
        # If autograd is enabled externally, i.e., in the training loop, naively passing in `self.gconv`
        # will cause PyTorch to severe the gradient flow from `self.gconv` weights and they will not be updated
        # during training. Its unclear why exactly this happens, but the solution is straightforward enough.
        max_vals, pos = _lowmem_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.kernel_size,
            first_stride=self.stride,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            channels=self.channels,
            activations_fn=partial(self.gconv.forward_functional, fp32=self.fp32),
            # activations_fn=self.gconv,  # WARN: this causes silent gradient disconnection!
        )
        assert max_vals.requires_grad is False
        assert not torch.is_grad_enabled() or not any(self.gconv.check_grad_connected(max_vals.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "gconv" in name)

        # If conducting inference, simply return the max-pooled activations.
        if not self.training:
            return max_vals

        # If conducting training, recompute only the winner windows (with autograd).
        wins_cat, meta = _gather_wins_via_preprocess_batched(
            preprocess=preprocess,
            ts=ts,
            positions=pos,
            rf=self.kernel_size,
            embedding_dim=self.embedding_dim,
        )
        assert wins_cat.requires_grad is True
        assert not any(self.gconv.check_grad_connected(wins_cat.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "gconv" in name)

        # Run the winner windows through the convolutional layers (with autograd).
        g_all: Tensor = self.gconv(wins_cat)
        g_all = g_all.squeeze(-1)
        assert g_all.requires_grad is True
        assert all(self.gconv.check_grad_connected(g_all.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "gconv" in name)

        # Scatter the per-window activations back to per-batch activations.
        z = _scatter_g_to_BC(
            g_all=g_all,
            meta=meta,
            batch_size=ts[0].shape[0],
            channels=self.channels,
        )
        assert z.requires_grad is True
        assert all(self.gconv.check_grad_connected(z.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "gconv" in name)

        return z


class GCGBlock(nn.Module):
    """
    Gated Convolution with Global Context (GCG) block.
    """

    def __init__(self, embedding_dim: int, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_1 = nn.Conv1d(embedding_dim, 2 * channels, kernel_size, stride)
        self.conv_2 = nn.Conv1d(channels, channels, 1)
        self.gct_proj = nn.Linear(channels, channels)

    def forward(self, z: Tensor, gct: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, E, T).
            gct: Global context tensor of shape (B, C).
        Returns:
            Output tensor of shape (B, C, S).
        """
        B, T = z.shape[0], z.shape[2]
        E, C = self.embedding_dim, self.channels
        check_tensor(z, (B, E, T), FLOATS)
        check_tensor(gct, (B, C), FLOATS)

        # TODO: comment
        h: Tensor
        h = self.conv_1(z)                  # (B, 2C, T')
        h = F.glu(h, dim=1)                 # (B, C,  T')
        h = self.conv_2(h)                  # (B, C,  T')
        h = F.leaky_relu(h)                 # (B, C,  T')
        h = h.transpose(1, 2).contiguous()  # (B, T', C)
        T_ = h.shape[1]

        # Project global context.
        q = self.gct_proj(gct)  # (B, C)
        q = torch.tanh(q)       # (B, C)
        q = q.unsqueeze(2)      # (B, C, 1)

        # TODO: comment
        t = h.reshape(1, B * C, T_)   # (1, BC, T')
        g = F.conv1d(t, q, groups=B)  # (1, B, T')
        g = g.view(B, 1, T_)          # (B, 1, T')
        g = torch.sigmoid(g)          # (B, 1, T')

        # TODO: comment
        out = (h * g.transpose(1, 2)).contiguous()  # (B, T', C)
        out = out.transpose(1, 2).contiguous()      # (B, C, T')

        check_tensor(out, (B, C, T_), FLOATS)
        return out

    @torch.no_grad()
    def forward_functional(self, z: Tensor, gct: Tensor, fp32: bool = False) -> Tensor:
        """
        Functional version for low-memory streaming scans.
        """
        B, T = z.shape[0], z.shape[2]
        E, C = self.embedding_dim, self.channels
        check_tensor(z, (B, E, T), FLOATS)
        check_tensor(gct, (B, C), FLOATS)

        w1 = self.conv_1.weight.detach()
        b1 = self.conv_1.bias.detach() if self.conv_1.bias is not None else None
        w2 = self.conv_2.weight.detach()
        b2 = self.conv_2.bias.detach() if self.conv_2.bias is not None else None
        wp = self.gct_proj.weight.detach()
        bp = self.gct_proj.bias.detach() if self.gct_proj.bias is not None else None

        if fp32:
            z = z.to(torch.float32)
            gct = gct.to(torch.float32)
            if any(w_b.dtype != torch.float32 for w_b in [w1, w2, wp, b1, b2, bp] if w_b is not None):
                warnings.warn("Some weights and/or biases are not in float32, which is unexpected, when `fp32=True`.")

        h = F.conv1d(z, w1, b1, **_get_conv_kwds(self.conv_1))
        h = F.glu(h, dim=1)
        h = F.conv1d(h, w2, b2, **_get_conv_kwds(self.conv_2))
        h = F.leaky_relu(h)
        h = h.transpose(1, 2).contiguous()
        T_ = h.shape[1]

        q = F.linear(gct, wp, bp)
        q = torch.tanh(q)
        q = q.unsqueeze(2)

        t = h.reshape(1, B * C, T_)
        gates = F.conv1d(t, q, groups=B)
        gates = gates.view(B, 1, T_)
        gates = torch.sigmoid(gates)

        out = (h * gates.transpose(1, 2)).contiguous()
        out = out.transpose(1, 2).contiguous()

        check_tensor(out, (B, C, T_), FLOATS)
        return out

    def check_grad_connected(self, x: Tensor) -> tuple[bool, bool, bool]:
        return tuple(_check_grad_connected(x, [self.conv_1.weight, self.conv_2.weight, self.gct_proj.weight]))  # type: ignore[return-value]


class ContextBlock(nn.Module):
    """
    Context subnetwork for MalConvGCG.
    """

    def __init__(self, embedding_dim: int, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(embedding_dim, 2 * channels, kernel_size, stride)
        self.share = nn.Conv1d(channels, channels, 1)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: input tensor of shape (B, E, T).
        Returns:
            Output tensor of shape (B, C, T').
        """
        check_tensor(z, (None, self.embedding_dim, None), FLOATS)
        x = self.conv(z)     # (B, 2C, T')
        x = F.glu(x, dim=1)  # (B, C,  T')
        x = self.share(x)    # (B, C,  T')
        x = F.leaky_relu(x)  # (B, C,  T')
        check_tensor(x, (z.shape[0], self.channels, None), FLOATS)
        return x

    @torch.no_grad()
    def forward_functional(self, z: Tensor, fp32: bool = False) -> Tensor:
        """
        Functional version for low-memory streaming scans.
        """
        check_tensor(z, (None, self.embedding_dim, None), FLOATS)

        w_ctx = self.conv.weight.detach()
        b_ctx = self.conv.bias.detach() if self.conv.bias is not None else None
        w_sh  = self.share.weight.detach()
        b_sh  = self.share.bias.detach() if self.share.bias is not None else None

        if fp32:
            z = z.to(torch.float32)
            if any(w_b.dtype != torch.float32 for w_b in [w_ctx, w_sh, b_ctx, b_sh] if w_b is not None):
                warnings.warn("Some weights and/or biases are not in float32, which is unexpected, when `fp32=True`.")

        x = F.conv1d(z, w_ctx, b_ctx, **_get_conv_kwds(self.conv))
        x = F.glu(x, dim=1)
        x = F.conv1d(x, w_sh, b_sh, **_get_conv_kwds(self.share))
        x = F.leaky_relu(x)

        check_tensor(x, (z.shape[0], self.channels, None), FLOATS)
        return x

    def check_grad_connected(self, x: Tensor) -> tuple[bool, bool]:
        return tuple(_check_grad_connected(x, [self.conv.weight, self.share.weight]))  # type: ignore[return-value]


class MalConvGCG(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.ctx_block  = ContextBlock(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.main_block = GCGBlock(self.embedding_dim, self.channels, self.kernel_size, self.stride)

    def forward_embeddings(self, z: Tensor) -> Tensor:
        z = z.transpose(1, 2).contiguous()  # (B, E, T)
        ctx_map = self.ctx_block(z)         # (B, C, T')
        gct = ctx_map.max(dim=-1)[0]        # (B, C)
        main_map = self.main_block(z, gct)  # (B, C, T')
        z = main_map.max(dim=-1)[0]         # (B, C)
        return z

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        # The steps taken here are quite similar to those in MalConvLowMem.forward_streaming,
        # but with two separate streaming scans: one for the context path and one for the main path.
        # See MalConvLowMem.forward_streaming for comment about what is going on.

        batch_size = ts[0].shape[0]

        # ------------------------------------------------------------------
        # -------------------------- Context Path --------------------------
        # ------------------------------------------------------------------

        # Get the streamed maxima of the context path.
        ctx_max_vals, ctx_pos = _lowmem_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.kernel_size,
            first_stride=self.stride,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            channels=self.channels,
            activations_fn=partial(self.ctx_block.forward_functional, fp32=self.fp32),
        )
        assert ctx_max_vals.requires_grad is False
        assert not torch.is_grad_enabled() or not any(self.ctx_block.check_grad_connected(ctx_max_vals.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "ctx_block" in name)

        # Get the global context tensor, gct, from the context path.
        gct: Tensor  # (B, C)
        if not self.training:
            gct = ctx_max_vals
        else:
            wins_ctx, meta_ctx = _gather_wins_via_preprocess_batched(
                preprocess=preprocess,
                ts=ts,
                positions=ctx_pos,
                rf=self.kernel_size,
                embedding_dim=self.embedding_dim,
            )
            assert wins_ctx.requires_grad is True
            assert not torch.is_grad_enabled() or not any(self.ctx_block.check_grad_connected(wins_ctx.sum()))
            assert all(p.requires_grad for name, p in self.named_parameters() if "ctx_block" in name)

            g_all_ctx: Tensor = self.ctx_block(wins_ctx)
            g_all_ctx = g_all_ctx.squeeze(-1)
            assert g_all_ctx.requires_grad is True
            assert not torch.is_grad_enabled() or all(self.ctx_block.check_grad_connected(g_all_ctx.sum()))
            assert all(p.requires_grad for name, p in self.named_parameters() if "ctx_block" in name)

            gct = _scatter_g_to_BC(
                g_all=g_all_ctx,
                meta=meta_ctx,
                batch_size=batch_size,
                channels=self.channels,
            )
            assert gct.requires_grad is True
            assert not torch.is_grad_enabled() or all(self.ctx_block.check_grad_connected(gct.sum()))
            assert all(p.requires_grad for name, p in self.named_parameters() if "ctx_block" in name)

        # ------------------------------------------------------------------
        # --------------------------- Main Path ----------------------------
        # ------------------------------------------------------------------

        # Get the streamed maxima of the main path.
        main_max_vals, main_pos = _lowmem_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.kernel_size,
            first_stride=self.stride,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            channels=self.channels,
            activations_fn=partial(self.main_block.forward_functional, gct=gct.detach(), fp32=self.fp32),
        )
        assert main_max_vals.requires_grad is False
        assert not torch.is_grad_enabled() or not any(self.main_block.check_grad_connected(main_max_vals.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "main_block" in name)

        # If conducting inference, simply return the max-pooled activations.
        if not self.training:
            return main_max_vals

        # If conducting training, recompute only the winner windows (with autograd).
        if not torch.is_grad_enabled():
            raise RuntimeError("Gradients are not enabled but the model is in training mode.")
        wins_main, meta_main = _gather_wins_via_preprocess_batched(
            preprocess=preprocess,
            ts=ts,
            positions=main_pos,
            rf=self.kernel_size,
            embedding_dim=self.embedding_dim,
        )
        assert wins_main.requires_grad is True
        assert not any(self.main_block.check_grad_connected(wins_main.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "main_block" in name)

        # Build per-window context aligned with wins_main.
        ctx_per_win: list[Tensor] = []
        for (_, _, u_b), b in zip(meta_main, range(batch_size)):
            if u_b == 0:
                continue
            ctx_per_win.append(gct[b].unsqueeze(0).expand(u_b, -1))
        if ctx_per_win:
            ctx_cat = torch.cat(ctx_per_win, dim=0)
        else:
            ctx_cat = gct.new_empty((0, self.channels))
        assert ctx_cat.shape[0] == wins_main.shape[0]

        # Main-path recomputation with full autograd graph.
        g_all_main: Tensor = self.main_block(wins_main, ctx_cat)
        g_all_main = g_all_main.squeeze(-1)
        assert g_all_main.requires_grad is True
        assert all(self.main_block.check_grad_connected(g_all_main.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "main_block" in name)

        # Scatter per-window activations back to per-(B,C) features.
        z = _scatter_g_to_BC(
            g_all=g_all_main,
            meta=meta_main,
            batch_size=batch_size,
            channels=self.channels,
        )
        assert z.requires_grad is True
        assert all(self.ctx_block.check_grad_connected(z.sum()))
        assert all(self.main_block.check_grad_connected(z.sum()))
        assert all(p.requires_grad for name, p in self.named_parameters() if "main_block" in name)

        return z

# -------------------------------------------------------------------------------- #
# Classifiers
# -------------------------------------------------------------------------------- #

class Classifier(nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T).
            g: FiLM conditioning vector of shape (B, T, G).
        Returns:
            z: Classification logits of shape (B, M).
        """
        ...

    @abstractmethod
    def fully_shard(self, **kwds: Any) -> None:
        ...

    def _check_forward_inputs(self, x: Tensor, g: Optional[Tensor]) -> None:
        check_tensor(x, (None, None), INTEGERS)
        if g is not None:
            check_tensor(g, (x.shape[0], x.shape[1], None), FLOATS)


class MalConvClassifier(Classifier):

    def __init__(self, embedding: nn.Embedding, filmer: FiLM | FiLMNoP, backbone: MalConvBase, head: ClassifificationHead) -> None:
        super().__init__()

        self.embedding = embedding
        self.filmer = filmer
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        self._check_forward_inputs(x, g)

        def preprocess(x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embedding(x)  # (B, T, E)
            z = self.filmer(z, g) if torch.is_grad_enabled() else self.filmer.forward_functional(z, g)  # type: ignore[arg-type]
            return z

        ts = (x, g) if g is not None else (x,)

        z = self.backbone(preprocess=preprocess, ts=ts)  # (B, D)
        z = self.head(z)                                 # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})


class ViTClassifier(Classifier):

    def __init__(self, embedding: nn.Embedding, filmer: FiLM | FiLMNoP, patcher: PatchEncoderBase, norm: Optional[nn.LayerNorm], patchposencoder: PatchPositionalityEncoder | Identity, backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__()

        self.embedding = embedding
        self.filmer = filmer
        self.patcher = patcher
        self.norm = norm if norm is not None else nn.LayerNorm(patcher.out_channels)
        self.patchposencoder = patchposencoder
        self.backbone = backbone
        self.head = head

        self.should_get_lengths = isinstance(self.patchposencoder, PatchPositionalityEncoder) and self.patchposencoder.max_length is not None

    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        self._check_forward_inputs(x, g)

        def preprocess(x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embedding(x)  # (B, T, E)
            z = self.filmer(z, g) if torch.is_grad_enabled() else self.filmer.forward_functional(z, g)  # type: ignore[arg-type]
            return z

        lengths = None
        if self.should_get_lengths:
            lengths = (x != 0).sum(dim=1)  # (B,)

        ts = (x, g) if g is not None else (x,)
        z = self.patcher(preprocess=preprocess, ts=ts)  # (B, N, C)
        z = self.norm(z)                                # (B, N, C)
        z = self.patchposencoder(z, lengths)            # (B, N, C)
        z = self.backbone(z)                            # (B, D)
        z = self.head(z)                                # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer, self.patcher], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

    @property
    def last_aux_loss(self) -> Optional[Tensor]:
        if isinstance(self.patcher, PatchEncoderLowMemSwitchMoE):
            return self.patcher.last_aux_loss
        return None

    @property
    def last_usage(self) -> Optional[Tensor]:
        if isinstance(self.patcher, PatchEncoderLowMemSwitchMoE):
            return self.patcher.last_usage
        return None

    @property
    def last_entropy(self) -> Optional[Tensor]:
        if isinstance(self.patcher, PatchEncoderLowMemSwitchMoE):
            return self.patcher.last_entropy
        return None

# -------------------------------------------------------------------------------- #
# Hierarchical Classifiers
# -------------------------------------------------------------------------------- #

class HierarchicalClassifier(nn.Module, ABC):

    def __init__(self, num_structures: int) -> None:
        super().__init__()
        if num_structures < 1:
            raise ValueError(f"num_structures must be at least 1. Got {num_structures} instead.")
        if num_structures == 1:
            warnings.warn("HierarchicalClassifier with num_structures=1 is equivalent to a standard Classifier.")
        self.num_structures = num_structures

    @abstractmethod
    def forward(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
        Returns:
            z: Classification logits of shape (B, M).
        """
        ...

    @abstractmethod
    def fully_shard(self, **kwds: Any) -> None:
        ...

    @property
    @abstractmethod
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        ...

    def _get_min_max_lengths(self) -> list[tuple[int, int]]:
        lengths = []
        for i in range(self.num_structures):
            trunk = self._trunks[i]
            los, his = zip(*(get_model_input_lengths(m) for m in trunk))
            lengths.append((max(los), min(his)))
        return lengths

    @property
    def min_lengths(self) -> list[int]:
        return [l[0] for l in self._get_min_max_lengths()]

    @property
    def max_lengths(self) -> list[int]:
        return [l[1] for l in self._get_min_max_lengths()]

    def _check_forward_inputs(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> None:
        if not (len(x) == len(g) == self.num_structures):
            raise ValueError(f"Expected {self.num_structures} structures, got {len(x)=} and {len(g)=} instead.")

        if all(x[i] is None for i in range(self.num_structures)):
            raise ValueError("At least one structure must have input.")

        x_ref: Tensor = next((x[i] for i in range(self.num_structures) if x[i] is not None))  # type: ignore[assignment]

        for i in range(self.num_structures):
            if x[i] is None:
                continue
            check_tensor(x[i], (x_ref.shape[0], None), INTEGERS)  # type: ignore[arg-type]
            if g[i] is not None:
                check_tensor(g[i], (x_ref.shape[0], x[i].shape[1], None), FLOATS)  # type: ignore[union-attr,arg-type]


class HierarchicalMalConvClassifier(HierarchicalClassifier):
    """
    Uses disparate MalConv models (Embedding + FiLM + MalConv) to process multiple input structures
        and averages these hidden representations before feeding them to a classification head.
    """

    def __init__(self, embeddings: Sequence[nn.Embedding], filmers: Sequence[FiLM | FiLMNoP], backbones: Sequence[MalConvBase], head: ClassifificationHead) -> None:
        super().__init__(len(embeddings))

        if not (len(embeddings) == len(filmers) == len(backbones)):
            raise ValueError("The number of embeddings, filmers, and backbones must be the same.")

        self.embeddings = nn.ModuleList(embeddings)
        self.filmers = nn.ModuleList(filmers)
        self.backbones = nn.ModuleList(backbones)
        self.head = head

    def forward(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> Tensor:
        self._check_forward_inputs(x, g)

        zs: list[Tensor] = []
        for i in range(self.num_structures):
            if x[i] is None:
                continue

            def preprocess(x: Tensor, g: Optional[Tensor] = None) -> Tensor:
                z = self.embeddings[i](x)  # (B, T, E)
                z = self.filmers[i](z, g) if torch.is_grad_enabled() else self.filmers[i].forward_functional(z, g)  # type: ignore[operator]
                return z

            ts: tuple[Tensor, ...] = (x[i], g[i]) if g[i] is not None else (x[i],)  # type: ignore[assignment]
            z = self.backbones[i](preprocess=preprocess, ts=ts)  # (B, C)
            zs.append(z)

        z = torch.stack(zs, dim=1)  # (B, sum(C))
        z = torch.mean(z, dim=1)    # (B, C)
        z = self.head(z)            # (B, M)

        check_tensor(z, (zs[0].shape[0], self.head.num_classes), FLOATS)
        return z

    @property
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        trunks = []
        for i in range(self.num_structures):
            trunk = (self.embeddings[i], self.filmers[i], self.backbones[i])
            trunks.append(trunk)
        return trunks

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(self.num_structures):
            fully_shard([self.embeddings[i], self.filmers[i]], **kwds)
            self.backbones[i].fully_shard(**kwds)  # type: ignore[operator]
        fully_shard(self, **kwds | {"reshard_after_forward": False})


class HierarchicalViTClassifier(HierarchicalClassifier):
    """
    Uses disparate ViT trunks (Embedding + FiLM + PatchEncoder) to process multiple input structures
        and feeds the encoded patches to a shared ViT backbone followed by a classification head.
    """

    def __init__(self, embeddings: Sequence[nn.Embedding], filmers: Sequence[FiLM | FiLMNoP], patchers: Sequence[PatchEncoderBase], norms: Sequence[Optional[nn.LayerNorm]], patchposencoders: Sequence[PatchPositionalityEncoder | Identity], backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__(len(embeddings))

        if not (len(embeddings) == len(filmers) == len(patchers)):
            raise ValueError("The number of embeddings, filmers, and patchers must be the same.")

        self.embeddings = nn.ModuleList(embeddings)
        self.filmers = nn.ModuleList(filmers)
        self.patchers = nn.ModuleList(patchers)
        self.norms = nn.ModuleList([n if n is not None else nn.LayerNorm(p.out_channels) for n, p in zip(norms, patchers)])
        self.patchposencoders = nn.ModuleList(patchposencoders)
        self.backbone = backbone
        self.head = head

        # The semantics of injecting absolute positional information into each structure's patches is a little murky.
        # Here, we treat each structure as its own "sequence" of patches, and inject the positional encoding accordingly.
        # So the positional information is not injected with regard to the entire file, only the structure's own patches.
        self.should_get_lengths = isinstance(self.patchposencoders[0], PatchPositionalityEncoder) and self.patchposencoders[0].max_length is not None

        C_s = [p.out_channels for p in patchers]
        if not all (c == C_s[0] for c in C_s):
            raise ValueError("For HierarchicalViT, all patchers must output the same number of channels.")
        self.C_s: list[int] = C_s
        self.C = self.C_s[0]

        N_s = [p.num_patches for p in patchers]
        if any(n is None for n in N_s):
            raise ValueError("For HierarchicalViT, all patchers must output a fixed number of patches.")
        self.N_s: list[int] = N_s  # type: ignore[assignment]
        self.N = sum(self.N_s)

    def forward(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
        Returns:
            z: Classification logits of shape (B, M).
        """
        self._check_forward_inputs(x, g)

        def preprocess(i: int, x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embeddings[i](x)  # (B, T, E)
            z = self.filmers[i](z, g) if torch.is_grad_enabled() else self.filmers[i].forward_functional(z, g)  # type: ignore[operator]
            return z

        # Process each structure separately.
        zs: list[Optional[Tensor]] = []
        for i in range(self.num_structures):
            if x[i] is None:
                zs.append(None)
                continue
            lengths = (x[i] != 0).sum(dim=1) if self.should_get_lengths else None  # type: ignore[union-attr]
            ts = (x[i], g[i]) if g[i] is not None else (x[i],)
            preprocess_ = partial(preprocess, i)
            z = self.patchers[i](preprocess=preprocess_, ts=ts)  # (B, N, C)
            z = self.norms[i](z)                                 # (B, N, C)
            z = self.patchposencoders[i](z, lengths)             # (B, N, C)
            zs.append(z)

        # Find a representative encoded patch.
        for i in range(self.num_structures):
            if x[i] is not None:
                B, device, dtype = zs[i].shape[0], zs[i].device, zs[i].dtype  # type: ignore[union-attr]
                break
        else:
            raise RuntimeError("At least one structure must have input.")

        # Fill in missing structures with zeros.
        for i in range(self.num_structures):
            if zs[i] is None:
                zs[i] = torch.zeros((B, self.N_s[i], self.C), device=device, dtype=dtype)
        assert all(z is not None for z in zs)
        zs: list[Tensor] = zs  # type: ignore[assignment]

        # Concatenate all encoded patches along patch dimension.
        z = torch.cat(zs, dim=1)      # (B, Σ(N), C)
        check_tensor(z, (B, self.N, self.C), FLOATS)

        # Feed concatenated patches to shared ViT backbone and classification head.
        z = self.backbone(z)          # (B, D)
        z = self.head(z)              # (B, M)

        check_tensor(z, (zs[0].shape[0], self.head.num_classes), FLOATS)
        return z

    @property
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        trunks = []
        for i in range(self.num_structures):
            trunk = (self.embeddings[i], self.filmers[i], self.patchers[i])
            trunks.append(trunk)
        return trunks

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(self.num_structures):
            fully_shard([self.embeddings[i], self.filmers[i], self.patchers[i]], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

    @property
    def last_aux_loss(self) -> Optional[Tensor]:
        raise NotImplementedError()

    @property
    def last_usage(self) -> Optional[Tensor]:
        raise NotImplementedError()

    @property
    def last_entropy(self) -> Optional[Tensor]:
        raise NotImplementedError()

# -------------------------------------------------------------------------------- #
# Structural Classifiers
# -------------------------------------------------------------------------------- #

def _reconcile_per_structure_patches_dynamic(zs: list[Tensor], order: list[list[tuple[int, int]]]) -> tuple[Tensor, Tensor]:
    """
    Reconcile the sample index and the input index.

    This function takes a list of unordered latent representations corresponding to different
    structures and stacks them into a single tensor according to the provided order, which
    indicates where each samples' structures should go.

    Args:
        zs: list of Tensors of shape (B_i, N_i, C) for each structure i.
        order: the order in which each input should be processed per sample, given as a list of lists of (structure_index, input_index) tuples.

    Returns:
        z: Tensor of shape (B, N, C).
        key_padding_mask: Tensor of shape (B, N) where True indicates padding positions.
    """
    B = len(order)

    per_sample_tokens: list[Tensor] = []

    for b in range(B):
        tokens_b: list[Tensor] = []

        for struct_idx, local_idx in order[b]:
            z_struct = zs[struct_idx][local_idx]  # (N_i, C)
            tokens_b.append(z_struct)

        if not tokens_b:
            raise RuntimeError(f"Sample {b} has no structures to process.")

        seq_b = torch.cat(tokens_b, dim=0)  # (N_b, C)
        per_sample_tokens.append(seq_b)

    # Now pad these variable-length sequences into a (B, N_max, C) tensor.
    M = max(seq.size(0) for seq in per_sample_tokens)
    C = per_sample_tokens[0].size(1)

    z = torch.zeros((B, M, C), device=per_sample_tokens[0].device, dtype=per_sample_tokens[0].dtype)

    key_padding_mask = torch.full((B, M), False, device=z.device, dtype=torch.bool)
    for b, seq in enumerate(per_sample_tokens):
        length = seq.size(0)
        z[b, :length] = seq
        key_padding_mask[b, length:] = True

    return z, key_padding_mask


def _reconcile_per_structure_patches_static(
    zs: list[Tensor],
    order_struct: Tensor,
    order_local: Tensor,
    struct_sizes: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Reconcile the sample index and the input index.

    This function takes a heterogeneous list of latent representations corresponding to different
    structure encoders and stacks them into a single tensor according to the provided order, into
    a sample-first, structure-first packed representation of the entire batch.

    Importantly, this function aims to be as efficient as possible by disallowing CUDA
    synchronization points, reducing the total volume of CUDA kernels, and minimizing
    the net work executed by kernels. To do so, we need several precomputed tensors on device.

    Args:
        zs: list of S float Tensors of shape (B_s, 1, C) for each structure s.
        order_struct: a (B, N) shaped long Tensor of structure indices (padded with -1)
            containing the same data as `order` but in Tensor form and on device.
        order_local: a (B, N) shaped long Tensor of local indices (padded with -1)
            containing the same data as `order` but in Tensor form and on device.
        struct_sizes: a (S,) shaped long Tensor of sizes of each structure's zs along
            the batch dimension.

    Returns:
        z: Tensor of shape (B, N, C).
        key_padding_mask: Tensor of shape (B, N) where True indicates padding positions.
    """
    B = order_struct.shape[0]
    N = order_struct.shape[1]
    S = len(zs)

    device = order_struct.device

    check_tensor(order_struct, (B, N), INTEGERS)
    check_tensor(order_local, (B, N), INTEGERS)
    check_tensor(struct_sizes, (S,), INTEGERS) if struct_sizes is not None else None

    # Concatenate all zs tensors into a single tensor
    z_concat = torch.cat(zs, dim=0)

    # Get the size of each structure tensor along the first dimension
    if struct_sizes is None:
        # NOTE: this will introduce a small CUDA sync if zs are on CUDA
        struct_sizes = torch.tensor([z.shape[0] for z in zs], dtype=torch.long, device=device)

    # Cumulative sum to get offsets [0, B_0, B_0+B_1, ...]
    zeros = torch.zeros(1, dtype=torch.long, device=device)
    struct_offsets = torch.cat([zeros, struct_sizes.cumsum(dim=0)[:-1]])

    # Create padding mask first (single kernel)
    key_padding_mask = (order_struct == -1)

    # Clamp struct indices to valid range to avoid indexing errors
    order_struct_safe = order_struct.clamp(min=0)

    # Gather offsets for each position
    gathered_offsets = struct_offsets[order_struct_safe]

    # Compute flat indices
    flat_indices = gathered_offsets + order_local

    # Clamp flat indices to valid range (for padding positions, index doesn't matter)
    flat_indices = flat_indices.clamp(min=0, max=z_concat.shape[0] - 1)

    # Gather from concatenated tensor using advanced indexing
    z = z_concat[flat_indices]

    # Squeeze out the singleton dimension
    z = z.squeeze(2)

    # Zero out padding positions
    z = z.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

    return z, key_padding_mask


class StructuralClassifier(nn.Module, ABC):

    def __init__(self, num_structures: int) -> None:
        super().__init__()
        if num_structures < 1:
            raise ValueError(f"num_structures must be at least 1. Got {num_structures} instead.")
        if num_structures == 1:
            warnings.warn("HierarchicalClassifier with num_structures=1 is equivalent to a standard Classifier.")
        self.num_structures = num_structures

    @abstractmethod
    def forward(
        self,
        x: list[Tensor],
        g: list[Optional[Tensor]],
        order: list[list[tuple[int, int]]],
        order_struct: Tensor,
        order_local: Tensor,
        struct_sizes: Tensor,
    ) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
            order: the order in which each input should be processed per sample, given as a list of lists of (structure_index, input_index) tuples.
            order_struct: (B, N) int tensor of structure indices; padded with -1.
            order_local: (B, N) int tensor of local indices into zs[struct][:, ...]; padded with -1.
            struct_sizes: (S,) int tensor of sizes of each structure's zs along the batch dimension.

        Returns:
            z: Classification logits of shape (B, M).
        """
        ...

    @abstractmethod
    def fully_shard(self, **kwds: Any) -> None:
        ...

    @property
    @abstractmethod
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        ...

    def _get_min_max_lengths(self) -> list[tuple[int, int]]:
        lengths = []
        for i in range(self.num_structures):
            trunk = self._trunks[i]
            los, his = zip(*(get_model_input_lengths(m) for m in trunk))
            lengths.append((max(los), min(his)))
        return lengths

    @property
    def min_lengths(self) -> list[int]:
        return [l[0] for l in self._get_min_max_lengths()]

    @property
    def max_lengths(self) -> list[int]:
        return [l[1] for l in self._get_min_max_lengths()]

    def _check_forward_inputs(
        self,
        x: list[Tensor],
        g: list[Optional[Tensor]],
        order: list[list[tuple[int, int]]],
        order_struct: Tensor,
        order_local: Tensor,
        struct_sizes: Tensor,
    ) -> None:
        if not (len(x) == len(g) == self.num_structures):
            raise ValueError(f"Expected {self.num_structures} structures, got {len(x)=} and {len(g)=} instead.")

        for i in range(self.num_structures):
            check_tensor(x[i], (x[i].shape[0], x[i].shape[1]), INTEGERS)
            if g[i] is not None:
                assert isinstance(g[i], Tensor)
                check_tensor(g[i], (x[i].shape[0], x[i].shape[1], None), FLOATS)  # type: ignore[arg-type]

        if len(order) <= 0:
            raise ValueError("Order must indicate at least one sample.")

        for sample_idx in range(len(order)):
            for structure_idx, local_idx in order[sample_idx]:
                if not (0 <= structure_idx < self.num_structures):
                    raise ValueError(f"Order for sample {sample_idx} has invalid structure index {structure_idx}.")
                if not (0 <= local_idx < x[structure_idx].shape[0]):
                    raise ValueError(f"Order for sample {sample_idx}, structure {structure_idx} has invalid local index {local_idx}.")
                if x[structure_idx].shape[0] <= 0:
                    raise ValueError(f"Structure {structure_idx} has no inputs, but is referenced in order for sample {sample_idx}.")

        max_structures = max(len(o) for o in order)
        check_tensor(order_struct, (len(order), max_structures), INTEGERS)
        check_tensor(order_local,  (len(order), max_structures), INTEGERS)
        check_tensor(struct_sizes, (self.num_structures,), INTEGERS)


class StructuralMalConvClassifier(StructuralClassifier):

    def __init__(self, embeddings: Sequence[nn.Embedding], filmers: Sequence[FiLM | FiLMNoP], backbones: Sequence[MalConvBase], head: ClassifificationHead) -> None:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    def forward(
        self,
        x: list[Tensor],
        g: list[Optional[Tensor]],
        order: list[list[tuple[int, int]]],
        order_struct: Tensor,
        order_local: Tensor,
        struct_sizes: Tensor,
    ) -> Tensor:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    @property
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    def fully_shard(self, **kwds: Any) -> None:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")


class StructuralViTClassifier(StructuralClassifier):

    def __init__(
        self,
        embeddings: Sequence[nn.Embedding],
        filmers: Sequence[FiLM | FiLMNoP],
        patchers: Sequence[PatchEncoderBase],
        norms: Sequence[Optional[nn.LayerNorm]],
        patchposencoders: Sequence[PatchPositionalityEncoder | Identity],
        backbone: ViT,
        head: ClassifificationHead,
        *,
        batch_sizes: Optional[tuple[int, ...]] = None,
        pad_to_batch_size: bool = False,
        do_ddp_keepalive: bool = True,
    ) -> None:
        """
        Args:
            batch_sizes (Optional[tuple[int, ...]]): If provided, will execute the per-structure forward
                with the largest possible batch size from this list. If not provided, will execute the
                per-structure forward with the batch size set to number of samples passed to the top-level
                forward method or the number of instances of said structure --- whichever is smaller.
            pad_to_batch_size: If True, the inputs in the per-structure forward call will be zero padded
                along the batch dimension to the next-largest batch size specified in `batch_sizes`.
            do_ddp_keepalive: If True, will apply DDP keepalive trick to ensure all trunks participate
                in the autograd graph even if they have no inputs for a given forward pass. Note that this
                does entail computational overhead. If False, expect that many trunks will not have gradients;
                training code must handle this case appropriately, e.g., DDP(find_unused_parameters=True), etc.
        """
        super().__init__(len(embeddings))

        if not (len(embeddings) == len(filmers) == len(patchers)):
            raise ValueError("The number of embeddings, filmers, and patchers must be the same.")

        self.embeddings = nn.ModuleList(embeddings)
        self.filmers = nn.ModuleList(filmers)
        self.patchers = nn.ModuleList(patchers)
        self.norms = nn.ModuleList([n if n is not None else nn.LayerNorm(p.out_channels) for n, p in zip(norms, patchers)])
        self.patchposencoders = nn.ModuleList(patchposencoders)
        self.backbone = backbone
        self.head = head

        if not all (isinstance(p, Identity) for p in patchposencoders):
            raise NotImplementedError("StructuralViTClassifier currently only supports Identity patchposencoders.")

        # The semantics of injecting absolute positional information into each structure's patches is a little murky.
        # Here, we treat each structure as its own "sequence" of patches, and inject the positional encoding accordingly.
        # So the positional information is not injected with regard to the entire file, only the structure's own patches.
        self.should_get_lengths = isinstance(self.patchposencoders[0], PatchPositionalityEncoder) and self.patchposencoders[0].max_length is not None

        C_s = [p.out_channels for p in patchers]
        if not all (c == C_s[0] for c in C_s):
            raise ValueError("For HierarchicalViT, all patchers must output the same number of channels.")
        self.C_s: list[int] = C_s
        self.C = self.C_s[0]

        if not all (p.num_patches == 1 for p in patchers):
            warnings.warn("Some patchers output more than one patch, so we'll have to use the slower dynamic reconcile method.")
            self._use_dynamic_reconcile = True
        else:
            self._use_dynamic_reconcile = False

        self.last_aux_loss: Optional[Tensor] = None  # (S,)   Most recent auxiliary loss for load balancing.
        self.last_usage: Optional[Tensor] = None     # (S,E)  Most recent expert usage statistic.
        self.last_entropy: Optional[Tensor] = None   # (S,)   Most recent routing entropy.
        self._maybe_reset_expert_stats()

        self.batch_sizes = batch_sizes
        self.pad_to_batch_size = pad_to_batch_size
        self._validate_batching_attributes()

        self.do_ddp_keepalive = do_ddp_keepalive

    def forward(
        self,
        x: list[Tensor],
        g: list[Optional[Tensor]],
        order: list[list[tuple[int, int]]],
        order_struct: Tensor,
        order_local: Tensor,
        struct_sizes: Tensor,
    ) -> Tensor:

        B = len(order)

        self._maybe_reset_expert_stats()

        self._check_forward_inputs(x, g, order, order_struct, order_local, struct_sizes)

        self._validate_batching_attributes()

        # Process each structure separately.
        zs: list[Tensor] = []
        for i in range(self.num_structures):
            zbs: list[Tensor] = []
            # Process the inputs within each structure in batches.
            B_i = x[i].shape[0]
            b = 0
            while b < B_i:
                # Select the internal batch size for this structure.
                remaining = B_i - b
                if self.batch_sizes is None:
                    # Use all remaining samples as the batch size.
                    bs = min(B, remaining)
                elif self.pad_to_batch_size:
                    # Find the smallest batch size that is at least as large as the remaining samples.
                    bs = min((s for s in self.batch_sizes if s >= remaining), default=max(self.batch_sizes))
                else:
                    # Find the largest batch size that is at most as large as the remaining samples.
                    bs = max((s for s in self.batch_sizes if s <= remaining), default=min(self.batch_sizes))
                # Determine how much padding is needed to reach the internal batch size
                # and how many samples from the internal input are going to be real.
                real = min(bs, remaining)
                pad  = bs - real
                # Slice the inputs for this internal batch and possibly pad them.
                x_i_b = x[i][b:b + real]
                g_i_b = g[i][b:b + real] if g[i] is not None else None  # type: ignore[index]
                if self.pad_to_batch_size and pad > 0:
                    x_i_b, g_i_b = self._pad_inputs(x_i_b, g_i_b, pad)
                # Run the inputs through the structure's trunk, remove the pad samples, and collect.
                z = self._forward_trunk(i, x_i_b, g_i_b)
                if self.pad_to_batch_size:
                    z = z[:real]
                zbs.append(z)
                # Bookkeeping.
                self._maybe_update_expert_stats(i)
                # Increment the internal batch pointer.
                b += real
            # Verify we processed the correct number of samples for this structure, concatenate, and collect.
            if b != B_i:
                raise RuntimeError(f"Structure {i} processed {b} samples, but expected {B_i}.")
            zs.append(torch.cat(zbs, dim=0))  # (B_i, N_i, C)

        # Reconcile the sample index and the input index.
        z, key_padding_mask = self.reconcile_per_structure_patches(zs, order, order_struct, order_local, struct_sizes)  # (B, N, C)
        check_tensor(z, (B, None, self.C), FLOATS)

        # Ensure all trunks participate in autograd graph.
        if self.do_ddp_keepalive:
            used = {s for sample in order for (s, _) in sample}
            unused = tuple(i for i in range(self.num_structures) if i not in used)
            z = self._ddp_keepalive(z, unused)

        # Feed concatenated patches to shared ViT backbone and classification head.
        z = self.backbone(z, key_padding_mask=key_padding_mask)  # (B, D)
        z = self.head(z)                                         # (B, M)

        check_tensor(z, (B, self.head.num_classes), FLOATS)
        return z

    def _forward_trunk(self, i: int, x: Tensor, g: Optional[Tensor]) -> Tensor:

        def preprocess(x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embeddings[i](x)  # (B, T, E)
            z = self.filmers[i](z, g) if torch.is_grad_enabled() else self.filmers[i].forward_functional(z, g)  # type: ignore[operator]
            return z

        ts = (x, g) if g is not None else (x,)
        lengths = (x != 0).sum(dim=1) if self.should_get_lengths else None

        z = self.patchers[i](preprocess=preprocess, ts=ts)
        z = self.norms[i](z)
        z = self.patchposencoders[i](z, lengths)

        return z

    @staticmethod
    def _pad_inputs(x: Tensor, g: Optional[Tensor], pad: int) -> tuple[Tensor, Optional[Tensor]]:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad), mode="constant", value=0)
        if g is not None:
            g = torch.nn.functional.pad(g, (0, 0, 0, pad), mode="constant", value=0.0)
        return x, g

    def reconcile_per_structure_patches(
        self,
        zs: list[Tensor],
        order: list[list[tuple[int, int]]],
        order_struct: Tensor,
        order_local: Tensor,
        struct_sizes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if STRUCTURAL_CLASSIFIER_USE_DYNAMIC_RECONCILE or self._use_dynamic_reconcile:
            return _reconcile_per_structure_patches_dynamic(zs, order)
        return _reconcile_per_structure_patches_static(zs, order_struct, order_local, struct_sizes)

    @property
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        trunks = []
        for i in range(self.num_structures):
            trunk = (self.embeddings[i], self.filmers[i], self.patchers[i], self.norms[i], self.patchposencoders[i])
            trunks.append(trunk)
        return trunks

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(self.num_structures):
            fully_shard([self.embeddings[i], self.filmers[i], self.patchers[i]], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

    def _ddp_keepalive(self, out: torch.Tensor, which: tuple[int, ...]) -> torch.Tensor:
        """
        Ensure all trunk parameters participate in the autograd graph every step.
        This avoids DDP 'unused parameter' errors when some trunks have no inputs.
        """
        if (not self.training) or (not torch.is_grad_enabled()) or (not which):
            return out

        parameters = []
        for i, trunk in enumerate(self._trunks):
            if i not in which:
                continue
            for mod in trunk:
                parameters.append(mod.parameters())
        out = _ddp_keepalive(out, chain.from_iterable(parameters))

        return out

    def _maybe_reset_expert_stats(self) -> None:
        """
        If using MoE patch encoders, zero the expert statistics used for monitoring them.
        """
        if not any(isinstance(p, PatchEncoderLowMemSwitchMoE) for p in self.patchers):
            return
        device = next(self.parameters()).device
        max_num_experts = max([p.num_experts if isinstance(p, PatchEncoderLowMemSwitchMoE) else 0 for p in self.patchers])
        self.last_aux_loss = torch.zeros((self.num_structures,), device=device)
        self.last_usage = torch.zeros((self.num_structures, max_num_experts), device=device)
        self.last_entropy = torch.zeros((self.num_structures,), device=device)

    def _maybe_update_expert_stats(self, i: int) -> None:
        """
        If using MoE patch encoders, update the expert statistics used for monitoring them.
        """
        patcher = self.patchers[i]
        if not isinstance(patcher, PatchEncoderLowMemSwitchMoE):
            return
        assert self.last_aux_loss is not None
        assert self.last_usage is not None
        assert self.last_entropy is not None
        self.last_aux_loss[i] += patcher.last_aux_loss
        self.last_usage[i, :patcher.num_experts] += patcher.last_usage
        self.last_entropy[i] += patcher.last_entropy

    def _validate_batching_attributes(self) -> None:
        """
        Validates that the batching properties are set correctly.
        """
        if self.batch_sizes is None:
            return
        if len(self.batch_sizes) == 0:
            raise ValueError("batch_sizes must contain at least one batch size.")
        if any(s < 1 for s in self.batch_sizes):
            raise ValueError("All batch sizes in batch_sizes must be at least 1.")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("All batch sizes in batch_sizes must be unique.")
        # This check is absolutely critical. If not met, the entire forward method is invalid.
        if not self.pad_to_batch_size and 1 not in self.batch_sizes:
            raise ValueError("If batch_sizes is provided, pad_to_batch_size must be True or 1 must be in batch_sizes.")
