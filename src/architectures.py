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


# mypy: disable-error-code=no-any-return


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


DISABLE_LOW_MEMORY_PATHS = os.environ.get("DISABLE_LOW_MEMORY_PATHS", "0") == "1"
if DISABLE_LOW_MEMORY_PATHS:
    print("[WARN] Low-memory paths are disabled via DISABLE_LOW_MEMORY_PATHS=1")


def get_model_input_lengths(model: nn.Module) -> tuple[int, int]:
    lo = 0
    hi = sys.maxsize

    for m in model.modules():
        if isinstance(l := getattr(m, "max_length", None), int):
            hi = min(hi, l)
        if isinstance(l := getattr(m, "min_length", None), int):
            lo = max(lo, l)

    return lo, hi


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

    if isinstance(module, GatedConvolution):
        return module.forward_functional(z, fp32=fp32)

    if hasattr(module, "forward_functional"):
        return module.forward_functional(z, fp32=fp32)  # type: ignore[operator]

    raise NotImplementedError(f"functional_forward does not support {type(module)} yet.")


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


class Identity(nn.Module):
    """
    Identity layer with a bit more functionality.

    Instead of just accepting one arg to `forward`, it accepts an arbitrary number of args,
    and furthermore will autocast floating point inputs if `autocast=True`.
    """

    def __init__(self, *args: Any, autocast: bool = False, **kwds: Any) -> None:
        super().__init__()
        self.autocast = autocast

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

    def __init__(self, guide_dim: int, embedding_dim: int, hidden_size: int, fp32: bool = False):
        super().__init__()

        self.guide_dim = guide_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.fp32 = fp32

        if self.fp32:
            raise NotImplementedError()

    def forward(self, x: Tensor, g: Literal[None]) -> Tensor:
        check_tensor(x, (None, None, None), FLOATS)
        if g is not None:
            raise ValueError(f"Expected g to be None, got {type(g)} instead.")

        if torch.is_autocast_enabled():
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
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if bool(patch_size is None) == bool(num_patches is None):
            raise ValueError(f"Exactly one of patch_size or num_patches must be specified. Got {patch_size=} and {num_patches=}.")

        if patch_size is not None and patch_size < kernel_size:
            raise ValueError(f"Patch size must be greater than or equal to kernel size. Got {patch_size=} and {kernel_size=}.")

        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.pool = nn.AdaptiveMaxPool1d(1)

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

    _last_aux_loss: Tensor  # ()   Most recent auxiliary loss for load balancing.
    _last_usage: Tensor     # (E,) Most recent expert usage statistic.
    _last_entropy: Tensor   # ()   Most recent routing entropy.

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

        self.register_buffer("_last_aux_loss", torch.zeros(()), persistent=False)
        self.register_buffer("_last_usage", torch.zeros((num_experts,)), persistent=False)
        self.register_buffer("_last_entropy", torch.zeros(()), persistent=False)

        _check_lowmem_config(kernel_size, stride, chunk_size, self.overlap)
        _check_lowmem_config(probe_kernel_size, probe_stride, chunk_size, self.probe_overlap)

    def __repr__(self) -> str:
        s: str = super().__repr__()  # type: ignore[no-untyped-call]
        add = (
            "("
            f"num_experts={self.num_experts},"
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
        self._last_aux_loss = aux
        self._last_entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
        self._last_usage = dispatch.float().mean(dim=(0, 1))

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

        # Expert passes.
        out = r.new_zeros((B, N, C))
        for e, expert in enumerate(self.experts):
            active = dispatch[..., e]                                      # (B, N)
            mask = active.unsqueeze(-1).expand(B, N, C).reshape(B, N * C)  # (B, NC)
            if not active.any():
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

    @property
    def last_aux_loss(self) -> Tensor:
        return self._last_aux_loss

    @property
    def last_usage(self) -> Tensor:
        return self._last_usage

    @property
    def last_entropy(self) -> Tensor:
        return self._last_entropy


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
        pooling: Literal["mean", "cls"] = "cls",
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

        z: Tensor

        z = self.proj(x)                                   # (B, T, D)
        if self.pooling == "cls":
            assert self.cls_token is not None
            t = self.cls_token.expand(z.shape[0], -1, -1)  # (B, 1, D)
            z = torch.cat((t.to(z.dtype), z), dim=1)       # (B, T, D)
            if key_padding_mask is not None:
                t = torch.zeros((key_padding_mask.shape[0], 1), dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat((t, key_padding_mask), dim=1)  # (B, T+1)
        z = self.posencoder(z)                             # (B, T, D)
        z = self.transformer(z, src_key_padding_mask=key_padding_mask)                            # (B, T, D)
        if self.pooling == "cls":
            z = z[:, 0, :].unsqueeze(1)                    # (B, 1, D)
        z = z.mean(dim=1).to(x.dtype)                      # (B, D)

        check_tensor(z, (x.shape[0], self.d_model), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(0, self.transformer.num_layers):
            fully_shard(self.transformer.layers[i], **kwds)
        fully_shard(self, **kwds)

# -------------------------------------------------------------------------------- #
# MalConv
# -------------------------------------------------------------------------------- #

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


def _get_conv_kwds(conv: nn.Conv1d) -> dict[str, Any]:
    return {
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
        "groups": conv.groups,
    }


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

class StructuralClassifier(nn.Module, ABC):

    def __init__(self, num_structures: int) -> None:
        super().__init__()
        if num_structures < 1:
            raise ValueError(f"num_structures must be at least 1. Got {num_structures} instead.")
        if num_structures == 1:
            warnings.warn("HierarchicalClassifier with num_structures=1 is equivalent to a standard Classifier.")
        self.num_structures = num_structures

    @abstractmethod
    def forward(self, x: list[Tensor], g: list[Optional[Tensor]], order: list[list[tuple[int, int]]]) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
            order: the order in which each input should be processed per sample, given as a list of lists of (structure_index, input_index) tuples.
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

    def _check_forward_inputs(self, x: list[Tensor], g: list[Optional[Tensor]], order: list[list[tuple[int, int]]]) -> None:
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


class StructuralMalConvClassifier(StructuralClassifier):

    def __init__(self, embeddings: Sequence[nn.Embedding], filmers: Sequence[FiLM | FiLMNoP], backbones: Sequence[MalConvBase], head: ClassifificationHead) -> None:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    def forward(self, x: list[Tensor], g: list[Optional[Tensor]], order: list[list[tuple[int, int]]]) -> Tensor:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    @property
    def _trunks(self) -> Sequence[tuple[nn.Module, ...]]:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")

    def fully_shard(self, **kwds: Any) -> None:
        raise NotImplementedError("StructuralMalConvClassifier is not yet implemented.")


class StructuralViTClassifier(StructuralClassifier):

    _last_aux_loss: Tensor  # (S,)   Most recent auxiliary loss for load balancing.
    _last_usage: Tensor     # (S,E)  Most recent expert usage statistic.
    _last_entropy: Tensor   # (S,)   Most recent routing entropy.

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

        max_num_experts = max([p.num_experts if isinstance(p, PatchEncoderLowMemSwitchMoE) else 0 for p in patchers])
        self.register_buffer("_last_aux_loss", torch.zeros((self.num_structures,)), persistent=False)
        self.register_buffer("_last_usage", torch.zeros((self.num_structures, max_num_experts)), persistent=False)
        self.register_buffer("_last_entropy", torch.zeros((self.num_structures,)), persistent=False)

    def forward(self, x: list[Tensor], g: list[Optional[Tensor]], order: list[list[tuple[int, int]]]) -> Tensor:

        B = len(order)

        self._maybe_clear_expert_stats()

        self._check_forward_inputs(x, g, order)

        def preprocess(i: int, x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embeddings[i](x)  # (B, T, E)
            z = self.filmers[i](z, g) if torch.is_grad_enabled() else self.filmers[i].forward_functional(z, g)  # type: ignore[operator]
            return z

        # Process each structure separately.
        zs: list[Tensor] = []
        for i in range(self.num_structures):
            preprocess_ = partial(preprocess, i)
            zbs: list[Tensor] = []
            for b in range(0, x[i].shape[0], B):
                x_i_b = x[i][b:b + B]
                g_i_b = g[i][b:b + B] if g[i] is not None else None  # type: ignore[index]
                ts    = (x_i_b, g_i_b) if g_i_b is not None else (x_i_b,)
                lengths = (x_i_b != 0).sum(dim=1) if self.should_get_lengths else None
                z = self.patchers[i](preprocess=preprocess_, ts=ts)  # (max(B, B_i), N_i, C)
                z = self.norms[i](z)                                 # (max(B, B_i), N_i, C)
                z = self.patchposencoders[i](z, lengths)             # (max(B, B_i), N_i, C)
                zbs.append(z)
                self._maybe_log_expert_stats(i)
            zs.append(torch.cat(zbs, dim=0))  # (B_i, N_i, C)

        # Reconcile the sample index and the input index.
        z, key_padding_mask = self.reconcile_per_structure_patches(zs, order)  # (B, N, C)
        check_tensor(z, (B, None, self.C), FLOATS)

        # Ensure all trunks participate in autograd graph.
        z = self._ddp_keepalive(z)

        # Feed concatenated patches to shared ViT backbone and classification head.
        z = self.backbone(z, key_padding_mask=key_padding_mask)  # (B, D)
        z = self.head(z)                                         # (B, M)

        check_tensor(z, (B, self.head.num_classes), FLOATS)
        return z

    def reconcile_per_structure_patches(self, zs: list[Tensor], order: list[list[tuple[int, int]]]) -> tuple[Tensor, Tensor]:
        """
        Reconcile the sample index and the input index.

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

    def _ddp_keepalive(self, out: torch.Tensor) -> torch.Tensor:
        """
        Ensure all trunk parameters participate in the autograd graph every step.
        This avoids DDP 'unused parameter' errors when some trunks have no inputs.
        """
        if (not self.training) or (not torch.is_grad_enabled()):
            return out

        dummy = out.sum() * 0.0
        for trunk in self._trunks:
            for mod in trunk:
                for p in mod.parameters():
                    dummy = dummy + p.view(-1)[0] * 0.0

        return out + dummy

    def _maybe_clear_expert_stats(self) -> None:
        for i in range(self.num_structures):
            patcher = self.patchers[i]
            if isinstance(patcher, PatchEncoderLowMemSwitchMoE):
                self._last_aux_loss[i].zero_()
                self._last_usage[i, :patcher.num_experts].zero_()
                self._last_entropy[i].zero_()

    def _maybe_log_expert_stats(self, i: int) -> None:
        patcher = self.patchers[i]
        if isinstance(patcher, PatchEncoderLowMemSwitchMoE):
            self._last_aux_loss[i] = patcher.last_aux_loss
            self._last_usage[i, :patcher.num_experts] = patcher.last_usage
            self._last_entropy[i] = patcher.last_entropy

    @property
    def last_aux_loss(self) -> Optional[Tensor]:
        if any(isinstance(p, PatchEncoderLowMemSwitchMoE) for p in self.patchers):
            return self._last_aux_loss
        return None

    @property
    def last_usage(self) -> Optional[Tensor]:
        if any(isinstance(p, PatchEncoderLowMemSwitchMoE) for p in self.patchers):
            return self._last_usage
        return None

    @property
    def last_entropy(self) -> Optional[Tensor]:
        if any(isinstance(p, PatchEncoderLowMemSwitchMoE) for p in self.patchers):
            return self._last_entropy
        return None
