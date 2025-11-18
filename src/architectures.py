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


class FiLMNoP(nn.Module):
    """
    No-op FiLM layer that does nothing but check the inputs.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, g: Literal[None]) -> Tensor:
        check_tensor(x, (None, None, None), FLOATS)
        if g is not None:
            raise ValueError(f"Expected g to be None, got {type(g)} instead.")

        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())

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

# -------------------------------------------------------------------------------- #
# Patch Encoders
# -------------------------------------------------------------------------------- #

class PatchEncoderBase(nn.Module, ABC):

    def __init__(self, in_channels: int, out_channels: int, num_patches: Optional[int], patch_size: Optional[int]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._num_patches = num_patches
        self._patch_size = patch_size

    @property
    def num_patches(self) -> Optional[int]:
        return self._num_patches

    @property
    def patch_size(self) -> Optional[int]:
        return self._patch_size

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
            z = self.forward_streaming(preprocess=preprocess, ts=ts)

        else:
            raise ValueError("Either `z` or both `preprocess` and `ts` must be provided.")

        # FIXME: check the patch dimension (below)
        check_tensor(z, (z.shape[0], None, self.out_channels), FLOATS)
        return z

    @property
    def min_length(self) -> int:
        return 1

    @abstractmethod
    def forward_embeddings(self, z: Tensor) -> Tensor:
        ...

    @abstractmethod
    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        ...


class PatchEncoder(nn.Module):
    """
    Breaks a sequence into patches and convolves them fixed-size kernels.

    NOTE: `N` refers to the number of patches for a sequence of particular length
        which is often less than PatchEncoder.num_patches.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 64, stride: int = 64, *, patch_size: Optional[int], num_patches: Optional[int]) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolutional kernel.
            patch_size: Size of each patch. The sequence will broken into patch(es) of this size
                with at most one smaller patch if necessary.
            num_patches: Number of patches. The sequence will be broken into up into this many patches of equal size
                with at most one smaller patch if necessary.
        """
        super().__init__()

        if bool(patch_size is None) == bool(num_patches is None):
            raise ValueError(f"Exactly one of patch_size or num_patches must be specified. Got {patch_size=} and {num_patches=}.")

        if patch_size is not None and patch_size < kernel_size:
            raise ValueError(f"Patch size must be greater than or equal to kernel size. Got {patch_size=} and {kernel_size=}.")

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, C).
        """
        check_tensor(z, (None, None, None), FLOATS)
        if z.shape[1] < self.min_length:
            raise RuntimeError(f"Input sequence length {z.shape[1]} is less than the minimum required length {self.min_length}.")

        B = z.shape[0]
        T = z.shape[1]
        E = z.shape[2]
        P = self.patch_dims(T, self.patch_size, self.num_patches)[0]
        N = self.patch_dims(T, self.patch_size, self.num_patches)[1]
        C = self.out_channels
        S = math.floor((T - self.kernel_size) / self.stride + 1)  # noqa

        z = self.split_patches(z, P, N)  # (B,  N, P, E)
        z = z.reshape(B * N, P, E)    # (BN, P, E)
        z = z.permute(0, 2, 1)        # (BN, E, P)
        z = self.conv(z)              # (BN, C, S)
        z = self.pool(z)              # (BN, C, 1)
        z = z.squeeze(-1)             # (BN, C)
        z = z.reshape(B, N, C)        # (B,  N, C)

        return z

    @staticmethod
    def split_patches(z: Tensor, patch_size: int, num_patches: int) -> Tensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, P, E).
        """
        B, T, E = z.shape
        P, N = patch_size, num_patches

        if T < (total := N * P):
            z = torch.cat([z, z.new_zeros(B, total - T, E)], dim=1)

        z = z.view(B, N, P, E)
        return z

    @staticmethod
    def patch_dims(seq_length: int, patch_size: Optional[int], num_patches: Optional[int]) -> tuple[int, int]:
        """
        Determine the patch_size and num_patches given one of them for patchifying a sequence of length seq_length.
        """
        if seq_length <= 0:
            raise ValueError(f"Sequence length must be positive. Got {seq_length}.")
        if bool(patch_size is None) == bool(num_patches is None):
            raise ValueError(f"Exactly one of patch_size or num_patches must be specified. Got {patch_size=} and {num_patches=}.")
        if patch_size is not None and (not 0 < patch_size <= seq_length):
            raise ValueError(f"Patch size must be positive and less than or equal to the sequence length. Got {patch_size=} and {seq_length=}.")
        if num_patches is not None and (not 0 < num_patches <= seq_length):
            raise ValueError(f"Number of patches must be positive and less than or equal to the sequence length. Got {num_patches=} and {seq_length=}.")

        if patch_size is not None:
            num_patches = (seq_length + patch_size - 1) // patch_size
            return patch_size, num_patches
        if num_patches is not None:
            patch_size = (seq_length + num_patches - 1) // num_patches
            num_patches = (seq_length + patch_size - 1) // patch_size
            return patch_size, num_patches

        raise RuntimeError("This should never happen.")

    @property
    def min_length(self) -> int:
        if self.patch_size is not None:
            return max(self.kernel_size, self.patch_size)
        if self.num_patches is not None:
            return self.num_patches * self.kernel_size
        raise RuntimeError("This should never happen.")


class ConvPatchEncoder(nn.Module):
    """
    An optimized patch encoder for fixed-size patches.

    Rather than splitting the sequence into patches and convolving each patch separately,
    this module linearly maps each patch into an output embedding via a 1D convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, num_patches: Optional[int] = None) -> None:
        super().__init__()
        if num_patches is not None:
            warnings.warn("ConvPatchEncoder ignores num_patches; only patch_size is used.")
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, C).
        """
        B, T, E = z.shape
        P = self.patch_size

        if T % P != 0:
            z = torch.cat([z, z.new_zeros(B, P - (T % P), E)], dim=1)

        z = z.permute(0, 2, 1)      # (B, E, T')
        z = self.proj(z)            # (B, C, N)
        z = z.permute(0, 2, 1)      # (B, N, C)
        return z


class PatchEncoderLowMem(PatchEncoderBase):
    """
    Patch encoder with constant activation memory in sequence length T.
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
        chunk_size: int = 2**16,
        overlap: Optional[int] = None,
        fp32: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, num_patches, patch_size)

        if num_patches is None or patch_size is not None:
            raise ValueError("PatchEncoderLowMem requires `num_patches` > 0 and `patch_size` = None.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = kernel_size - stride if overlap is None else overlap
        self.fp32 = fp32

        self.gconv = GatedConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

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
        return self.kernel_size

    def forward_embeddings(self, z: Tensor) -> Tensor:
        B, T = z.shape[0], z.shape[1]
        E, C = self.in_channels, self.out_channels
        P = (T + self.num_patches - 1) // self.num_patches
        N = (T + P - 1) // P

        p = z.new_zeros(B, max(N * P - T, 0), E)  # (B, N*P - T, E)
        z = torch.cat([z, p], dim=1)              # (B, N*P, E)
        z = z.view(B, N, P, E)                    # (B, N, P, E)
        z = z.view(B * N, P, E).permute(0, 2, 1)  # (BN, E, P)
        g: Tensor = self.gconv(z)                 # (BN, C, S)
        g, _ = g.max(dim=-1)                      # (BN, C)
        g = g.view(B, N, C)                       # (B, N, C)

        return g

    def forward_streaming(self, preprocess: PreprocessFn, ts: Sequence[Tensor]) -> Tensor:
        B, T = ts[0].shape[0], ts[0].shape[1]
        E, C = self.in_channels, self.out_channels
        P = (T + self.num_patches - 1) // self.num_patches
        N = (T + P - 1) // P

        max_vals, pos = _lowmem_patchwise_max_over_time_streaming(
            preprocess=preprocess,
            ts=ts,
            rf=self.kernel_size,
            first_stride=self.stride,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            channels=C,
            num_patches=self.num_patches,
            activations_fn=partial(self.gconv.forward_functional, fp32=self.fp32),
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

        g_all: Tensor = self.gconv(wins_cat)
        g_all = g_all.squeeze(-1)

        z = _scatter_g_to_BNC(
            g_all=g_all,
            meta=meta,
            batch_size=B,
            num_patches=self.num_patches,
            channels=C,
        )

        return z


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

    max_vals: torch.Tensor
    max_pos: torch.Tensor

    step = max(1, chunk_size - overlap)
    start = 0

    with torch.no_grad():
        while start < T:
            end = min(start + chunk_size, T)
            end_ext = min(end + overlap, T)

            # Preprocess slice of inputs: (B, L, *) -> (B, L, E)
            slices = [t[:, start:end_ext] for t in ts]
            z_chunk = preprocess(*slices)                   # (B, L, E)

            if dtype is None or device is None:
                dtype = z_chunk.dtype
                device = z_chunk.device
                max_vals = torch.full(
                    (B, num_patches, channels),
                    torch.finfo(dtype).min,
                    device=device,
                    dtype=dtype,
                )
                max_pos = torch.zeros(
                    (B, num_patches, channels),
                    device=device,
                    dtype=torch.long,
                )

            z_chunk = z_chunk.transpose(1, 2).contiguous()  # (B, E, L)

            if z_chunk.shape[-1] >= rf:
                g = activations_fn(z_chunk)                 # (B, C, L_out)
                B_, C_, L_out = g.shape
                assert B_ == B and C_ == channels

                idx = torch.arange(L_out, device=device)        # (L_out,)
                pos = start + idx * first_stride                # (L_out,) input start indices
                # Map each conv output position to a patch index
                patch_idx = torch.div(pos, patch_size, rounding_mode="floor")
                patch_idx.clamp_(0, num_patches - 1)

                # For each patch, update maxima
                for j in range(num_patches):
                    mask = (patch_idx == j)                    # (L_out,)
                    if not mask.any():
                        continue

                    g_j = g[..., mask]                         # (B, C, L_j)
                    v_j, idx_j_local = g_j.max(dim=-1)         # (B, C)

                    pos_candidates = pos[mask]                 # (L_j,)
                    # Map local argmax indices to global positions (B,C)
                    pos_j = pos_candidates[idx_j_local]        # (B, C)

                    cur_v = max_vals[:, j, :]                  # (B, C)
                    cur_p = max_pos[:, j, :]

                    upd = v_j > cur_v
                    max_vals[:, j, :] = torch.where(upd, v_j, cur_v)
                    max_pos[:, j, :]  = torch.where(upd, pos_j, cur_p)

            if end == T:
                break
            start += step

    # Clamp positions to valid window starts
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
    out = torch.empty(
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

        g_b = g_all[start:start + u_b]     # (U_b, C)
        inv = inv_flat.view(num_patches, channels)  # (N, C)

        # Build per-patch/channel indices
        row_idx = inv                                  # (N, C)
        col_idx = torch.arange(channels, device=g_all.device).view(1, -1)
        col_idx = col_idx.expand(num_patches, -1)      # (N, C)
        out[b] = g_b[row_idx, col_idx]                 # (N, C)

    return out

# -------------------------------------------------------------------------------- #
# ViT
# -------------------------------------------------------------------------------- #

class ViT(nn.Module):
    """
    Vision Transformer.

    See: Dosovitskiy "An image is worth 16x16 words: Transformers for image recognition at scale" ICLR 2021.

    # FIXME: remove proj; upscaling the embeddings should be done outside of ViT, e.g., in PatchEncoder.
    """

    cls_token: Optional[nn.Parameter]

    def __init__(
        self,
        embedding_dim: int = 8,
        d_model: int = 256,
        nhead: int = 1,
        dim_feedforward: int = -1,
        num_layers: int = 1,
        activation: str = "gelu",
        norm: Optional[str] = "rms",
        pooling: Literal["mean", "cls"] = "cls",
    ) -> None:
        super().__init__()

        dim_feedforward = 4 * d_model if dim_feedforward == -1 else dim_feedforward

        self.posencoder = SinusoidalPositionalEncoding(embedding_dim)
        self.proj = nn.Linear(embedding_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation=ACTVS[activation], batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, norm=NORMS[norm](d_model) if norm is not None else None)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
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
        self.norm = norm
        self.pooling = pooling

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, T, E).
        Returns:
            z: Hidden representation of shape (B, D).
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)

        if self.pooling == "cls":  # T <-- T + 1
            assert self.cls_token is not None
            t = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, E)
            x = torch.cat((t.to(x.dtype), x), dim=1)  # (B, T, E)
        z = self.posencoder(x)            # (B, T, E)
        z = self.proj(z)                  # (B, T, D)
        z = self.transformer(z)           # (B, T, D)
        if self.pooling == "cls":
            z = z[:, 0, :].unsqueeze(1)   # (B, 1, D)
        z = z.mean(dim=1).to(x.dtype)     # (B, D)

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
            if any(w_b.dtype != torch.float32 for w_b in filter(lambda w_b: w_b is not None, [w1, w2, b1, b2])):
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
        self.overlap = kernel_size - stride if overlap is None else overlap
        self.fp32 = fp32

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
            if any(w_b.dtype != torch.float32 for w_b in filter(lambda w_b: w_b is not None, [w1, w2, wp, b1, b2, bp])):
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
            if any(w_b.dtype != torch.float32 for w_b in filter(lambda w_b: w_b is not None, [w_ctx, w_sh, b_ctx, b_sh])):
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
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor, int]]]:
    B, T = ts[0].shape[:2]
    device = ts[0].device

    posuniq_list, inv_list, u_counts = [], [], []
    for b in range(B):
        pos_b = positions[b]
        posuniq_b, inv_b = torch.unique(pos_b, sorted=True, return_inverse=True)
        posuniq_list.append(posuniq_b)
        inv_list.append(inv_b)
        u_counts.append(int(posuniq_b.numel()))

    sum_U = int(sum(u_counts))
    if sum_U == 0:
        wins_empty = ts[0].new_empty((0, rf) + tuple(ts[0].shape[2:]))
        z = preprocess(*([wins_empty] * len(ts)))            # (0, rf, E)
        return z.transpose(1, 2).contiguous(), [(0, torch.empty(0, device=device, dtype=torch.long), 0) for _ in range(B)]

    arange_rf = torch.arange(rf, device=device, dtype=torch.int32)
    batch_ix_chunks, time_ix_chunks = [], []
    offset = 0
    meta: list[tuple[int, torch.Tensor, int]] = []

    for b, posuniq_b in enumerate(posuniq_list):
        U_b = int(posuniq_b.numel())
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
            z = self.filmer(z, g)  # (B, T, E)
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

    def __init__(self, embedding: nn.Embedding, filmer: FiLM | FiLMNoP, patcher: PatchEncoderBase, backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__()

        self.embedding = embedding
        self.filmer = filmer
        self.patcher = patcher
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        self._check_forward_inputs(x, g)

        def preprocess(x: Tensor, g: Optional[Tensor] = None) -> Tensor:
            z = self.embedding(x)  # (B, T, E)
            z = self.filmer(z, g)  # (B, T, E)
            return z

        ts = (x, g) if g is not None else (x,)
        z = self.patcher(preprocess=preprocess, ts=ts)  # (B, N, E')
        z = self.backbone(z)                            # (B, D)
        z = self.head(z)                                # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer, self.patcher], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

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
                z = self.filmers[i](z, g)  # (B, T, E)
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

    def __init__(self, embeddings: Sequence[nn.Embedding], filmers: Sequence[FiLM | FiLMNoP], patchers: Sequence[PatchEncoder], backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__(len(embeddings))

        if not (len(embeddings) == len(filmers) == len(patchers)):
            raise ValueError("The number of embeddings, filmers, and patchers must be the same.")

        self.embeddings = nn.ModuleList(embeddings)
        self.filmers = nn.ModuleList(filmers)
        self.patchers = nn.ModuleList(patchers)
        self.backbone = backbone
        self.head = head

    def forward(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
        Returns:
            z: Classification logits of shape (B, M).
        """
        self._check_forward_inputs(x, g)

        zs: list[Tensor] = []
        for i in range(self.num_structures):
            if x[i] is None:
                continue
            z = self.embeddings[i](x[i])  # (B, T, E)
            z = self.filmers[i](z, g[i])  # (B, T, E)
            z = self.patchers[i](z)       # (B, N, D)
            zs.append(z)

        z = torch.cat(zs, dim=1)      # (B, sum(N), D)
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
