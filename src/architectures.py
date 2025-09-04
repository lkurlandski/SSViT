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

from abc import ABC
from abc import abstractmethod
from functools import partial
import inspect
import math
import sys
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Sequence
import warnings

import torch
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F
from torch import nn
from torch import Tensor

from src.utils import check_tensor
from src.utils import pad_sequence


NORMS: dict[str, nn.Module] = {
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


# -------------------------------------------------------------------------------- #
# Other
# -------------------------------------------------------------------------------- #

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

class FiLM(nn.Module):  # type: ignore[misc]
    """
    Feature-wise linear modulation (FiLM) layer.

    See: Perez "Film: Visual reasoning with a general conditioning layer" AAAI 2018.
    """

    def __init__(self, guide_dim: int, embedding_dim: int, hidden_size: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(guide_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * embedding_dim)
        )

        self.guide_dim = guide_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

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
    
        film: Tensor = self.mlp(g)               # (B, T, 2E)
        gamma, beta = film.chunk(2, dim=-1)      # (B, T, E), (B, T, E)
        z = x * gamma + beta                     # (B, T, E)

        check_tensor(z, (x.shape[0], x.shape[1], self.embedding_dim), FLOATS)
        return z


class FiLMNoP(nn.Module):  # type: ignore[misc]
    """
    No-op FiLM layer that does nothing but check the inputs.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, g: Literal[None]) -> Tensor:
        check_tensor(x, (None, None, None), FLOATS)
        if g is not None:
            raise ValueError(f"Expected g to be None, got {type(g)} instead.")

        return x

# -------------------------------------------------------------------------------- #
# ViT
# -------------------------------------------------------------------------------- #

class SinusoidalPositionalEncoding(nn.Module):  # type: ignore[misc]
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


class PatchEncoder(nn.Module):  # type: ignore[misc]
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
        S = math.floor((T - self.kernel_size) / self.stride + 1)

        z = self.split_patches(z, P)  # (B,  N, P, E)
        z = z.reshape(B * N, P, E)    # (BN, P, E)
        z = z.permute(0, 2, 1)        # (BN, E, P)
        z = self.conv(z)              # (BN, C, S)
        z = self.pool(z)              # (BN, C, 1)
        z = z.squeeze(-1)             # (BN, C)
        z = z.reshape(B, N, C)        # (B,  N, C)

        return z

    @staticmethod
    def split_patches(z: Tensor, patch_size: int) -> Tensor:
        """
        Args:
            z: Input tensor of shape (B, T, E).
        Returns:
            Output tensor of shape (B, N, P, E).
        """
        if z.shape[1] <= patch_size:
            return z.unsqueeze(1)

        patches = torch.split(z.permute(1, 0, 2), patch_size)  # N x (P, B, E)
        patches = pad_sequence(patches, batch_first=True)      # (N, P, B, E)
        patches = patches.permute(2, 0, 1, 3)                  # (B, N, P, E)

        return patches

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


class ViT(nn.Module):  # type: ignore[misc]
    """
    Vision Transformer.

    See: Dosovitskiy "An image is worth 16x16 words: Transformers for image recognition at scale" ICLR 2021.

    # FIXME: remove proj; upscaling the embeddings should be done outside of ViT, e.g., in PatchEncoder.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        d_model: int = 256,
        nhead: int = 1,
        dim_feedforward: int = -1,
        activation: str = "gelu",
        num_layers: int = 1,
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
            t = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, E)
            x = torch.cat((t, x), dim=1)  # (B, T, E)
        z = self.posencoder(x)            # (B, T, E)
        z = self.proj(z)                  # (B, T, D)
        z = self.transformer(z)           # (B, T, D)
        if self.pooling == "cls":
            z = z[:, 0, :].unsqueeze(1)   # (B, 1, D)
        z = z.mean(dim=1)                 # (B, D)

        check_tensor(z, (x.shape[0], self.d_model), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(0, self.transformer.num_layers):
            fully_shard(self.transformer.layers[i], **kwds)
        fully_shard(self, **kwds)

# -------------------------------------------------------------------------------- #
# MalConv
# -------------------------------------------------------------------------------- #

class MalConvBase(nn.Module, ABC):  # type: ignore[misc]
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

    def __init__(
        self,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        chunk_size: int = 64_000,
        overlap: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.overlap = kernel_size - stride if overlap is None else overlap

    @abstractmethod
    def forward(self, x: Tensor, recompute: Optional[Callable[[int, Tensor, int], Tensor]] = None) -> Tensor:
        """
        Construct a dense hidden representation from a long sequence.

        Args:
            x: Input embeddings of shape (B, T, E).
            recompute: Callable that recomputes the activations on the winning windows. See `MalConvBase.recompute`.

        Returns:
            z: Hidden representation of shape (B, C).
        """
        ...

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard(self, **kwds)

    @staticmethod
    def recompute(preprocess: Callable[[Any], Tensor], ts: Sequence[Tensor], b: int, pos: Tensor, rf: int) -> Tensor:
        """
        Basic recomputation function that can be passed to `MalConvLowMem.forward`.

        Args:
            preprocess: Module that computed the hidden representation, `z`, passed to `forward`.
            ts:     Original input tensor(s) of shape (B, T, *).
            b:      batch index
            pos:    (U,) winning start positions
            rf:     receptive field

        Returns:
            z:     Output tensor of shape (U, rf, E), which is a subset of the original input to `forward`.

        Usage:
            >>> embedding = nn.Embedding(...)
            >>> model = MalConvLowMem(...)
            >>> x = torch.randint(...)
            >>> z = embedding(x)
            >>> model(z)                       # No gradient flow to embedding
            >>> ...
            >>> from functools import partial
            >>> recompute = partial(MalConvLowMem.recompute, embedding, [x])
            >>> model(z, recompute=recompute)  # Gradients flow to embedding
        """
        if not isinstance(ts, Sequence):
            raise TypeError(f"`ts` must be a sequence of tensors. Got {type(ts)} instead.")
        if not all(isinstance(t, Tensor) for t in ts):
            raise TypeError(f"All elements of `ts` must be tensors. Got {[type(t) for t in ts]} instead.")
        if not all(t.dim() >= 2 for t in ts):
            raise ValueError(f"All tensors in `ts` must have at least 2 dimensions. Got {[t.dim() for t in ts]} instead.")

        # Build all length-rf windows along time for the whole batch.
        # Move dimensions to ensure the dimensions larger than 2 are in the correct order.
        t_unfs = [t.unfold(dimension=1, size=rf, step=1).movedim(-1, 2).contiguous() for t in ts]  # (B, T - rf + 1, rf, *)

        # Handle degenerate case of no winners cleanly.
        if pos.numel() == 0:
            wins = [t.new_empty((0, rf) + tuple(t.shape[2:])) for t in ts]
            z = preprocess(*wins)
            z = z.contiguous()
            check_tensor(z, (0, rf, None))
            return z

        # Select this sample's winning windows and preprocess them.
        wins = [t_unf[b, pos] for t_unf in t_unfs]   # (U, rf, *)
        z = preprocess(*wins)                        # (U, rf, None)
        z = z.contiguous()

        check_tensor(z, (wins[0].shape[0] if len(wins) > 0 else None, rf, None), FLOATS)
        return z


class MalConv(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.conv_1 = nn.Conv1d(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.conv_2 = nn.Conv1d(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: Tensor, recompute: Optional[Callable[[int, Tensor, int], Tensor]] = None) -> Tensor:
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if x.shape[1] < self.min_length:
            raise RuntimeError(f"Input sequence length {x.shape[1]} is less than the minimum required length {self.min_length}.")

        z = x.transpose(1, 2)        # (B, E, T)
        c_1 = self.conv_1(z)         # (B, C, S - 1)
        c_2 = self.conv_2(z)         # (B, C, S - 1)
        g = c_1 * self.sigmoid(c_2)  # (B, C, S - 1)
        z = self.pool(g)             # (B, C, 1)
        z = z.squeeze(-1)            # (B, C)

        check_tensor(z, (x.shape[0], self.channels), FLOATS)
        return z


class MalConvLowMem(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.conv_1 = nn.Conv1d(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.conv_2 = nn.Conv1d(self.embedding_dim, self.channels, self.kernel_size, self.stride)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, recompute: Optional[Callable[[int, Tensor, int], Tensor]] = None) -> Tensor:
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if x.shape[1] < self.min_length:
            raise RuntimeError(f"Input sequence length {x.shape[1]} is less than the minimum required length {self.min_length}.")

        if self.training and recompute is None:
            warnings.warn("`recompute` is None during training; gradients will not flow to previous components.")

        z_full = x.transpose(1, 2)  # (B, E, T)

        _, pos = _lowmem_max_over_time(
            z_full,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=self._activations,
        )  # (B, C)

        if recompute is not None:
            wins_cat, meta = _gather_wins_via_recompute(
                positions=pos,
                rf=self.kernel_size,
                embedding_dim=self.embedding_dim,
                recompute=recompute,
                to_device=self.conv_1.weight.device,
                to_dtype=self.conv_1.weight.dtype,
            )                                                # wins_cat: (sum_U, E, rf)

            g_all = self._activations(wins_cat).squeeze(-1)     # (sum_U, C)
            z = _scatter_g_to_BC(
                g_all=g_all,
                meta=meta,
                batch_size=x.shape[0],
                channels=self.channels,
                to_device=g_all.device,
                to_dtype=g_all.dtype,
            )                                                   # (B, C)
        else:
            z = _eval_max_windows_vectorized(
                z_full,
                rf=self.kernel_size,
                channels=self.channels,
                positions=pos,
                activations_fn=self._activations,
                stop_grad_input=True,
            )

        check_tensor(z, (x.shape[0], self.channels), FLOATS)
        return z

    def _activations(self, z: torch.Tensor) -> torch.Tensor:
        c1 = self.conv_1(z)
        c2 = self.conv_2(z)
        return c1 * self.sigmoid(c2)


class MalConvGCG(MalConvBase):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        # Context path (no gating)
        self.ctx_conv  = nn.Conv1d(self.embedding_dim, 2 * self.channels, self.kernel_size, self.stride)
        self.ctx_share = nn.Conv1d(self.channels, self.channels, 1)
        # Main path (with gating)
        self.conv_1 = nn.Conv1d(self.embedding_dim, 2 * self.channels, self.kernel_size, self.stride)
        self.conv_2 = nn.Conv1d(self.channels, self.channels, 1)
        self.gct_proj = nn.Linear(self.channels, self.channels)

    def forward(self, x: Tensor, recompute: Optional[Callable[[int, Tensor, int], Tensor]] = None) -> Tensor:
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if x.shape[1] < self.min_length:
            raise RuntimeError(f"Input sequence length {x.shape[1]} is less than the minimum required length {self.min_length}.")
        if self.training and recompute is None:
            warnings.warn("`recompute` is None during training; gradients will not flow to previous components.")

        z_full = x.transpose(1, 2).contiguous()  # (B, E, T)

        # -------- Context path (produces gct: (B, C)) --------
        _, ctx_pos = _lowmem_max_over_time(
            z_full,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=self._ctx_activations,
        )

        if recompute is not None:
            wins_ctx, meta_ctx = _gather_wins_via_recompute(
                positions=ctx_pos,
                rf=self.kernel_size,
                embedding_dim=self.embedding_dim,
                recompute=recompute,
                to_device=self.ctx_conv.weight.device,
                to_dtype=self.ctx_conv.weight.dtype,
            )                                              # (sum_Uc, E, rf)
            g_all_ctx = self._ctx_activations(wins_ctx).squeeze(-1)     # (sum_Uc, C)
            gct = _scatter_g_to_BC(
                g_all=g_all_ctx,
                meta=meta_ctx,
                batch_size=x.shape[0],
                channels=self.channels,
                to_device=g_all_ctx.device,
                to_dtype=g_all_ctx.dtype,
            )                                              # (B, C)
        else:
            gct = _eval_max_windows_vectorized(
                z_full,
                rf=self.kernel_size,
                channels=self.channels,
                positions=ctx_pos,
                activations_fn=self._ctx_activations,
            )                                              # (B, C)

        # -------- Main path (uses gct) --------
        _, main_pos = _lowmem_max_over_time(
            z_full,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=lambda z_chunk: self._main_activations(z_chunk, gct),
        )

        if recompute is not None:
            # Gather windows for main, plus a repeated ctx per-window tensor aligned with wins_cat
            wins_main, meta_main = _gather_wins_via_recompute(
                positions=main_pos,
                rf=self.kernel_size,
                embedding_dim=self.embedding_dim,
                recompute=recompute,
                to_device=self.conv_1.weight.device,
                to_dtype=self.conv_1.weight.dtype,
            )                                              # (sum_Um, E, rf)

            # Build ctx per-window: concat repeats of gct[b] for each U_b
            ctx_per_win = []
            for (start, inv, u_b), b in zip(meta_main, range(x.shape[0])):
                if u_b == 0:
                    continue
                ctx_per_win.append(gct[b].unsqueeze(0).expand(u_b, -1))  # (U_b, C)
            ctx_cat = torch.cat(ctx_per_win, dim=0) if ctx_per_win else gct.new_empty((0, self.channels))

            g_all_main = self._main_activations(wins_main, ctx_cat).squeeze(-1)  # (sum_Um, C)

            z = _scatter_g_to_BC(
                g_all=g_all_main,
                meta=meta_main,
                batch_size=x.shape[0],
                channels=self.channels,
                to_device=g_all_main.device,
                to_dtype=g_all_main.dtype,
            )                                              # (B, C)
        else:
            z = _eval_max_windows_vectorized_per_batch(
                z_full,
                rf=self.kernel_size,
                channels=self.channels,
                positions=main_pos,
                activations_per_batch_fn=lambda wins_b, gct_b: self._main_activations(wins_b, gct_b),
                ctx=gct,
            )

        check_tensor(z, (x.shape[0], self.channels), FLOATS)
        return z

    def _ctx_activations(self, z: Tensor) -> Tensor:
        """
        Context subnetwork activations on a chunk.
        z: (B, E, T_chunk) -> (B, C, T_out)
        """
        x = F.glu(self.ctx_conv(z), dim=1)
        x = F.leaky_relu(self.ctx_share(x))
        return x

    def _main_activations(self, z: Tensor, gct: Tensor) -> Tensor:
        """
        Main subnetwork activations with GCG gating on a chunk.
        z:   (B, E, T_chunk)
        gct: (B, C)
        ->   (B, C, T_out)
        """
        h = F.glu(self.conv_1(z), dim=1)
        h = F.leaky_relu(self.conv_2(h))
        q = torch.tanh(self.gct_proj(gct))           # (B, C)
        gate = torch.sigmoid((h * q.unsqueeze(-1)).sum(dim=1, keepdim=True))  # (B, 1, T)
        return h * gate

# -------------------------------------------------------------------------------- #
# MalConv Helpers
# -------------------------------------------------------------------------------- #

def _eval_max_windows_vectorized(
    z: torch.Tensor,
    *,
    rf: int,
    channels: int,
    positions: torch.Tensor,
    activations_fn: Callable[[torch.Tensor], torch.Tensor],
    stop_grad_input: bool = True,
) -> torch.Tensor:
    B, E, T = z.shape
    device = z.device
    out = torch.empty((B, channels), device=device, dtype=z.dtype)

    z_unf = z.unfold(dimension=2, size=rf, step=1)  # (B, E, T-rf+1, rf)

    win_batches: list[torch.Tensor] = []
    meta: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    total = 0
    for b in range(B):
        pos_b = positions[b]                              # (C,)
        uniq_b, inv_b = torch.unique(pos_b, sorted=True, return_inverse=True)
        wins_b = z_unf[b, :, uniq_b].permute(1, 0, 2)     # (U_b, E, rf)
        if stop_grad_input:
            with torch.no_grad():
                wins_b = wins_b.clone()                  # detach from z; no grad to input
        win_batches.append(wins_b)
        meta.append((total, uniq_b, inv_b))
        total += wins_b.shape[0]

    wins = torch.cat(win_batches, dim=0) if len(win_batches) > 1 else win_batches[0]  # (sum_U, E, rf)

    g = activations_fn(wins)     # (sum_U, C, 1)  <- params still get grad
    g = g.squeeze(-1)            # (sum_U, C)

    for b in range(B):
        offset, uniq_b, inv_b = meta[b]
        g_b = g[offset:offset + uniq_b.shape[0]]          # (U_b, C)
        out[b] = g_b[inv_b, torch.arange(channels, device=device)]

    return out


def _lowmem_max_over_time(
    z: torch.Tensor,
    *,
    chunk_size: int,
    overlap: int,
    rf: int,
    first_stride: int,
    channels: int,
    activations_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (unchanged interface; minor micro-opts)
    """
    B, E, T = z.shape
    device, dtype = z.device, z.dtype
    if T < rf:
        raise RuntimeError(f"Input sequence length {T} is less than the minimum required length {rf}.")

    max_vals = torch.full((B, channels), torch.finfo(dtype).min, device=device, dtype=dtype)
    max_pos  = torch.zeros((B, channels), device=device, dtype=torch.long)

    step = max(1, chunk_size - overlap)
    start = 0

    with torch.no_grad():
        while start < T:
            end = min(start + chunk_size, T)
            # include extra overlap so every rf-window crossing the right edge is seen
            end_ext = min(end + overlap, T)
            z_chunk = z[:, :, start:end_ext]  # (B, E, L_chunk)

            if z_chunk.shape[-1] >= rf:
                g = activations_fn(z_chunk)             # (B, C, T_out)
                v, idx = g.max(dim=-1)                  # (B, C)
                pos = start + idx * first_stride        # map back to input

                upd = v > max_vals
                max_vals = torch.where(upd, v, max_vals)
                max_pos  = torch.where(upd, pos, max_pos)

            if end == T:
                break
            start += step

    max_pos.clamp_(0, max(0, T - rf))
    return max_vals, max_pos


def _gather_wins_via_recompute(
    *,
    positions: torch.Tensor,              # (B, C)
    rf: int,
    embedding_dim: int,
    recompute: Callable[[int, torch.Tensor, int], torch.Tensor],
    to_device: torch.device,
    to_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor, int]]]:
    """
    Calls `recompute(b, posuniq, rf)` per sample, validates shapes, casts,
    permutes to (U_b, E, rf), concatenates all windows, and returns:
      wins_cat : (sum_U, E, rf)
      meta     : list of (offset, inv, U_b) for scattering back
    """
    B = positions.shape[0]
    wins_list: list[torch.Tensor] = []
    meta: list[tuple[int, torch.Tensor, int]] = []
    offset = 0

    for b in range(B):
        pos_b = positions[b]                                                # (C,)
        posuniq, inv = torch.unique(pos_b, sorted=True, return_inverse=True)  # (U_b,), (C,)
        z_b = recompute(b, posuniq, rf)                                     # (U_b, rf, E)
        if z_b.dim() != 3 or z_b.shape[1] != rf or z_b.shape[2] != embedding_dim:
            raise RuntimeError(f"recompute must return (U_b, rf, E); got {tuple(z_b.shape)}")
        z_b = z_b.to(device=to_device, dtype=to_dtype).transpose(1, 2).contiguous()  # (U_b, E, rf)
        wins_list.append(z_b)
        meta.append((offset, inv, z_b.shape[0]))
        offset += z_b.shape[0]

    wins_cat = (torch.cat(wins_list, dim=0)
                if wins_list else
                torch.empty((0, embedding_dim, rf), device=to_device, dtype=to_dtype))
    return wins_cat, meta


def _scatter_g_to_BC(
    *,
    g_all: torch.Tensor,                                     # (sum_U, C)
    meta: list[tuple[int, torch.Tensor, int]],
    batch_size: int,
    channels: int,
    to_device: torch.device,
    to_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Maps concatenated per-window activations back to (B, C) using meta.
    """
    out = torch.empty((batch_size, channels), device=to_device, dtype=to_dtype)
    for b, (start, inv, u_b) in enumerate(meta):
        g_b = g_all[start:start + u_b]                       # (U_b, C)
        out[b] = g_b[inv, torch.arange(channels, device=to_device)]
    return out


def _eval_max_windows_vectorized_per_batch(
    z: torch.Tensor,
    *,
    rf: int,
    channels: int,
    positions: torch.Tensor,
    activations_per_batch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ctx: torch.Tensor,
    stop_grad_input: bool = True,
) -> torch.Tensor:
    """
    Vectorized recomputation per batch for activations that depend on a per-batch context.
    Args:
        z:   (B, E, T)
        rf:  receptive field
        positions: (B, C) winning start positions
        activations_per_batch_fn: (wins_b, ctx_b) -> (U_b, C, 1)
        ctx: (B, C) context (e.g., gct)
    Returns:
        (B, C)
    """
    B, E, T = z.shape
    device = z.device
    out = torch.empty((B, channels), device=device, dtype=z.dtype)

    # (B, E, T-rf+1, rf) view of all windows
    z_unf = z.unfold(dimension=2, size=rf, step=1)

    for b in range(B):
        pos_b = positions[b]                                   # (C,)
        uniq_b, inv_b = torch.unique(pos_b, sorted=True, return_inverse=True)
        wins_b = z_unf[b, :, uniq_b].permute(1, 0, 2).contiguous()  # (U_b, E, rf)

        if stop_grad_input:
            with torch.no_grad():
                wins_b = wins_b.clone()                        # detach from z graph

        ctx_b = ctx[b].unsqueeze(0).expand(wins_b.shape[0], -1)    # (U_b, C)
        g_b = activations_per_batch_fn(wins_b, ctx_b).squeeze(-1)  # (U_b, C)
        out[b] = g_b[inv_b, torch.arange(channels, device=device)]
    return out

# -------------------------------------------------------------------------------- #
# Classifiers
# -------------------------------------------------------------------------------- #

class Classifier(nn.Module, ABC):  # type: ignore[misc]

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

        recompute = None
        if self.training:
            recompute = partial(self.backbone.recompute, preprocess, (x, g) if g is not None else (x,))

        z = preprocess(x, g)              # (B, T, E)
        z = self.backbone(z, recompute)   # (B, D)
        z = self.head(z)                  # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})


class ViTClassifier(Classifier):

    def __init__(self, embedding: nn.Embedding, filmer: FiLM | FiLMNoP, patcher: PatchEncoder, backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__()

        self.embedding = embedding
        self.filmer = filmer
        self.patcher = patcher
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        self._check_forward_inputs(x, g)

        z = self.embedding(x)  # (B, T, E)
        z = self.filmer(z, g)  # (B, T, E)
        z = self.patcher(z)    # (B, N, E')
        z = self.backbone(z)   # (B, D)
        z = self.head(z)       # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer, self.patcher], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

# -------------------------------------------------------------------------------- #
# Hierarchical Classifiers
# -------------------------------------------------------------------------------- #

class HierarchicalClassifier(nn.Module, ABC):  # type: ignore[misc]

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

        x_ref: Tensor = next((x[i] for i in range(self.num_structures) if x[i] is not None))

        for i in range(self.num_structures):
            if x[i] is None:
                continue
            check_tensor(x[i], (x_ref.shape[0], None), INTEGERS)
            if g[i] is not None:
                check_tensor(g[i], (x_ref.shape[0], x[i].shape[1], None), FLOATS)  # type: ignore[union-attr]


class HierarchicalMalConvClassifier(HierarchicalClassifier):
    """
    Uses disparate MalConv models (Embedding + FiLM + MalConv) to process multiple input structures
        and averages these hidden representations before feeding them to a classification head.
    """

    def __init__(self, embeddings: list[nn.Embedding], filmers: list[FiLM | FiLMNoP], backbones: list[MalConv], head: ClassifificationHead) -> None:
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
            z = self.embeddings[i](x[i])  # (B, T, E)
            z = self.filmers[i](z, g[i])  # (B, T, E)
            z = self.backbones[i](z)      # (B, C)
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
            self.backbones[i].fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})


class HierarchicalViTClassifier(HierarchicalClassifier):
    """
    Uses disparate ViT trunks (Embedding + FiLM + PatchEncoder) to process multiple input structures
        and feeds the encoded patches to a shared ViT backbone followed by a classification head.
    """

    def __init__(self, embeddings: list[nn.Embedding], filmers: list[FiLM | FiLMNoP], patchers: list[PatchEncoder], backbone: ViT, head: ClassifificationHead) -> None:
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
