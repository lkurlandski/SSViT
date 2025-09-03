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

import math
import sys
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
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

class MalConv(nn.Module):  # type: ignore[misc]
    """
    MalConv backbone.

    See: Raff "Malware detection by eating a whole EXE." AICS 2018.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(embedding_dim, channels, kernel_size, stride)
        self.conv_2 = nn.Conv1d(embedding_dim, channels, kernel_size, stride)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input embeddings of shape (B, T, E).
        Returns:
            z: Hidden representation of shape (B, C).
        """
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

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard(self, **kwds)

# -------------------------------------------------------------------------------- #
# MalConv2
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


class MalConvLowMem(nn.Module):  # type: ignore[misc]
    """
    MalConv backbone with low-memory global max.

    See: Raff "Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection" AAAI 2021.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        *,
        chunk_size: int = 64_000,
        overlap: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(embedding_dim, channels, kernel_size, stride)
        self.conv_2 = nn.Conv1d(embedding_dim, channels, kernel_size, stride)
        self.sigmoid = nn.Sigmoid()

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap if overlap is not None else max(kernel_size - 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if x.shape[1] < self.min_length:
            raise RuntimeError(f"Input sequence length {x.shape[1]} is less than the minimum required length {self.min_length}.")

        z = x.transpose(1, 2)  # (B, E, T)

        # Find per-channel winners (no grad)
        _, max_pos = _lowmem_max_over_time(
            z,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=self._activations,
        )

        # Recomputation for ONLY unique winning windows
        z = _eval_max_windows_vectorized(
            z,
            rf=self.kernel_size,
            channels=self.channels,
            positions=max_pos,
            activations_fn=self._activations,
        )

        check_tensor(z, (x.shape[0], self.channels), FLOATS)
        return z

    def _activations(self, z: torch.Tensor) -> torch.Tensor:
        c1 = self.conv_1(z)
        c2 = self.conv_2(z)
        return c1 * self.sigmoid(c2)

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard(self, **kwds)


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

        ctx_b = ctx[b].unsqueeze(0).expand(wins_b.shape[0], -1)  # (U_b, C)
        g_b = activations_per_batch_fn(wins_b, ctx_b).squeeze(-1)  # (U_b, C)
        out[b] = g_b[inv_b, torch.arange(channels, device=device)]
    return out


class MalConvGCG(nn.Module):  # type: ignore[misc]
    """
    MalConv backbone with low-memory global max and global channel gating (GCG).

    See: Raff "Classifying Sequences of Extreme Length with Constant Memory Applied to Malware Detection" AAAI 2021.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        channels: int = 128,
        kernel_size: int = 512,
        stride: int = 512,
        *,
        chunk_size: int = 64_000,
        overlap: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Context path (no gating)
        self.ctx_conv   = nn.Conv1d(embedding_dim, 2 * channels, kernel_size, stride=stride, bias=True)
        self.ctx_share  = nn.Conv1d(channels, channels, 1, bias=True)

        # Main path (with GCG gating)
        self.main_conv  = nn.Conv1d(embedding_dim, 2 * channels, kernel_size, stride=stride, bias=True)
        self.main_share = nn.Conv1d(channels, channels, 1, bias=True)
        self.gct_proj   = nn.Linear(channels, channels)

        self.embedding_dim = embedding_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap if overlap is not None else max(kernel_size - 1, 0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, E) input embeddings
        Returns:
            z: (B, C) feature vector (global max over time)
        """
        check_tensor(x, (None, None, self.embedding_dim), FLOATS)
        if x.shape[1] < self.min_length:
            raise RuntimeError(
                f"Input sequence length {x.shape[1]} is less than the minimum required length {self.min_length}."
            )

        z_full = x.transpose(1, 2)  # (B, E, T)

        _, ctx_pos = _lowmem_max_over_time(
            z_full,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=self._ctx_activations,
        )
        gct = _eval_max_windows_vectorized(
            z_full,
            rf=self.kernel_size,
            channels=self.channels,
            positions=ctx_pos,
            activations_fn=self._ctx_activations,
        )
        _, main_pos = _lowmem_max_over_time(
            z_full,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            rf=self.kernel_size,
            first_stride=self.stride,
            channels=self.channels,
            activations_fn=lambda z_chunk: self._main_activations(z_chunk, gct),
        )
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
        h = F.glu(self.main_conv(z), dim=1)
        h = F.leaky_relu(self.main_share(h))
        q = torch.tanh(self.gct_proj(gct))           # (B, C)
        gate = torch.sigmoid((h * q.unsqueeze(-1)).sum(dim=1, keepdim=True))  # (B, 1, T)
        return h * gate

    @property
    def min_length(self) -> int:
        return self.kernel_size

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard(self, **kwds)

# -------------------------------------------------------------------------------- #
# Classifier
# -------------------------------------------------------------------------------- #

class Classifier(nn.Module):  # type: ignore[misc]

    def __init__(self, embedding: nn.Embedding, filmer: FiLM | FiLMNoP, patcher: Optional[PatchEncoder], backbone: MalConv | ViT, head: ClassifificationHead) -> None:
        super().__init__()

        if patcher is None and isinstance(backbone, ViT):
            warnings.warn("ViT backbone is being used without a PatchEncoder.")
        if patcher is not None and isinstance(backbone, MalConv):
            warnings.warn("MalConv backbone is being used with a PatchEncoder.")

        self.embedding = embedding
        self.filmer = filmer
        self.patcher = patcher if patcher is not None else nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor, g: Optional[Tensor]) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T).
            g: FiLM conditioning vector of shape (B, T, G).
        Returns:
            z: Classification logits of shape (B, M).
        """
        check_tensor(x, (None, None), INTEGERS)
        if g is not None:
            check_tensor(g, (x.shape[0], x.shape[1], None), FLOATS)

        z = self.embedding(x)  # (B, T, E)
        z = self.filmer(z, g)  # (B, T, E)
        z = self.patcher(z)    # (B, N, E') or (B, T, E)
        z = self.backbone(z)   # (B, D)
        z = self.head(z)       # (B, M)

        check_tensor(z, (x.shape[0], self.head.num_classes), FLOATS)
        return z

    def fully_shard(self, **kwds: Any) -> None:
        fully_shard([self.embedding, self.filmer, self.patcher], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})

# -------------------------------------------------------------------------------- #
# Hierarchical Models.
# -------------------------------------------------------------------------------- #

def _check_hierarchical_inputs(x: list[Optional[Tensor]], g: list[Optional[Tensor]], num_structures: int) -> None:
    if not (len(x) == len(g) == num_structures):
        raise ValueError(f"Expected {num_structures} structures, got {len(x)=} and {len(g)=} instead.")

    if all(x[i] is None for i in range(num_structures)):
        raise ValueError("At least one structure must have input.")

    x_ref: Tensor = next((x[i] for i in range(num_structures) if x[i] is not None))

    for i in range(num_structures):
        if x[i] is None:
            continue
        check_tensor(x[i], (x_ref.shape[0], None), INTEGERS)
        if g[i] is not None:
            check_tensor(g[i], (x_ref.shape[0], x[i].shape[1], None), FLOATS)  # type: ignore[union-attr]


def _partition_to_hierarchical_inputs(x: Tensor, g: Optional[Tensor], m: Tensor, min_lengths: Optional[list[int]] = None) -> list[Optional[tuple[Tensor, Optional[Tensor]]]]:
    """
    Partitions the input tensor x (and guide tensor g) into multiple structures based on the structure indicator m.

    Args:
        x: Tensor: Input tensor of shape (B, T).
        g: Optional[Tensor]: FiLM conditioning vector of shape (B, T, G).
        m: Tensor: Structure indicator of shape (B, T, R) where R is the number of structures.
        min_lengths: Optional[list[int]]: Minimum lengths for each structure to pad to.
            If None, no minimum length padding is applied.
    """
    warnings.warn("src::architectures::_partition_to_hierarchical_inputs is depracated.")

    check_tensor(x, (None, None), INTEGERS)
    if g is not None:
        check_tensor(g, (x.shape[0], x.shape[1], None), FLOATS)
    check_tensor(m, (x.shape[0], x.shape[1], None), torch.bool)

    num_structures = m.shape[2]
    if min_lengths is None:
        min_lengths = [0] * num_structures
    if len(min_lengths) != num_structures:
        raise ValueError(f"Expected {num_structures} min_lengths, got {len(min_lengths)} instead.")

    x_g: list[Optional[tuple[Tensor, Optional[Tensor]]]] = []
    for i in range(num_structures):
        xs = []
        gs = []
        all_none = True
        for j in range(x.shape[0]):
            idx = m[j, :, i]
            all_none = all_none and not bool(idx.any())
            x_i_j = x[j, idx]
            g_i_j = g[j, idx, :] if g is not None else None
            xs.append(x_i_j)
            gs.append(g_i_j)
        if not all_none:
            xs = pad_sequence(xs, batch_first=True, padding_value=0,   pad_to_multiple_of=8, min_length=min_lengths[i])
            gs = pad_sequence(gs, batch_first=True, padding_value=0.0, pad_to_multiple_of=8, min_length=min_lengths[i]) if g is not None else None
            x_g.append((xs, gs))
        else:
            x_g.append(None)

    return x_g


class HierarchicalMalConvClassifier(nn.Module):  # type: ignore[misc]
    """
    Uses disparate MalConv models (Embedding + FiLM + MalConv) to process multiple input structures
        and averages these hidden representations before feeding them to a classification head.
    """

    def __init__(self, embeddings: list[nn.Embedding], filmers: list[FiLM | FiLMNoP], backbones: list[MalConv], head: ClassifificationHead) -> None:
        super().__init__()

        if not (len(embeddings) == len(filmers) == len(backbones)):
            raise ValueError("The number of embeddings, filmers, and backbones must be the same.")
        self.num_structures = len(embeddings)

        self.embeddings = nn.ModuleList(embeddings)
        self.filmers = nn.ModuleList(filmers)
        self.backbones = nn.ModuleList(backbones)
        self.head = head

    def forward(self, x: list[Optional[Tensor]], g: list[Optional[Tensor]]) -> Tensor:
        """
        Args:
            x: Input tensors of shape (B, T_i) for each structure i. None if no input for that structure.
            g: FiLM conditioning vectors of shape (B, T_i, G) for each structure i. None if no input for that structure or not using guides.
        Returns:
            z: Classification logits of shape (B, M).
        """
        _check_hierarchical_inputs(x, g, self.num_structures)

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

    def _get_min_max_lengths(self) -> list[tuple[int, int]]:
        lengths = []
        for i in range(self.num_structures):
            lo_e, hi_e = get_model_input_lengths(self.embeddings[i])
            lo_f, hi_f = get_model_input_lengths(self.filmers[i])
            lo_b, hi_b = get_model_input_lengths(self.backbones[i])
            lo = max(lo_e, lo_f, lo_b)
            hi = min(hi_e, hi_f, hi_b)
            lengths.append((lo, hi))
        return lengths

    @property
    def min_lengths(self) -> list[int]:
        return [l[0] for l in self._get_min_max_lengths()]

    @property
    def max_lengths(self) -> list[int]:
        return [l[1] for l in self._get_min_max_lengths()]

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(self.num_structures):
            fully_shard([self.embeddings[i], self.filmers[i]], **kwds)
            self.backbones[i].fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})


class HierarchicalViTClassifier(nn.Module):  # type: ignore[misc]
    """
    Uses disparate ViT trunks (Embedding + FiLM + PatchEncoder) to process multiple input structures
        and feeds the encoded patches to a shared ViT backbone followed by a classification head.
    """

    def __init__(self, embeddings: list[nn.Embedding], filmers: list[FiLM | FiLMNoP], patchers: list[PatchEncoder], backbone: ViT, head: ClassifificationHead) -> None:
        super().__init__()

        if not (len(embeddings) == len(filmers) == len(patchers)):
            raise ValueError("The number of embeddings, filmers, and patchers must be the same.")
        self.num_structures = len(embeddings)

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
        _check_hierarchical_inputs(x, g, self.num_structures)

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

    def _get_min_max_lengths(self) -> list[tuple[int, int]]:
        lengths = []
        for i in range(self.num_structures):
            lo_e, hi_e = get_model_input_lengths(self.embeddings[i])
            lo_f, hi_f = get_model_input_lengths(self.filmers[i])
            lo_p, hi_p = get_model_input_lengths(self.patchers[i])
            lo_b, hi_b = get_model_input_lengths(self.backbone)
            lo = max(lo_e, lo_f, lo_p, lo_b)
            hi = min(hi_e, hi_f, hi_p, hi_b)
            lengths.append((lo, hi))
        return lengths

    @property
    def min_lengths(self) -> list[int]:
        return [l[0] for l in self._get_min_max_lengths()]

    @property
    def max_lengths(self) -> list[int]:
        return [l[1] for l in self._get_min_max_lengths()]

    def fully_shard(self, **kwds: Any) -> None:
        for i in range(self.num_structures):
            fully_shard([self.embeddings[i], self.filmers[i], self.patchers[i]], **kwds)
        self.backbone.fully_shard(**kwds)
        fully_shard(self, **kwds | {"reshard_after_forward": False})
