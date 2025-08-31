"""
Utilities.
"""

from __future__ import annotations
import math
import os
import random
from typing import Literal
from typing import NamedTuple
from typing import Optional
import warnings

import numpy as np
import psutil
import torch
from torch import Tensor
from torch import BoolTensor
from torch import ByteTensor
from torch import IntTensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as _pad_sequence


def str_to_bool(v: bool | str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False
    raise ValueError(f"Boolean value expected, got {v!r}.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_optimal_num_workers(ncpu: int = psutil.cpu_count(logical=False), ngpu: int = torch.cuda.device_count()) -> int:
    # TODO: its unclear how the CPU check will behave with SLURM and torchrun.
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if ngpu > ncpu:
        raise RuntimeError(f"Number of GPUs ({ngpu}) exceeds number of CPU cores ({ncpu}).")
    return max(0, ncpu // max(1, ngpu) - 1)


def get_optimal_num_worker_threads(num_workers: int = 0, ncpu: int = psutil.cpu_count(logical=False)) -> int:
    # TODO: its unclear how the CPU check will behave with SLURM and torchrun.
    if ncpu <= 0:
        raise RuntimeError(f"Number of CPU cores ({ncpu}) must be greater than 0.")
    if num_workers - 1 > ncpu:
        raise RuntimeError(f"Number of worker processes ({num_workers} + 1) exceeds number of CPU cores ({ncpu}).")
    if num_workers == 0:
        return ncpu
    return max(1, (ncpu - 1) // num_workers)


class TensorError(ValueError):
    """
    Exception raised when the shape of a tensor does not match the expected shape.
    """

    def __init__(self, x: Tensor, s: Optional[tuple[Optional[int], ...]], t: Optional[torch.dtype]) -> None:
        super().__init__(
            f"Expected tensor with dtype {t} and shape {s}. Got tensor with dtype {x.dtype} and shape {tuple(x.shape)}."
        )


def check_tensor(x: Tensor, s: Optional[tuple[Optional[int], ...]] = None, t: Optional[torch.dtype | tuple[torch.dtype]] = None) -> None:
    if t is not None and isinstance(t, torch.dtype):
        t = (t,)
    if t is not None and x.dtype not in t:
        raise TensorError(x, s, t)
    if s is not None and len(x.shape) != len(s):
        raise TensorError(x, s, t)
    if s is not None:
        for i, j in zip(x.shape, s, strict=True):
            if j is not None and i != j:
                raise TensorError(x, s, t)


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float | int | bool = 0.0,
    padding_side: Literal["right", "left"] = "right",
    pin_memory: bool = False,
    pad_to_multiple_of: int = 1,
    min_length: int = 0,
) -> Tensor:

    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty list of sequences.")
    if pad_to_multiple_of < 1:
        raise ValueError(f"pad_to_multiple_of must be a positive integer. Got {pad_to_multiple_of}.")

    if not pin_memory and pad_to_multiple_of == 1 and min_length == 0:
        return _pad_sequence(sequences, batch_first, padding_value, padding_side)

    if padding_side != "right":
        raise NotImplementedError("pad_sequence with pin_memory=True requires padding_side='right'.")
    if not batch_first:
        raise NotImplementedError("pad_sequence with pin_memory=True requires batch_first=True.")

    for s in sequences:
        if pin_memory and s.device.type != "cpu":
            raise ValueError("All sequences must be on CPU when pin_memory=True.")
        if s.shape[1:] != sequences[0].shape[1:]:
            raise ValueError("All sequences must have the same shape except for the first dimension.")
        if s.dtype != sequences[0].dtype:
            raise ValueError("All sequences must have the same dtype.")
        if s.device != sequences[0].device:
            raise ValueError("All sequences must be on the same device.")

    batch_size = len(sequences)
    seq_length = max(min_length, math.ceil(max(s.shape[0] for s in sequences) / pad_to_multiple_of) * pad_to_multiple_of)
    other_dims = sequences[0].shape[1:]
    size = (batch_size, seq_length) + tuple(other_dims)

    padded = torch.full(size, fill_value=padding_value, dtype=sequences[0].dtype, device=sequences[0].device, pin_memory=pin_memory)
    for i, s in enumerate(sequences):
        s = s.contiguous() if not s.is_contiguous() else s
        padded[i, :s.shape[0]].copy_(s)
    return padded
