"""
Sliding window entropy calculations for binary data.
"""

import os
from enum import Enum

import numpy as np
from numpy import typing as npt
import numba as nb
import torch
from torch import Tensor
from torch import BoolTensor
from torch import ByteTensor
from torch import HalfTensor
from torch import FloatTensor
from torch import DoubleTensor
from torch import LongTensor


AnyFloatTensor = HalfTensor | FloatTensor | DoubleTensor
AnyFloatArray  = npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]


TORCH_JIT_KWDS = {
    "fullgraph": False,
    "dynamic": False,
}

NUMBA_JIT_KWDS = {
    "nopython": True,
    "cache": True,
    "nogil": True,
    "parallel": True,
}


class Backend(Enum):
    """
    Enum for specifying the backend to use for entropy calculations.
    """
    NUMPY = "numpy"
    NUMBA = "numba"
    TORCH = "torch"


def is_integer_tensor(t: Tensor) -> bool:
    return t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def isfinite(x: np.ndarray | Tensor) -> BoolTensor | npt.NDArray[np.bool_]:  # type: ignore[type-arg]
    if isinstance(x, Tensor):
        return torch.isfinite(x)
    elif isinstance(x, np.ndarray):
        return np.isfinite(x)
    raise TypeError(f"Unsupported type: {type(x)}. Expected Tensor or ndarray.")


def isnan(x: np.ndarray | Tensor) -> BoolTensor | npt.NDArray[np.bool_]:  # type: ignore[type-arg]
    if isinstance(x, Tensor):
        return torch.isnan(x)
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    raise TypeError(f"Unsupported type: {type(x)}. Expected Tensor or ndarray.")


def _get_numpy_array_from_input(input: str | os.PathLike[str] | bytes | memoryview | np.ndarray | torch.Tensor) -> npt.NDArray[np.uint8]:  # type: ignore[type-arg]
    x: npt.NDArray[np.uint8]

    if isinstance(input, (str, os.PathLike)):
        x = np.fromfile(input, dtype=np.uint8)  # type: ignore[no-untyped-call]
    elif isinstance(input, (bytes, memoryview)):
        x = np.frombuffer(input, dtype=np.uint8)  # type: ignore[no-untyped-call]
    elif isinstance(input, torch.Tensor) and is_integer_tensor(input) and input.ndim == 1:
        x = input.numpy(force=True).astype(np.uint8)
    elif isinstance(input, np.ndarray) and np.issubdtype(input.dtype, np.integer) and input.ndim == 1:
        x = input.astype(np.uint8)
    else:
        raise TypeError(
            f"Unsupported input type: {type(input)}"
            f"{' or shape/dtype ' + str(tuple(input.shape)) + ' ' + str(input.dtype) if isinstance(input, (np.ndarray, Tensor)) else ''}"
            "."
        )

    if x.ndim != 1:
        raise ValueError(f"Unsupported input shape: {x.shape}. Expected 1D array.")

    return x


def _get_torch_array_from_input(input: str | os.PathLike[str] | bytes | memoryview | np.ndarray | torch.Tensor) -> ByteTensor:  # type: ignore[type-arg]
    x: ByteTensor

    if isinstance(input, (str, os.PathLike)):
        x = torch.from_file(input, shared=False, dtype=torch.uint8, size=os.path.getsize(input))
    elif isinstance(input, (bytes, memoryview)):
        x = torch.frombuffer(input, dtype=torch.uint8)
    elif isinstance(input, np.ndarray) and np.issubdtype(input.dtype, np.integer) and input.ndim == 1:
        x = torch.from_numpy(input).to(torch.uint8)
    elif isinstance(input, Tensor) and is_integer_tensor(input) and input.ndim == 1:
        x = input.to(torch.uint8)
    else:
        raise TypeError(
            f"Unsupported input type: {type(input)}"
            f"{' or shape/dtype ' + str(tuple(input.shape)) + ' ' + str(input.dtype) if isinstance(input, (np.ndarray, Tensor)) else ''}"
            "."
        )

    if x.ndim != 1:
        raise ValueError(f"Unsupported input shape: {x.shape}. Expected 1D array.")

    return x


def compute_entropy(
    input: str | os.PathLike[str] | bytes | memoryview | np.ndarray | torch.Tensor,  # type: ignore[type-arg]
    backend: str | Backend,
    *,
    radius: int = 1,
    stride: int = 1,
) -> DoubleTensor | npt.NDArray[np.float64]:
    """
    Compute the sliding window entropy of the input data.

    Args:
        input: Input data representing raw bytes.
        measure: Type of entropy measure to compute.
        algorithm: Algorithm to use for entropy calculation.
        backend: Backend to use for computation.
        check: Whether to check the result for non-finite values.
        radius: Radius of the sliding window, i.e., the number of bytes in either
            direction that will contribute to a local entropy calculation.
        stride: Stride of the sliding window, i.e., the number of bytes to skip
            between consecutive entropy calculations.
        padding: Value to use for padding on either side of the input data.

    Returns:
        Entropy values as a tensor or array within a sliding window of each byte in the input data.
    """
    backend = Backend(backend) if isinstance(backend, str) else backend

    if backend == Backend.NUMPY:
        x = _get_numpy_array_from_input(input)
        h = compute_entropy_numpy(x, radius=radius, stride=stride)
    elif backend == Backend.NUMBA:
        x = _get_numpy_array_from_input(input)
        h = compute_entropy_numba(x, radius=radius, stride=stride)
    elif backend == Backend.TORCH:
        x = _get_torch_array_from_input(input)
        h = compute_entropy_torch(x, radius=radius, stride=stride)
    else:
        raise ValueError(f"Unsupported backend: {backend}.")

    return h


@nb.jit(**NUMBA_JIT_KWDS)
def _compute_discrete_entropy_from_counts_numpy(n: int, c: npt.NDArray[np.int64]) -> float:
    c = c[c > 0]
    c = c.astype(np.float64)
    h = np.log2(n) - (c * np.log2(c)).sum() / n
    return float(h)


@torch.compile(**TORCH_JIT_KWDS)
def _compute_discrete_entropy_from_counts_torch(n: int, c: LongTensor) -> float:
    c = c[c > 0]
    c = c.to(torch.float64)
    h = torch.log2(torch.tensor(n, dtype=torch.float64)) - (c * torch.log2(c)).sum() / n
    return float(h)


@nb.jit(**NUMBA_JIT_KWDS)
def _compute_discrete_entropy_numpy(x: npt.NDArray[np.uint8]) -> float:
    n = len(x)
    c = np.bincount(x, minlength=256)
    return _compute_discrete_entropy_from_counts_numpy(n, c)


@torch.compile(**TORCH_JIT_KWDS)
def _compute_discrete_entropy_torch(x: npt.NDArray[np.uint8]) -> float:
    n = len(x)
    c = torch.bincount(x, minlength=256)
    return _compute_discrete_entropy_from_counts_torch(n, c)


def _check_inputs(length: int, radius: int, stride: int) -> None:
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}.")
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got {stride}.")
    if length <= 2 * radius + 1:
        raise ValueError(f"Input data length {length} is too short for the specified radius {radius}.")
    if stride > 2 * radius + 1:
        raise ValueError(f"Stride {stride} is too large for the specified radius {radius}.")


@nb.jit(**NUMBA_JIT_KWDS)
def _compute_entropy_naive_numpy(x: npt.NDArray[np.uint8], r: int, s: int = 1) -> npt.NDArray[np.float64]:
    w = 2 * r + 1
    o = np.full((len(x),), float("nan"), dtype=np.float64)
    # i is the index of the middle byte in the current window.
    for i in range(r, len(x) - r, s):
        l = i - r
        u = i + r + 1
        if len(x[l:u]) != w:
            raise RuntimeError(f"Expected window length {w}, got {len(x[l:u])} at iteration {i}.")
        h = _compute_discrete_entropy_numpy(x[l:u])
        o[i:i + s] = h
    return o

def compute_entropy_naive_numpy(x: npt.NDArray[np.uint8], r: int, s: int = 1) -> npt.NDArray[np.float64]:
    _check_inputs(len(x), r, s)
    return _compute_entropy_naive_numpy(x, r, s)


@torch.compile(**TORCH_JIT_KWDS)
def compute_entropy_naive_torch(x: ByteTensor, r: int, s: int = 1) -> DoubleTensor:
    _check_inputs(len(x), r, s)
    w = 2 * r + 1
    o = torch.full((len(x),), float("nan"), dtype=torch.float64)
    # i is the index of the middle byte in the current window.
    for i in range(r, len(x) - r, s):
        l = i - r
        u = i + r + 1
        if len(x[l:u]) != w:
            raise RuntimeError(f"Expected window length {w}, got {len(x[l:u])} at iteration {i}.")
        h = _compute_discrete_entropy_torch(x[l:u])
        o[i:i + s] = h
    return o


@nb.jit(**NUMBA_JIT_KWDS)
def _compute_entropy_rolling_numpy(x: npt.NDArray[np.uint8], r: int, s: int = 1) -> npt.NDArray[np.float64]:
    w = 2 * r + 1
    o = np.full((len(x),), float("nan"), dtype=np.float64)
    c = np.bincount(x[0:w-s], minlength=256)
    # i is one above the index of the last byte in the current window.
    for i in range(w, len(x) + 1, s):
        for j in range(i - s, i):
            c[x[j]] += 1
        for j in range(max(i - s - w, 0), i - w):
            c[x[j]] -= 1
        if len(np.nonzero(c)[0]) > w:
            raise RuntimeError(f"Expected at most {w} unique values in the window, got {len(np.nonzero(c)[0])} at iteration {i}.")
        if np.sum(c) != w:
            raise RuntimeError(f"Expected sum of counts to be {w}, got {np.sum(c)} at iteration {i}.")
        l = i - r - 1
        u = i - r + s - 1
        h = _compute_discrete_entropy_from_counts_numpy(w, c)
        o[l:u] = h
    return o

def compute_entropy_rolling_numpy(x: npt.NDArray[np.uint8], r: int, s: int = 1) -> npt.NDArray[np.float64]:
    _check_inputs(len(x), r, s)
    return _compute_entropy_rolling_numpy(x, r, s)


@torch.compile(**TORCH_JIT_KWDS)
def compute_entropy_rolling_torch(x: ByteTensor, r: int, s: int = 1) -> DoubleTensor:
    _check_inputs(len(x), r, s)
    w = 2 * r + 1
    o = torch.full((len(x),), float("nan"), dtype=torch.float64)
    c = torch.bincount(x[0:w-s], minlength=256)
    # i is one above the index of the last byte in the current window.
    for i in range(w, len(x) + 1, s):
        for j in range(i - s, i):
            c[x[j].item()] += 1
        for j in range(max(i - s - w, 0), i - w):
            c[x[j].item()] -= 1
        print(c)
        if len(torch.nonzero(c, as_tuple=True)[0]) > w:
            raise RuntimeError(f"Expected at most {w} unique values in the window, got {len(torch.nonzero(c, as_tuple=True)[0])} at iteration {i}. {c}")
        if torch.sum(c) != w:
            raise RuntimeError(f"Expected sum of counts to be {w}, got {torch.sum(c)} at iteration {i}.")
        l = i - r - 1
        u = i - r + s - 1
        h = _compute_discrete_entropy_from_counts_torch(w, c)
        o[l:u] = h
    return o
