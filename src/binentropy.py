"""
Sliding window entropy calculations for binary data.
"""

import os
from enum import Enum
from typing import Any

import numpy as np
from numpy import typing as npt
import numba as nb
import torch
from torch import Tensor


TORCH_JIT_KWDS: dict[str, Any] = {
    "fullgraph": False,
    "dynamic": False,
}

NUMBA_JIT_KWDS: dict[str, Any] = {
    "nopython": True,
    "cache": True,
    "nogil": True,
    "parallel": False,
    "fastmath": True,
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


def isfinite(x: np.ndarray | Tensor) -> Tensor | npt.NDArray[np.bool_]:  # type: ignore[type-arg]
    if isinstance(x, Tensor):
        return torch.isfinite(x)
    elif isinstance(x, np.ndarray):
        return np.isfinite(x)  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported type: {type(x)}. Expected Tensor or ndarray.")


def isnan(x: np.ndarray | Tensor) -> Tensor | npt.NDArray[np.bool_]:  # type: ignore[type-arg]
    if isinstance(x, Tensor):
        return torch.isnan(x)
    elif isinstance(x, np.ndarray):
        return np.isnan(x)  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported type: {type(x)}. Expected Tensor or ndarray.")


def _get_numpy_array_from_input(input: str | os.PathLike[str] | bytes | memoryview | np.ndarray | torch.Tensor) -> npt.NDArray[np.uint8]:  # type: ignore[type-arg]
    x: npt.NDArray[np.uint8]

    if isinstance(input, (str, os.PathLike)):
        x = np.fromfile(input, dtype=np.uint8)
    elif isinstance(input, (bytes, memoryview)):
        x = np.frombuffer(input, dtype=np.uint8)
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


def _get_torch_array_from_input(input: str | os.PathLike[str] | bytes | memoryview | np.ndarray | torch.Tensor) -> Tensor:  # type: ignore[type-arg]
    x: Tensor

    if isinstance(input, (str, os.PathLike)):
        x = torch.from_file(str(input), shared=False, dtype=torch.uint8, size=os.path.getsize(input))
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


@nb.jit(**NUMBA_JIT_KWDS)  # type: ignore[misc]
def _compute_discrete_entropy_from_counts_numpy(n: int, c: npt.NDArray[np.int64]) -> float:
    c = c[c > 0]
    c = c.astype(np.float64)
    h = np.log2(n) - (c * np.log2(c)).sum() / n
    return float(h)


@torch.compile(**TORCH_JIT_KWDS)
def _compute_discrete_entropy_from_counts_torch(n: int, c: Tensor) -> float:
    c = c[c > 0]
    c = c.to(torch.float64)
    h = torch.log2(torch.tensor(n, dtype=torch.float64)) - (c * torch.log2(c)).sum() / n
    return float(h)


@nb.jit(**NUMBA_JIT_KWDS)  # type: ignore[misc]
def _compute_discrete_entropy_numpy(x: npt.NDArray[np.uint8]) -> float:
    n = len(x)
    c = np.bincount(x, minlength=256)
    h: float = _compute_discrete_entropy_from_counts_numpy(n, c)
    return h


@torch.compile(**TORCH_JIT_KWDS)
def _compute_discrete_entropy_torch(x: Tensor) -> float:
    if x.dtype != torch.uint8:
        raise TypeError(f"Expected torch.uint8 tensor, got {x.dtype}.")
    n = len(x)
    c = torch.bincount(x, minlength=256)
    h: float = _compute_discrete_entropy_from_counts_torch(n, c)
    return h


def _check_inputs(length: int, radius: int, stride: int, errors: str = "raise") -> bool:
    if radius <= 0:
        if errors == "raise":
            raise ValueError(f"Radius must be positive, got {radius}.")
        return False
    if stride <= 0:
        if errors == "raise":
            raise ValueError(f"Stride must be positive, got {stride}.")
        return False
    if length <= 2 * radius + 1:
        if errors == "raise":
            raise ValueError(f"Input data length {length} is too short for the specified radius {radius}.")
        return False
    if stride > 2 * radius + 1:
        if errors == "raise":
            raise ValueError(f"Stride {stride} is too large for the specified radius {radius}.")
        return False
    return True


@nb.jit(**NUMBA_JIT_KWDS)  # type: ignore[misc]
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
    h: npt.NDArray[np.float64] = _compute_entropy_naive_numpy(x, r, s)
    return h


@torch.compile(**TORCH_JIT_KWDS)
def compute_entropy_naive_torch(x: Tensor, r: int, s: int = 1) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError()
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


@nb.jit(**NUMBA_JIT_KWDS)  # type: ignore[misc]
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

@nb.jit(**NUMBA_JIT_KWDS)  # type: ignore[misc]
def _compute_entropy_rolling_numpy_fast(x: npt.NDArray[np.uint8], r: int) -> npt.NDArray[np.float64]:
    w = 2 * r + 1
    o = np.full((len(x)), float("nan"), np.float64)

    # precompute logs once
    log_tbl = np.empty(w + 1, np.float64)
    log_tbl[0] = 0.0
    for k in range(1, w + 1):
        log_tbl[k] = np.log2(k)
    logW = log_tbl[w]

    counts = np.zeros(256, np.int32)

    # initial histogram over [0:W)
    for i in range(w):
        counts[int(x[i])] += 1

    # initial s_clogc = sum c*log(c)
    s_clogc = 0.0
    for v in range(256):
        c = counts[v]
        if c:
            s_clogc += c * log_tbl[c]

    # place entropy at center of first window
    o[r] = logW - (s_clogc / w)

    # slide one-by-one
    for i in range(w, len(x)):
        v_out = int(x[i - w])
        v_in  = int(x[i])
        if v_out != v_in:
            co = counts[v_out]
            ci = counts[v_in]
            # remove old contributions
            s_clogc -= co * log_tbl[co]
            s_clogc -= ci * log_tbl[ci]
            # update counts
            co -= 1; ci += 1
            counts[v_out] = co
            counts[v_in]  = ci
            # add new contributions
            s_clogc += co * log_tbl[co]
            s_clogc += ci * log_tbl[ci]
        o[i - r] = logW - (s_clogc / w)

    return o

def compute_entropy_rolling_numpy(x: npt.NDArray[np.uint8], r: int, s: int = 1) -> npt.NDArray[np.float64]:
    _check_inputs(len(x), r, s)
    h: npt.NDArray[np.float64] = _compute_entropy_rolling_numpy_fast(x, r) if s == 1 else _compute_entropy_rolling_numpy(x, r, s)
    return h


def _int_(x: float | int) -> int:
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"Expected integer value, got float {x}.")
        return int(x)
    return x


@torch.compile(**TORCH_JIT_KWDS)
def compute_entropy_rolling_torch(x: Tensor, r: int, s: int = 1) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError()
    _check_inputs(len(x), r, s)
    w = 2 * r + 1
    o = torch.full((len(x),), float("nan"), dtype=torch.float64)
    c = torch.bincount(x[0:w-s], minlength=256)
    # i is one above the index of the last byte in the current window.
    for i in range(w, len(x) + 1, s):
        for j in range(i - s, i):
            c[_int_(x[j].item())] += 1
        for j in range(max(i - s - w, 0), i - w):
            c[_int_(x[j].item())] -= 1
        if len(torch.nonzero(c, as_tuple=True)[0]) > w:
            raise RuntimeError(f"Expected at most {w} unique values in the window, got {len(torch.nonzero(c, as_tuple=True)[0])} at iteration {i}. {c}")
        if torch.sum(c) != w:
            raise RuntimeError(f"Expected sum of counts to be {w}, got {torch.sum(c)} at iteration {i}.")
        l = i - r - 1
        u = i - r + s - 1
        h = _compute_discrete_entropy_from_counts_torch(w, c)
        o[l:u] = h
    return o
