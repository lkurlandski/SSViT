"""
Tests.
"""

from pprint import pformat
from typing import Callable

import numpy as np
from numpy import typing as npt
import pytest
import torch
from torch import DoubleTensor

from src.binentropy import compute_entropy_naive_numpy
from src.binentropy import compute_entropy_rolling_numpy
from src.binentropy import compute_entropy_naive_torch
from src.binentropy import compute_entropy_rolling_torch
from src.binentropy import _check_inputs


# ---- helpers ---- #

def pformat_isclose(a: np.ndarray, b: np.ndarray, equal_nan: bool = True) -> str:  # type: ignore[type-arg]
    e = np.isclose(a, b, equal_nan=equal_nan)
    a = a.tolist()
    b = b.tolist()
    e = list(map(lambda x: "==" if x else "!=", e.tolist()))
    return pformat(list(zip(e, a, b)))


def check_entropy(h: npt.NDArray[np.float64], size: int, radius: int, stride: int) -> None:
    assert len(h) == size
    nan_lo = radius
    nan_up = size - radius + stride
    assert np.isnan(h[:nan_lo]).all()
    assert np.isnan(h[nan_up:]).all()
    assert np.all(np.isfinite(h[radius:-radius])), pformat(np.isfinite(h[radius:-radius]).tolist())
    assert np.all(h[radius:-radius] >= 0)
    assert np.all(h[radius:-radius] <= np.log2(256))
    idx = np.arange(radius, size - radius - stride + 1, stride)
    for i in range(stride):
        assert np.isclose(h[idx], h[idx + i], equal_nan=True).all(), pformat(np.isclose(h[idx], h[idx + i], equal_nan=True).tolist())

# ---- tests ---- #

SIZE = [15, 16, 17, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025]
RADIUS = [1, 2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 257]
STRIDE = [1, 2, 3, 4, 5, 6, 7]

# ---- numpy ----- #

@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_naive_numpy(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = np.random.randint(0, 256, size=size, dtype=np.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_naive_numpy(b, radius, stride)
        return

    h = compute_entropy_naive_numpy(b, radius, stride)
    check_entropy(h, size, radius, stride)


@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_rolling_numpy(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = np.random.randint(0, 256, size=size, dtype=np.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_rolling_numpy(b, radius, stride)
        return

    h = compute_entropy_rolling_numpy(b, radius, stride)
    check_entropy(h, size, radius, stride)


@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_naive_and_rolling_numpy_equivalence(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = np.random.randint(0, 256, size=size, dtype=np.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_naive_numpy(b, radius, stride)
        with pytest.raises(ValueError):
            compute_entropy_rolling_numpy(b, radius, stride)
        return

    h_n = compute_entropy_naive_numpy(b, radius, stride)
    h_r = compute_entropy_rolling_numpy(b, radius, stride)
    check_entropy(h_n, size, radius, stride)
    check_entropy(h_r, size, radius, stride)
    assert np.isclose(h_n, h_r, equal_nan=True).all(), pformat_isclose(h_n, h_r, equal_nan=True)

# ---- torch ---- #

@pytest.mark.skip(reason="torch entropy functions are slow and have no intention of being used.")
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_naive_torch(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_naive_torch(b, radius, stride)
        return

    h = compute_entropy_naive_torch(b, radius, stride).numpy()
    check_entropy(h, size, radius, stride)


@pytest.mark.skip(reason="torch entropy functions are slow and have no intention of being used.")
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_rolling_torch(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_rolling_torch(b, radius, stride)
        return

    h = compute_entropy_rolling_torch(b, radius, stride).numpy()
    check_entropy(h, size, radius, stride)


@pytest.mark.skip(reason="torch entropy functions are slow and have no intention of being used.")
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_naive_and_rolling_torch_equivalence(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b = torch.randint(0, 256, size=(size,), dtype=torch.uint8)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_naive_torch(b, radius, stride)
        with pytest.raises(ValueError):
            compute_entropy_rolling_torch(b, radius, stride)
        return

    h_n = compute_entropy_naive_torch(b, radius, stride).numpy()
    h_r = compute_entropy_rolling_torch(b, radius, stride).numpy()
    check_entropy(h_n, size, radius, stride)
    check_entropy(h_r, size, radius, stride)
    assert np.isclose(h_n, h_r, equal_nan=True).all(), pformat_isclose(h_n, h_r, equal_nan=True)

# ---- numba ---- #


# ---- equiv ---- #

@pytest.mark.skip(reason="torch entropy functions are slow and have no intention of being used.")
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("radius", RADIUS)
@pytest.mark.parametrize("stride", STRIDE)
def test_compute_entropy_numpy_and_torch_equivalence(size: int, radius: int, stride: int) -> None:
    should_raise = False
    try:
        _check_inputs(size, radius, stride)
    except ValueError:
        should_raise = True

    b_n = np.random.randint(0, 256, size=size, dtype=np.uint8)
    b_t = torch.from_numpy(b_n)
    if should_raise:
        with pytest.raises(ValueError):
            compute_entropy_naive_numpy(b_n, radius, stride)
        with pytest.raises(ValueError):
            compute_entropy_naive_torch(b_t, radius, stride)
        return

    h_n = compute_entropy_naive_numpy(b_n, radius, stride)
    h_t = compute_entropy_rolling_torch(b_t, radius, stride).numpy()
    check_entropy(h_n, size, radius, stride)
    check_entropy(h_t, size, radius, stride)
    assert np.isclose(h_n, h_t, equal_nan=True).all(), pformat_isclose(h_n, h_t, equal_nan=True)

# ---- other ---- #

def test_check_inputs_raises() -> None:
    with pytest.raises(ValueError, match="Radius must be positive, got 0."):
        _check_inputs(10, 0, 1)
    with pytest.raises(ValueError, match="Stride must be positive, got 0."):
        _check_inputs(10, 1, 0)
    with pytest.raises(ValueError, match="Input data length 10 is too short for the specified radius 5."):
        _check_inputs(10, 5, 1)
    with pytest.raises(ValueError, match="Stride 10 is too large for the specified radius 4."):
        _check_inputs(10, 4, 10)
