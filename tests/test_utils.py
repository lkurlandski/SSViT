"""
Tests.
"""

import math
import pytest
import numpy as np
import torch

from src.utils import TensorError
from src.utils import check_tensor
from src.utils import pack_bool_tensor
from src.utils import unpack_bit_tensor
from src.utils import smallest_unsigned_integer_dtype
from src.utils import packbits
from src.utils import unpackbits


class TestTensorError:

    def test_check_good(self) -> None:
        check_tensor(torch.randn(4, 5), (None, 5), torch.float)
        check_tensor(torch.randn(4, 5), (None, None), None)
        check_tensor(torch.randn(4, 5), (4, 5), torch.float)

    def test_check_bad_dtype(self) -> None:
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, 5), torch.int)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, None), torch.int)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (4, 5), torch.int)

    def test_check_bad_shape(self) -> None:
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, 3), torch.float)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None,), None)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (4, 5, None), torch.float)


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
def test_pack_bool_tensor(ndim: int, batch_size: int, num_channels: int) -> None:
    size = [batch_size, num_channels]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    b = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b)
    assert x.shape == size[:-1]
    assert x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    assert x.device == b.device


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
@pytest.mark.parametrize("otype", [torch.bool, torch.int8, torch.uint8, torch.uint16, torch.int16, torch.int32, torch.float16, torch.float32])
def test_unpack_bit_tensor(ndim: int, batch_size: int, num_channels: int, otype: torch.dtype) -> None:
    dtype = smallest_unsigned_integer_dtype(num_channels)
    size = [batch_size]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    x = torch.from_numpy(np.random.randint(0, torch.iinfo(dtype).max, size=size, dtype=np.uint64)).to(dtype)
    b = unpack_bit_tensor(x, num_channels)
    assert b.shape == size + (num_channels,)
    assert b.dtype == torch.bool
    assert b.device == x.device
    y = unpack_bit_tensor(x, num_channels, dtype=otype)
    assert y.shape == size + (num_channels,)
    assert y.dtype == otype
    assert y.device == x.device
    assert torch.all(y == b.to(otype))


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
def test_pack_then_unpack_tensor(ndim: int, batch_size: int, num_channels: int) -> None:
    size = [batch_size, num_channels]
    for k in range(2, ndim):
        size.insert(0, k)
    size = tuple(size)
    b_1 = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b_1)
    b_2 = unpack_bit_tensor(x, num_channels)
    assert b_1.shape == b_2.shape
    assert b_1.dtype == b_2.dtype
    assert b_1.device == b_2.device
    assert torch.all(b_1 == b_2)


@pytest.mark.parametrize("a", [1, 2, 7, 8, 9])
@pytest.mark.parametrize("b", [1, 2, 15, 16, 17])
@pytest.mark.parametrize("c", [1, 2, 31, 32, 33])
@pytest.mark.parametrize("d", [1, 2, 63, 64, 65])
@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1])
def test_packunpackbits(a: int, b: int, c: int, d: int, axis: int) -> None:
    print(f"{a=} {b=} {c=} {d=} {axis=}")
    if all(t == 1 for t in (a, b, c, d)):
        return

    x = torch.randint(0, 2, (a, b, c, d), dtype=torch.bool).squeeze()
    axis = min(axis, x.ndim - 1)
    axis_ = axis if axis >= 0 else x.ndim + axis
    print(f"{axis=} {axis_=}")
    print(f"{x.shape=}")

    y = packbits(x, axis)
    print(f"{y.shape=}")
    assert y.dtype == torch.uint8
    assert y.shape == x.shape[:axis_] + (x.shape[axis] // 8 + (x.shape[axis] % 8 > 0),) + x.shape[axis_ + 1:]

    z = unpackbits(y, x.shape[axis], axis=axis)
    print(f"{z.shape=}")
    assert z.dtype == torch.bool
    assert z.shape == x.shape
    assert torch.all(x == z)

    w = unpackbits(y, axis=axis)
    print(f"{w.shape=}")
    assert w.dtype == torch.bool
    assert w.shape == x.shape[:axis_] + (math.ceil(x.shape[axis] / 8) * 8,) + x.shape[axis_ + 1:]
    w_ = w.movedim(axis, 0)[0:x.shape[axis]].movedim(0, axis)
    print(f"{w_.shape=}")
    assert torch.all(x == w_)
