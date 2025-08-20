"""
Tests.
"""

import pytest
import numpy as np
import torch

from src.utils import TensorError
from src.utils import check_tensor
from src.utils import pack_bool_tensor
from src.utils import unpack_bit_tensor
from src.utils import smallest_unsigned_integer_dtype


class TestTensorError:
    def test_check_good(self):
        check_tensor(torch.randn(4, 5), (None, 5), torch.float)
        check_tensor(torch.randn(4, 5), (None, None), None)
        check_tensor(torch.randn(4, 5), (4, 5), torch.float)

    def test_check_bad_dtype(self):
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, 5), torch.int)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, None), torch.int)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (4, 5), torch.int)

    def test_check_bad_shape(self):
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None, 3), torch.float)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (None,), None)
        with pytest.raises(TensorError):
            check_tensor(torch.randn(4, 5), (4, 5, None), torch.float)


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
def test_pack_bool_tensor(ndim: int, batch_size: int, num_channels: int):
    size = (batch_size, num_channels)
    for k in range(2, ndim):
        size = (k,) + size
    b = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b)
    assert x.shape == size[:-1]
    assert x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
    assert x.device == b.device


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("batch_size", list(range(10)))
@pytest.mark.parametrize("num_channels", list(range(1, 65)))
@pytest.mark.parametrize("otype", [torch.bool, torch.int8, torch.uint8, torch.uint16, torch.int16, torch.int32, torch.float16, torch.float32])
def test_unpack_bit_tensor(ndim: int, batch_size: int, num_channels: int, otype: torch.dtype):
    dtype = smallest_unsigned_integer_dtype(num_channels)
    size = (batch_size,)
    for k in range(2, ndim):
        size = (k,) + size
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
def test_pack_then_unpack_tensor(ndim: int, batch_size: int, num_channels: int):
    size = (batch_size, num_channels)
    for k in range(2, ndim):
        size = (k,) + size
    b_1 = torch.randint(0, 2, size, dtype=torch.bool)
    x = pack_bool_tensor(b_1)
    b_2 = unpack_bit_tensor(x, num_channels)
    assert b_1.shape == b_2.shape
    assert b_1.dtype == b_2.dtype
    assert b_1.device == b_2.device
    assert torch.all(b_1 == b_2)
