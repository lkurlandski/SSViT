"""
Tests.
"""

import pytest
import torch

from src.utils import TensorError
from src.utils import check_tensor



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
