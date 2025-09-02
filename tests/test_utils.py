"""
Tests.
"""

import pytest
import torch

from src.utils import TensorError
from src.utils import check_tensor
from src.utils import get_optimal_num_workers
from src.utils import get_optimal_num_worker_threads


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


class TestOptimalNumWorkers:

    def test_gpu_zero(self) -> None:
        with pytest.raises(RuntimeError):
            get_optimal_num_workers(ncpu=0, ngpu=1)
        assert get_optimal_num_workers(ncpu=1, ngpu=0) == 0
        assert get_optimal_num_workers(ncpu=2, ngpu=0) == 1
        assert get_optimal_num_workers(ncpu=3, ngpu=0) == 2
        assert get_optimal_num_workers(ncpu=4, ngpu=0) == 3

    def test_gpu_one(self) -> None:
        with pytest.raises(RuntimeError):
            get_optimal_num_workers(ncpu=0, ngpu=1)
        assert get_optimal_num_workers(ncpu=1, ngpu=1) == 0
        assert get_optimal_num_workers(ncpu=2, ngpu=1) == 1
        assert get_optimal_num_workers(ncpu=3, ngpu=1) == 2
        assert get_optimal_num_workers(ncpu=4, ngpu=1) == 3

    def test_gpu_two(self) -> None:
        with pytest.raises(RuntimeError):
            get_optimal_num_workers(ncpu=0, ngpu=2)
        with pytest.raises(RuntimeError):
            get_optimal_num_workers(ncpu=1, ngpu=2)
        assert get_optimal_num_workers(ncpu=2, ngpu=2) == 0
        assert get_optimal_num_workers(ncpu=3, ngpu=2) == 0
        assert get_optimal_num_workers(ncpu=4, ngpu=2) == 1


class TestOptimalNumWorkerThreads:

    def test_gpu_zero(self) -> None:
        assert get_optimal_num_worker_threads(num_workers=0, ncpu=8, ngpu=0) == 8
        assert get_optimal_num_worker_threads(num_workers=1, ncpu=8, ngpu=0) == 7
        assert get_optimal_num_worker_threads(num_workers=2, ncpu=8, ngpu=0) == 3
        assert get_optimal_num_worker_threads(num_workers=3, ncpu=8, ngpu=0) == 2
        assert get_optimal_num_worker_threads(num_workers=4, ncpu=8, ngpu=0) == 1
        assert get_optimal_num_worker_threads(num_workers=5, ncpu=8, ngpu=0) == 1
        assert get_optimal_num_worker_threads(num_workers=6, ncpu=8, ngpu=0) == 1
        assert get_optimal_num_worker_threads(num_workers=7, ncpu=8, ngpu=0) == 1

    def test_gpu_one(self) -> None:
        assert get_optimal_num_worker_threads(num_workers=0, ncpu=8, ngpu=1) == 8
        assert get_optimal_num_worker_threads(num_workers=1, ncpu=8, ngpu=1) == 7
        assert get_optimal_num_worker_threads(num_workers=2, ncpu=8, ngpu=1) == 3
        assert get_optimal_num_worker_threads(num_workers=3, ncpu=8, ngpu=1) == 2
        assert get_optimal_num_worker_threads(num_workers=4, ncpu=8, ngpu=1) == 1
        assert get_optimal_num_worker_threads(num_workers=5, ncpu=8, ngpu=1) == 1
        assert get_optimal_num_worker_threads(num_workers=6, ncpu=8, ngpu=1) == 1
        assert get_optimal_num_worker_threads(num_workers=7, ncpu=8, ngpu=1) == 1

    def test_gpu_two(self) -> None:
        assert get_optimal_num_worker_threads(num_workers=0, ncpu=8, ngpu=2) == 4
        assert get_optimal_num_worker_threads(num_workers=1, ncpu=8, ngpu=2) == 3
        assert get_optimal_num_worker_threads(num_workers=2, ncpu=8, ngpu=2) == 1
        assert get_optimal_num_worker_threads(num_workers=3, ncpu=8, ngpu=2) == 1
        with pytest.raises(RuntimeError):
            assert get_optimal_num_worker_threads(num_workers=4, ncpu=8, ngpu=2)
