"""
Fixtures.
"""

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires CUDA device(s)")


@pytest.fixture
def cuda0() -> torch.device:
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 1):
        pytest.skip("CUDA unavailable")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    return dev
