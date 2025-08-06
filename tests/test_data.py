"""
Tests.
"""

import random
from pathlib import Path
import tempfile
from typing import Optional

import pytest
import torch
from torch import LongTensor

from src.data import BinaryDataset
from src.data import IterableBinaryDataset
from src.data import IterableBinaryDatasetBatchedLoader
from src.data import MapBinaryDataset
from src.data import MapBinaryDatasetBatchedLoader
from src.data import MapBinaryDatasetMemoryMapped


def get_random_bytes(size: int) -> bytes:
    return bytes([random.randint(0, 255) for _ in range(size)])


def get_random_str(size: int) -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=size))


class TestDataset:

    num_samples = 100
    labels = [random.randint(0, 1) for _ in range(num_samples)]

    def check_dataset(self, dataset: BinaryDataset, files: list[Path], buffers: list[bytes], max_length: Optional[int]):
        assert len(dataset) == self.num_samples
        for i, (n, x, y) in enumerate(dataset):
            assert isinstance(n, str)
            assert isinstance(x, LongTensor)
            assert isinstance(y, LongTensor)
            assert len(n) == 64
            assert tuple(x.shape) == (min(len(buffers[i]), max_length) if max_length is not None else len(buffers[i]),)
            assert tuple(y.shape) == ()
            assert n == files[i].name.split(".")[0]
            assert torch.equal(x, torch.frombuffer(buffers[i][0:max_length], dtype=torch.uint8).to(torch.long))
            assert torch.equal(y, torch.tensor(self.labels[i], dtype=torch.long))

    def populate(self, tmpdir: Path) -> tuple[list[Path], list[int]]:
        files = []
        buffers = []
        for _ in range(self.num_samples):
            b = get_random_bytes(random.randint(512, 2048))
            f = tmpdir / f"{get_random_str(64)}.bin"
            f.write_bytes(b)
            files.append(f)
            buffers.append(b)
        return files, buffers

    @pytest.mark.parametrize("max_length", [None, 512, 1024, 2048])
    def test_map_binary_dataset(self, max_length: Optional[int]):
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = MapBinaryDataset(files, self.labels, max_length)
            self.check_dataset(dataset, files, buffers, max_length)

    @pytest.mark.parametrize("max_length", [None, 512, 1024, 2048])
    def test_map_binary_dataset_batched_loader(self, max_length: Optional[int]):
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = MapBinaryDatasetBatchedLoader(files, self.labels, max_length)
            self.check_dataset(dataset, files, buffers, max_length)

    @pytest.mark.parametrize("max_length", [None, 512, 1024, 2048])
    def test_map_binary_dataset_memory_mapped(self, max_length: Optional[int]):
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = MapBinaryDatasetMemoryMapped(files, self.labels, max_length)
            self.check_dataset(dataset, files, buffers, max_length)

    @pytest.mark.parametrize("max_length", [None, 512, 1024, 2048])
    def test_iterable_binary_dataset(self, max_length: Optional[int]):
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = IterableBinaryDataset(files, self.labels, max_length)
            self.check_dataset(list(dataset), files, buffers, max_length)

    @pytest.mark.parametrize("max_length", [None, 512, 1024, 2048])
    def test_iterable_binary_dataset_batched_loader(self, max_length: Optional[int]):
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = IterableBinaryDatasetBatchedLoader(files, self.labels, max_length)
            self.check_dataset(list(dataset), files, buffers, max_length)
