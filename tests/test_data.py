"""
Tests.
"""

import math
import os
import random
from pathlib import Path
import tempfile
from typing import Any

import pytest
import torch
from torch import Tensor
from torch import LongTensor

from src.binanal import HierarchicalStructureNone
from src.data import Sample
from src.data import BinaryDataset
from src.data import SemanticGuider
from src.data import SemanticGuides
from src.data import StructureMap
from src.data import Preprocessor
from src.data import GroupedLengthBatchSampler

from tests import FILES


def get_random_bytes(size: int) -> bytes:
    return bytes([random.randint(0, 255) for _ in range(size)])


def get_random_str(size: int) -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=size))


class TestSemanticGuider:

    @pytest.mark.parametrize("do_parse", [False, True])
    @pytest.mark.parametrize("do_entropy", [False, True])
    @pytest.mark.parametrize("do_characteristics", [False, True])
    def test(self, do_parse: bool, do_entropy: bool, do_characteristics: bool) -> None:
        file = FILES[0]
        b = file.read_bytes()
        guider = SemanticGuider(do_parse=do_parse, do_entropy=do_entropy, do_characteristics=do_characteristics)
        sample = guider(b, inputs=torch.frombuffer(b, dtype=torch.uint8))
        assert bool(sample.parse is None) != do_parse
        assert bool(sample.entropy is None) != do_entropy
        assert bool(sample.characteristics is None) != do_characteristics


class TestBinaryDataset:

    num_samples = 100
    labels = [random.randint(0, 1) for _ in range(num_samples)]

    def check_dataset(self, dataset: BinaryDataset, files: list[Path], buffers: list[bytes]) -> None:
        assert len(dataset) == self.num_samples
        for i, sample in enumerate(dataset):
            n = sample.name
            x = sample.inputs
            y = sample.label
            assert isinstance(n, str)
            assert isinstance(x, Tensor)
            assert isinstance(y, Tensor)
            assert len(n) == 64
            assert tuple(x.shape) == (len(buffers[i]),)
            assert tuple(y.shape) == ()
            assert n == files[i].name.split(".")[0]
            assert torch.equal(x, torch.frombuffer(buffers[i], dtype=torch.uint8).to(torch.long))
            assert torch.equal(y, torch.tensor(self.labels[i], dtype=torch.long))

    def populate(self, tmpdir: Path) -> tuple[list[Path], list[bytes]]:
        files = []
        buffers = []
        for _ in range(self.num_samples):
            b = get_random_bytes(random.randint(512, 2048))
            f = tmpdir / f"{get_random_bytes(32).hex()}"
            f.write_bytes(b)
            files.append(f)
            buffers.append(b)
        return files, buffers

    def test_map_binary_dataset(self) -> None:

        class PreprocessorMock(Preprocessor):

            def __call__(self, file: Path, label: int) -> tuple[LongTensor, SemanticGuides, StructureMap]:
                b = file.read_bytes()
                inputs = torch.frombuffer(b, dtype=torch.uint8).to(torch.long)
                guides = SemanticGuides(None, None, None)
                structure = StructureMap(torch.full((len(b), 1), False), {0: HierarchicalStructureNone.ANY})
                return Sample(file, file.stem, torch.tensor(label, dtype=torch.int32), inputs, guides, structure)

        preprocessor = PreprocessorMock(False, False, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = BinaryDataset(files, self.labels, preprocessor)
            self.check_dataset(dataset, files, buffers)


CHUNKS_A = [[0, 1], [2, 3], [4, 5]]
CHUNKS_B = [[0, 1], [2, 3], [6], [4, 5]]
CHUNKS_C = [[0, 1], [2, 3], [4, 5], [6], [7, 8, 9]]

class TestGroupedLengthBatchSampler:

    @pytest.mark.parametrize("chunks", [CHUNKS_A, CHUNKS_B, CHUNKS_C])
    def test_basic(self, chunks: list[list[int]]) -> None:
        batchsampler = GroupedLengthBatchSampler(chunks, False, False, False)
        assert sorted(list(batchsampler)) == sorted(chunks)

    @pytest.mark.parametrize("chunks", [CHUNKS_A, CHUNKS_B, CHUNKS_C])
    def test_first(self, chunks: list[list[int]]) -> None:
        for _ in range(100):
            batchsampler = GroupedLengthBatchSampler(chunks, True, False, False)
            assert next(iter(batchsampler)) == chunks[0]

        for _ in range(100):
            batchsampler = GroupedLengthBatchSampler(chunks, False, False, False)
            if next(iter(batchsampler)) != chunks[0]:
                break
        else:
            raise AssertionError("First chunk was always emitted first when first=True")

    @pytest.mark.parametrize("chunks", [CHUNKS_A, CHUNKS_B, CHUNKS_C])
    def test_shuffle(self, chunks: list[list[int]]) -> None:

        batchsampler = GroupedLengthBatchSampler(chunks, False, False, False)
        prv = list(batchsampler)
        for _ in range(100):
            nxt = list(batchsampler)
            assert prv == nxt, "Order should be stable when shuffle=False"
            prv = nxt

        batchsampler = GroupedLengthBatchSampler(chunks, False, True, False)
        for _ in range(100):
            nxt = list(batchsampler)
            if prv != nxt:
                break
            prv = nxt
        else:
            raise AssertionError("Order was not changed when shuffle=True")

    def test_drop_last_a(self) -> None:
        chunks = [[0, 1], [2, 3], [4, 5]]
        batchsampler = GroupedLengthBatchSampler(chunks, False, False, True)
        assert sorted(list(batchsampler)) == sorted(chunks)

    def test_drop_last_b(self) -> None:
        chunks = [[0, 1], [2, 3], [6], [4, 5]]
        batchsampler = GroupedLengthBatchSampler(chunks, False, False, True)
        assert sorted(list(batchsampler)) == sorted([c for c in chunks if len(c) > 1])

    def test_drop_last_c(self) -> None:
        chunks = [[0, 1], [2, 3], [4, 5], [6], [7, 8, 9]]
        with pytest.raises(RuntimeError):
            GroupedLengthBatchSampler(chunks, False, False, True)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_from_lengths(self, batch_size: int) -> None:
        lengths = list(range(101))
        random.shuffle(lengths)
        batchsampler = GroupedLengthBatchSampler.from_lengths(batch_size, lengths, first=True, shuffle=False, drop_last=False)
        batches = list(batchsampler)
        assert len(batches) == math.ceil(len(lengths) / batch_size) == len(batchsampler)
        got_smaller = False
        for i, batch in enumerate(batches):
            assert len(batch) <= batch_size
            assert len(batch) == batch_size or not got_smaller
            got_smaller = len(batch) < batch_size
            longest = max(lengths[i] for i in batch)
            shortest = min(lengths[i] for i in batch)
            assert longest - shortest < batch_size, "Batch should contain similar lengths."
