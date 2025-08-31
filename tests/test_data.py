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
from torch import ShortTensor
from torch import LongTensor

from src.binanal import HierarchicalStructureNone
from src.data import Name
from src.data import FSample
from src.data import FSamples
from src.data import HSample
from src.data import HSamples
from src.data import SemanticGuide
from src.data import SemanticGuides
from src.data import SemanticGuider
from src.data import StructureMap
from src.data import StructureMaps
from src.data import StructurePartitioner
from src.data import Preprocessor
from src.data import BinaryDataset
from src.data import GroupedLengthBatchSampler
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.bitpacking import packbits
from src.bitpacking import unpackbits

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


class TestCollateFn:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("seq_length", [15, 16, 17, 31, 32, 33, 63, 64, 65])
    @pytest.mark.parametrize("min_length", [0, 16, 32, 64, 128])
    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_basic(self, batch_size: int, seq_length: int, min_length: int, pin_memory: bool) -> None:
        print(f"TestCollateFn::test_basic: {batch_size=} {seq_length=} {min_length=} {pin_memory=}")

        collate_fn = CollateFn(pin_memory=pin_memory, bitpack=False, min_length=min_length)

        samples = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            guides = SemanticGuide(None, None, None)
            structure = StructureMap(torch.full((seq_length + i, 1), True), {0: HierarchicalStructureNone.ANY})
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        # Everything should be padded to the nearest multiple of 8.
        out_seq_length = max(min_length, seq_length + batch_size - 1)
        out_seq_length = math.ceil(out_seq_length / 8) * 8

        batch = collate_fn(samples)
        assert isinstance(batch, FSamples)

        assert isinstance(batch.file, list)
        assert len(batch.file) == batch_size
        for f_1, f_2 in zip(batch.file, (s.file for s in samples), strict=True):
            assert f_1 == f_2

        assert isinstance(batch.name, list)
        assert len(batch.name) == batch_size
        for n_1, n_2 in zip(batch.name, (s.name for s in samples), strict=True):
            assert n_1 == n_2

        assert isinstance(batch.label, LongTensor)
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, ShortTensor)
        assert batch.inputs.shape == (batch_size, out_seq_length)
        assert batch.inputs.is_pinned() == pin_memory
        # The inputs should be incremented by 1 to reserve 0 for padding.
        for i_1, i_2 in zip(batch.inputs, (s.inputs for s in samples), strict=True):
            assert torch.equal(i_1[:i_2.shape[0]].to(torch.float64), i_2.to(torch.float64) + 1)

        assert isinstance(batch.guides, SemanticGuides)
        assert isinstance(batch.structure, StructureMaps)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("seq_length", [15, 16, 17, 31, 32, 33, 63, 64, 65])
    @pytest.mark.parametrize("min_length", [0, 16, 32, 64, 128])
    @pytest.mark.parametrize("bitpack_in", [False, True])
    @pytest.mark.parametrize("bitpack_out", [False, True])
    def test_bitpacked(self, batch_size: int, seq_length: int, min_length: int, bitpack_in: bool, bitpack_out: bool) -> None:
        print(f"TestCollateFn::test_bitpacked: {batch_size=} {seq_length=} {min_length=} {bitpack_in=} {bitpack_out=}")

        collate_fn = CollateFn(pin_memory=False, bitpack=bitpack_out, min_length=min_length)

        samples = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            characteristics = torch.randint(0, 2, (seq_length + i, 12), dtype=torch.bool)
            if bitpack_in:
                characteristics = packbits(characteristics, axis=0)
            guides = SemanticGuide(None, None, characteristics)
            structure = StructureMap(torch.full((seq_length + i, 1), True), {0: HierarchicalStructureNone.ANY})
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        # Everything should be padded to the nearest multiple of 8.
        out_seq_length = max(min_length, seq_length + batch_size - 1)
        out_seq_length = math.ceil(out_seq_length / 8) * 8

        batch = collate_fn(samples)
        assert isinstance(batch, FSamples)

        assert isinstance(batch.file, list)
        assert len(batch.file) == batch_size
        for f_1, f_2 in zip(batch.file, (s.file for s in samples), strict=True):
            assert f_1 == f_2

        assert isinstance(batch.name, list)
        assert len(batch.name) == batch_size
        for n_1, n_2 in zip(batch.name, (s.name for s in samples), strict=True):
            assert n_1 == n_2

        assert isinstance(batch.label, LongTensor)
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, ShortTensor)
        assert batch.inputs.shape == (batch_size, out_seq_length)
        # The inputs should be incremented by 1 to reserve 0 for padding.
        for i_1, i_2 in zip(batch.inputs, (s.inputs for s in samples), strict=True):
            assert torch.equal(i_1[:i_2.shape[0]].to(torch.float64), i_2.to(torch.float64) + 1)

        assert isinstance(batch.guides, SemanticGuides)
        assert batch.guides.parse is None
        assert batch.guides.entropy is None
        assert isinstance(batch.guides.characteristics, Tensor)
        if bitpack_in or bitpack_out:
            assert batch.guides.characteristics.dtype == torch.uint8
            assert batch.guides.characteristics.shape == (batch_size, math.ceil(out_seq_length / 8), 12)
        else:
            assert batch.guides.characteristics.dtype == torch.bool
            assert batch.guides.characteristics.shape == (batch_size, out_seq_length, 12)

        assert isinstance(batch.structure, StructureMaps)


class TestCollateFnHierarchical:

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("seq_length", [15, 16, 17, 31, 32, 33, 63, 64, 65])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 4, 5])
    def test_basic(self, batch_size: int, seq_length: int, num_structures: int) -> None:
        print(f"TestCollateFnHierarchical::test_basic: {batch_size=} {seq_length=} {num_structures=}")

        collate_fn = CollateFnHierarchical(pin_memory=False, bitpack=False, num_structures=num_structures)

        samples = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            guides = SemanticGuide(None, None, None)
            structure = StructureMap(
                torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool),
                {k: HierarchicalStructureNone.ANY for k in range(num_structures)}
            )
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        batch = collate_fn(samples)
        assert isinstance(batch, HSamples)

        assert isinstance(batch.file, list)
        assert len(batch.file) == batch_size
        for f_1, f_2 in zip(batch.file, (s.file for s in samples), strict=True):
            assert f_1 == f_2

        assert isinstance(batch.name, list)
        assert len(batch.name) == batch_size
        for n_1, n_2 in zip(batch.name, (s.name for s in samples), strict=True):
            assert n_1 == n_2

        assert isinstance(batch.label, LongTensor)
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, list)
        assert all(isinstance(x, ShortTensor) for x in batch.inputs)
        assert all(x.shape[0] == batch_size for x in batch.inputs)

        assert isinstance(batch.guides, list)
        assert all(isinstance(x, SemanticGuides) for x in batch.guides)
        assert all(x.parse is None and x.entropy is None and x.characteristics is None for x in batch.guides)

        assert isinstance(batch.structure, StructureMaps)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("seq_length", [15, 16, 17, 31, 32, 33, 63, 64, 65])
    @pytest.mark.parametrize("num_structures", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("min_length", [0, 16, 32, 64, 128])
    @pytest.mark.parametrize("bitpack_in", [False, True])
    @pytest.mark.parametrize("bitpack_out", [False, True])
    def test_bitpacked(self, batch_size: int, seq_length: int, num_structures: int, min_length: int, bitpack_in: bool, bitpack_out: bool) -> None:
        print(f"TestCollateFnHierarchical::test_bitpacked: {batch_size=} {seq_length=} {num_structures=} {min_length=} {bitpack_in=} {bitpack_out=}")

        collate_fn = CollateFnHierarchical(
            pin_memory=False,
            bitpack=bitpack_out,
            num_structures=num_structures,
            min_lengths=[min_length + i * 8 for i in range(num_structures)],
        )

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            characteristics = torch.randint(0, 2, (seq_length + i, 12), dtype=torch.bool)
            if bitpack_in:
                characteristics = packbits(characteristics, axis=0)
            guides = SemanticGuide(None, None, characteristics)
            structure = StructureMap(
                torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool),
                {k: HierarchicalStructureNone.ANY for k in range(num_structures)}
            )
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        batch = collate_fn(samples)
        assert isinstance(batch, HSamples)

        assert isinstance(batch.file, list)
        assert len(batch.file) == batch_size
        for f_1, f_2 in zip(batch.file, (s.file for s in samples), strict=True):
            assert f_1 == f_2

        assert isinstance(batch.name, list)
        assert len(batch.name) == batch_size
        for n_1, n_2 in zip(batch.name, (s.name for s in samples), strict=True):
            assert n_1 == n_2

        assert isinstance(batch.label, LongTensor)
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, list)
        assert all(isinstance(x, ShortTensor) for x in batch.inputs)
        assert all(x.shape[0] == batch_size for x in batch.inputs)

        assert isinstance(batch.guides, list)
        assert all(isinstance(x, SemanticGuides) for x in batch.guides)
        assert all(x.parse is None and x.entropy is None for x in batch.guides)
        assert all(isinstance(x.characteristics, Tensor) for x in batch.guides)
        if bitpack_in or bitpack_out:
            assert all(x.characteristics.dtype == torch.uint8 for x in batch.guides)
        else:
            assert all(x.characteristics.dtype == torch.bool for x in batch.guides)

        assert isinstance(batch.structure, StructureMaps)


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
                return FSample(file, file.stem, torch.tensor(label, dtype=torch.int32), inputs, guides, structure)

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
