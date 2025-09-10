"""
Tests.
"""

from collections.abc import Iterable
import itertools
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
from src.binanal import get_ranges_numpy
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
from src.data import ShardAwareBatchSampler
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


def indicator_mask_to_ranges(mask: Tensor) -> list[list[tuple[int, int]]]:
    """
    Convert a boolean indicator mask to ranges.

    Useful for transitioning between indicator-based and range-based StructureMaps.
    """
    ranges: list[list[tuple[int, int]]] = [[] for _ in range(mask.shape[1])]
    for j in range(mask.shape[1]):
        lo, hi = get_ranges_numpy(mask[:, j].numpy())
        index = [(l, h) for l, h in zip(lo.tolist(), hi.tolist())]
        ranges[j] = index
    return ranges


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
            structure = StructureMap(
                indicator_mask_to_ranges(torch.full((seq_length + i, 1), True)),
                {0: HierarchicalStructureNone.ANY},
            )
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
            structure = StructureMap(
                indicator_mask_to_ranges(torch.full((seq_length + i, 1), True)),
                {0: HierarchicalStructureNone.ANY},
            )
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
                indicator_mask_to_ranges(torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool)),
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
                indicator_mask_to_ranges(torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool)),
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

        preprocessor = Preprocessor(False, False, False)
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


class TestShardAwareBatchSampler:

    # ---------- helpers ----------

    @staticmethod
    def _build_mappings(
        shard_sizes: list[list[int]],
        *,
        offset_gap: int = 10,
    ) -> tuple[Tensor, Tensor, Tensor, dict[int, list[int]], dict[int, list[int]]]:
        """
        Create mapping tensors for a synthetic dataset.
        shard_sizes: list per shard, each inner list is the sample 'size' in bytes for that shard.
        Offsets are increasing within each shard; different shards separated in offset space.
        Returns:
            sample_idx_to_shard_idx, sample_idx_to_sample_size, sample_idx_to_sample_offset,
            per_shard_indices (by global sample idx), per_shard_offsets (parallel to indices)
        """
        shard_to_idxs: dict[int, list[int]] = {}
        shard_to_offsets: dict[int, list[int]] = {}

        shard_ids: list[int] = []
        sizes: list[int] = []
        offsets: list[int] = []

        idx = 0
        for sh, sizes_list in enumerate(shard_sizes):
            shard_to_idxs[sh] = []
            shard_to_offsets[sh] = []
            off = (sh + 1) * 1_000_000  # keep shards far apart in offset space
            for s in sizes_list:
                shard_ids.append(sh)
                sizes.append(s)
                offsets.append(off)
                shard_to_idxs[sh].append(idx)
                shard_to_offsets[sh].append(off)
                idx += 1
                off += offset_gap  # strictly increasing within shard

        t_shard = torch.tensor(shard_ids, dtype=torch.int64)
        t_size = torch.tensor(sizes, dtype=torch.int64)
        t_off = torch.tensor(offsets, dtype=torch.int64)
        return t_shard, t_size, t_off, shard_to_idxs, shard_to_offsets

    @staticmethod
    def _flatten(batches: Iterable[list[int]]) -> list[int]:
        return list(itertools.chain.from_iterable(batches))

    @staticmethod
    def _count_full_batches(shard_sizes: list[list[int]], batch_size: int) -> int:
        return sum(len(lst) // batch_size for lst in shard_sizes)

    @staticmethod
    def _leftover_blocks(per_shard_indices: dict[int, list[int]], batch_size: int) -> list[list[int]]:
        blocks: list[list[int]] = []
        for _, idxs in per_shard_indices.items():
            r = len(idxs) % batch_size
            if r:
                blocks.append(idxs[-r:])
        return blocks

    @staticmethod
    def _batch_sums_bytes(batches: list[list[int]], size_tensor: Tensor) -> list[int]:
        return [int(size_tensor[b].sum().item()) for b in batches]

    # ---------- tests ----------

    def test_single_shard_contiguous_full_batches(self) -> None:
        # One shard, N multiple of B → purely contiguous batches in offset order
        B = 3
        shard_sizes = [[10, 20, 30, 40, 50, 60]]  # 6 samples => 2 batches
        t_shard, t_size, t_off, per_shard_idxs, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=False,
            seed=123,
        )

        batches = list(iter(sampler))
        assert len(batches) == 2
        # Validate exact batches are contiguous slices of the single shard
        expected = [per_shard_idxs[0][0:3], per_shard_idxs[0][3:6]]
        assert batches == expected

    def test_multi_shard_full_and_leftover_combination(self) -> None:
        # Two shards, each with leftovers. Leftovers should be kept together per shard
        # and combined at the tail without splitting a block.
        B = 4
        shard_sizes = [
            [10, 10, 10, 10, 10],      # 5 -> 1 full + leftover of 1
            [7, 7, 7, 7, 7, 7, 7],     # 7 -> 1 full + leftover of 3
        ]
        t_shard, t_size, t_off, per_shard_idxs, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=False,
            seed=0,
        )

        batches = list(iter(sampler))
        # 2 full batches (one per shard) + 1 leftover batch combining (1 + 3)
        assert len(batches) == 3

        # First two must be full, single-shard, size B
        assert len(batches[0]) == B and len(batches[1]) == B
        # last is leftover (size B here due to 1+3)
        assert len(batches[2]) == B

        # Check that full batches are exactly contiguous slices from shards
        s0 = per_shard_idxs[0]
        s1 = per_shard_idxs[1]
        full_candidates = [s0[0:4], s1[0:4]]
        assert sorted(map(tuple, batches[:2])) == sorted(map(tuple, full_candidates))  # type: ignore[arg-type]

        # Check leftover blocks are not split across batches
        blocks = self._leftover_blocks(per_shard_idxs, B)  # [[last_of_shard0], [last3_of_shard1]]
        leftover_batches = batches[2:]
        flat_left = self._flatten(leftover_batches)
        for block in blocks:
            # block must appear fully in leftover concatenation
            for i in block:
                assert i in flat_left
            # and must be fully contained in exactly one batch (not split)
            found_in = [i for i, b in enumerate(leftover_batches) if set(block).issubset(set(b))]
            assert len(found_in) == 1, "Leftover block should be inside exactly one batch"

    def test_drop_last_removes_short_leftovers(self) -> None:
        B = 4
        shard_sizes = [
            [1, 1, 1, 1, 1],  # 5 -> 1 full + 1 leftover
            [1, 1, 1, 1, 1],  # 5 -> 1 full + 1 leftover
        ]
        t_shard, t_size, t_off, _, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=True,   # drop last short batch
            seed=0,
        )

        batches = list(iter(sampler))
        # There are exactly 2 full batches total and the leftover (size 2) is dropped.
        assert len(batches) == 2
        assert all(len(b) == B for b in batches)

    def test_first_places_largest_full_batch_first(self) -> None:
        # Construct two full batches with distinct total byte sums.
        B = 4
        shard_sizes = [
            [100, 100, 100, 100],  # sum 400
            [200, 1, 1, 1],        # sum 203
        ]
        t_shard, t_size, t_off, per_shard_idxs, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=True,       # ensure largest-by-bytes goes first
            shuffle=True,     # allow shuffle; first should still be largest
            drop_last=False,
            seed=42,
        )
        # Keep epoch deterministic
        sampler.set_epoch(0)

        batches = list(iter(sampler))
        assert len(batches) == 2
        sums = self._batch_sums_bytes(batches, t_size)
        assert sums[0] == max(sums), "First batch should have the largest total bytes"

        # Also verify each is single-shard full batch
        # (order may be swapped by 'first', but composition is as expected)
        expected_batches = [per_shard_idxs[0], per_shard_idxs[1]]
        assert sorted(map(tuple, batches)) == sorted(map(tuple, expected_batches))  # type: ignore[arg-type]

    def test_shuffle_and_epoch_control(self) -> None:
        # With shuffle=True and set_epoch controlling RNG, epoch 0 should be reproducible;
        # epoch 1 should differ (typically).
        B = 3
        shard_sizes = [
            list(range(1, 10)),  # 9 samples => 3 full batches
            list(range(1, 7)),   # 6 samples => 2 full batches
        ]
        t_shard, t_size, t_off, _, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=False,
            first=False,
            shuffle=True,
            drop_last=False,
            seed=1234,
        )

        sampler.set_epoch(0)
        batches_e0_a = list(iter(sampler))
        sampler.set_epoch(0)
        batches_e0_b = list(iter(sampler))
        sampler.set_epoch(1)
        batches_e1 = list(iter(sampler))

        # Same epoch → identical sequence
        assert batches_e0_a == batches_e0_b

        # Different epoch → likely different sequence (not guaranteed by math, but very likely)
        # Instead of strict inequality, test that composition is identical (a permutation)
        flat_e0 = self._flatten(batches_e0_a)
        flat_e1 = self._flatten(batches_e1)
        assert sorted(flat_e0) == sorted(flat_e1), "Different epochs must yield same sample multiset"
        # And very likely order differs:
        assert batches_e0_a != batches_e1

    def test_local_shuffle_window_preserves_locality(self) -> None:
        # Use windowed local shuffle with window == batch_size so each full batch
        # should come from one offset-window per shard.
        B = 4
        shard_sizes = [
            [1] * 16,  # 4 full batches
            [1] * 8,   # 2 full batches
        ]
        t_shard, t_size, t_off, per_shard_idxs, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=False,
            first=False,
            shuffle=True,
            drop_last=False,
            local_shuffle_window=B,  # align window with batch size
            seed=2025,
        )
        sampler.set_epoch(0)
        batches = list(iter(sampler))

        # Compute mapping from idx -> ordinal position within its shard (offset-sorted)
        pos_in_shard: dict[int, dict[int, int]] = {}
        for sh, idxs in per_shard_idxs.items():
            pos_in_shard[sh] = {i: p for p, i in enumerate(idxs)}

        # First N_full batches must be single shard full batches
        n_full_expected = self._count_full_batches(shard_sizes, B)
        full_batches = batches[:n_full_expected]
        assert all(len(b) == B for b in full_batches)

        # For each full batch, check all indices belong to the same shard
        # and their positions are within a small window (<= B).
        for b in full_batches:
            shards = {int(t_shard[i].item()) for i in b}
            assert len(shards) == 1
            sh = next(iter(shards))
            positions = sorted(pos_in_shard[sh][i] for i in b)
            assert positions[-1] - positions[0] < B, "Batch should fit within one shuffled offset window"

    def test_len_matches_formula(self) -> None:
        shard_sizes = [
            [1] * 10,  # 10
            [1] * 7,   # 7
            [1] * 5,   # 5
        ]
        B = 4
        t_shard, t_size, t_off, _, _ = self._build_mappings(shard_sizes)

        sampler_keep = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=False,
            seed=0,
        )
        sampler_drop = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=True,
            seed=0,
        )

        # Manual expectation
        full = sum(len(lst) // B for lst in shard_sizes)
        leftover = sum(len(lst) % B for lst in shard_sizes)
        expected_keep = full + math.ceil(leftover / B)
        expected_drop = full + math.floor(leftover / B)

        assert len(sampler_keep) == expected_keep
        assert len(sampler_drop) == expected_drop

    def test_contiguous_true_uses_offset_order(self) -> None:
        # Ensure that with contiguous=True the batches correspond to contiguous
        # slices in offset order for each shard (though batch order may be shuffled=False here)
        B = 3
        shard_sizes = [
            [5, 6, 7, 8, 9, 10],  # two batches
            [1, 2, 3, 4],         # one batch
        ]
        t_shard, t_size, t_off, per_shard_idxs, _ = self._build_mappings(shard_sizes)

        sampler = ShardAwareBatchSampler(
            batch_size=B,
            sample_idx_to_shard_idx=t_shard,
            sample_idx_to_sample_size=t_size,
            sample_idx_to_sample_offset=t_off,
            contiguous=True,
            first=False,
            shuffle=False,
            drop_last=False,
            seed=7,
        )

        batches = list(iter(sampler))
        # Expected full batches are contiguous slices per shard
        expected_full = [
            per_shard_idxs[0][0:3], per_shard_idxs[0][3:6],
            per_shard_idxs[1][0:3],
        ]
        # The last leftover batch (size 1) should contain the leftover from shard 1
        expected_leftover = [per_shard_idxs[1][3:4]]
        assert batches[:3] == expected_full
        assert batches[3:] == expected_leftover

    def test_invalid_inputs_raise(self) -> None:
        B = 4
        # Mismatched lengths
        with pytest.raises(ValueError):
            ShardAwareBatchSampler(
                batch_size=B,
                sample_idx_to_shard_idx=torch.tensor([0, 0, 1]),
                sample_idx_to_sample_size=torch.tensor([1, 2]),  # shorter
                sample_idx_to_sample_offset=torch.tensor([0, 10, 20]),
                contiguous=True,
                first=False,
                shuffle=False,
                drop_last=False,
            )

        # Non-1D tensors
        with pytest.raises(ValueError):
            ShardAwareBatchSampler(
                batch_size=B,
                sample_idx_to_shard_idx=torch.tensor([[0, 0], [1, 1]]),
                sample_idx_to_sample_size=torch.tensor([1, 2, 3, 4]),
                sample_idx_to_sample_offset=torch.tensor([0, 10, 20, 30]),
                contiguous=True,
                first=False,
                shuffle=False,
                drop_last=False,
            )

