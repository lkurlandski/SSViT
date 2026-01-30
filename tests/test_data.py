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
from typing import Optional

import pytest
import torch
from torch import Tensor

from src.binanal import CHARACTERISTICS
from src.binanal import HierarchicalStructureNone
from src.binanal import get_ranges_numpy
from src.data import PAD_TO_MULTIPLE_OF
from src.data import Name
from src.data import _InputOrInputs
from src.data import Input
from src.data import Inputs
from src.data import _FSampleOrSamples
from src.data import FSample
from src.data import FSamples
from src.data import _HSampleOrSamples
from src.data import HSample
from src.data import HSamples
from src.data import SSamples
from src.data import _SemanticGuideOrSemanticGuides
from src.data import SemanticGuide
from src.data import SemanticGuides
from src.data import _StructureMapOrStructureMaps
from src.data import StructureMap
from src.data import StructureMaps
from src.data import SemanticGuider
from src.data import StructurePartitioner
from src.data import Preprocessor
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.data import CollateFnStructural
from src.data import _muddy_pad_add_one_
from src.data import _compute_length_to_pad_sequence_to
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
    @pytest.mark.parametrize("num_characteristics", [0, 1, 2, 3])
    def test(self, do_parse: bool, do_entropy: bool, num_characteristics: int) -> None:
        which_characteristics = random.sample(list(CHARACTERISTICS), k=num_characteristics)
        file = FILES[0]
        b = file.read_bytes()
        guider = SemanticGuider(do_parse=do_parse, do_entropy=do_entropy, which_characteristics=which_characteristics)
        sample = guider(b, inputs=torch.frombuffer(b, dtype=torch.uint8))
        assert bool(sample.parse is None) != do_parse
        assert bool(sample.entropy is None) != do_entropy
        assert bool(sample.characteristics is None) != (num_characteristics > 0)
        if num_characteristics > 0:
            assert isinstance(sample.characteristics, Tensor)
            assert sample.characteristics.shape[1] == num_characteristics


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


P = PAD_TO_MULTIPLE_OF
_BATCH_SIZES = tuple([1, 2, 3, 4, 5])
_SEQ_LENGTHS = tuple([P // 2 - 1, P // 2, P // 2 + 1, P - 1, P, P + 1, 2 * P - 1, 2 * P, 2 * P + 1])
_MIN_LENGTHS = tuple([0, P // 2, P, 2 * P])
_LENGTH_BINS = tuple([None, (P, 2 * P, 3 * P, 4 * P)])
_NUM_STRUCTURES = tuple([1, 2, 3, 4, 5])


class TestCollateFn:

    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    @pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
    @pytest.mark.parametrize("min_length", _MIN_LENGTHS)
    @pytest.mark.parametrize("length_bins", _LENGTH_BINS)
    @pytest.mark.parametrize("pin_memory", [False, True])
    @pytest.mark.parametrize("muddy_padded", [False, True])
    def test_basic(self, batch_size: int, seq_length: int, min_length: int, length_bins: Optional[tuple[int, ...]], pin_memory: bool, muddy_padded: bool) -> None:
        print(f"TestCollateFn::test_basic: {batch_size=} {seq_length=} {min_length=} {pin_memory=} {muddy_padded=} {length_bins}")

        collate_fn = CollateFn(pin_memory=pin_memory, bitpack=False, min_length=min_length, length_bins=length_bins, muddy_padded=muddy_padded)

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            inputs = Input(inputs, torch.tensor(inputs.shape[0]), True)
            guides = SemanticGuide(None, None, None)
            structure = StructureMap(
                indicator_mask_to_ranges(torch.full((seq_length + i, 1), True)),
                {0: HierarchicalStructureNone.ANY},
            )
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        out_seq_length = _compute_length_to_pad_sequence_to([s.inputs.inputids.shape[0] for s in samples], min_length, PAD_TO_MULTIPLE_OF, length_bins)

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

        assert batch.label.dtype == torch.long
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, Inputs)
        if muddy_padded:
            assert batch.inputs.inputids.dtype == torch.uint8
            assert batch.inputs.inputids.min() == 0 or (batch_size == 1 and batch.inputs.inputids.min() >= 0)
            assert batch.inputs.inputids.max() <= 255
        else:
            assert batch.inputs.inputids.dtype == torch.int16
            assert batch.inputs.inputids.min() == 0 or (batch_size == 1 and batch.inputs.inputids.min() >= 0)
            assert batch.inputs.inputids.max() <= 256
        assert batch.inputs.shape == (batch_size, out_seq_length)
        assert batch.inputs.inputids.is_pinned() == pin_memory
        for i_1, i_2 in zip(batch.inputs.inputids, (s.inputs.inputids for s in samples), strict=True):
            if muddy_padded:
                assert torch.equal(i_1[:i_2.shape[0]].to(torch.float64), i_2.to(torch.float64))
            else:
                assert torch.equal(i_1[:i_2.shape[0]].to(torch.float64), i_2.to(torch.float64) + 1)

        assert isinstance(batch.guides, SemanticGuides)
        assert isinstance(batch.structure, StructureMaps)


    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    @pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
    @pytest.mark.parametrize("min_length", _MIN_LENGTHS)
    @pytest.mark.parametrize("length_bins", _LENGTH_BINS)
    @pytest.mark.parametrize("bitpack_in", [False, True])
    @pytest.mark.parametrize("bitpack_out", [False, True])
    @pytest.mark.parametrize("muddy_padded", [False, True])
    def test_bitpacked(self, batch_size: int, seq_length: int, min_length: int, length_bins: Optional[tuple[int, ...]], bitpack_in: bool, bitpack_out: bool, muddy_padded: bool) -> None:
        print(f"TestCollateFn::test_bitpacked: {batch_size=} {seq_length=} {min_length=} {length_bins=} {bitpack_in=} {bitpack_out=} {muddy_padded=}")

        collate_fn = CollateFn(pin_memory=False, bitpack=bitpack_out, min_length=min_length, length_bins=length_bins, muddy_padded=muddy_padded)

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            inputs = Input(inputs, torch.tensor(inputs.shape[0]), True)
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

        out_seq_length = _compute_length_to_pad_sequence_to([s.inputs.inputids.shape[0] for s in samples], min_length, PAD_TO_MULTIPLE_OF, length_bins)

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

        assert batch.label.dtype == torch.long
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, Inputs)
        if muddy_padded:
            assert batch.inputs.inputids.dtype == torch.uint8
            assert batch.inputs.inputids.min() == 0 or (batch_size == 1 and batch.inputs.inputids.min() >= 0)
            assert batch.inputs.inputids.max() <= 255
        else:
            assert batch.inputs.inputids.dtype == torch.int16
            assert batch.inputs.inputids.min() == 0 or (batch_size == 1 and batch.inputs.inputids.min() >= 0)
            assert batch.inputs.inputids.max() <= 256
        assert batch.inputs.shape == (batch_size, out_seq_length)
        for i_1, i_2 in zip(batch.inputs.inputids, (s.inputs.inputids for s in samples), strict=True):
            if muddy_padded:
                assert torch.equal(i_1[:i_2.shape[0]].to(torch.float64), i_2.to(torch.float64))
            else:
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

    @staticmethod
    def _compute_out_seq_lengths(num_structures: int, samples: list[FSample], min_lengths: list[int], length_bins: Optional[tuple[int, ...]]) -> list[int]:
        out_seq_lengths = []
        for struct_idx in range(num_structures):
            lengths = []
            for s in samples:
                length = 0
                for lo, hi in s.structure.index[struct_idx]:
                    length += (hi - lo)
                lengths.append(length)
            out_seq_length = _compute_length_to_pad_sequence_to(lengths, min_lengths[struct_idx], PAD_TO_MULTIPLE_OF, length_bins)
            out_seq_lengths.append(out_seq_length)
        return out_seq_lengths

    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    @pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
    @pytest.mark.parametrize("num_structures", _NUM_STRUCTURES)
    @pytest.mark.parametrize("min_length", _MIN_LENGTHS)
    @pytest.mark.parametrize("length_bins", _LENGTH_BINS)
    @pytest.mark.parametrize("muddy_padded", [False, True])
    def test_basic(self, batch_size: int, seq_length: int, num_structures: int, min_length: int, length_bins: Optional[tuple[int, ...]], muddy_padded: bool) -> None:
        print(f"TestCollateFnHierarchical::test_basic: {batch_size=} {seq_length=} {num_structures=} {min_length} {length_bins=} {muddy_padded=}")

        min_lengths = [min_length + i * 8 for i in range(num_structures)]

        collate_fn = CollateFnHierarchical(
            pin_memory=False,
            bitpack=False,
            num_structures=num_structures,
            min_lengths=min_lengths,
            length_bins=length_bins,
            muddy_padded=muddy_padded,
        )

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            inputs = Input(inputs, torch.tensor(inputs.shape[0]), True)
            guides = SemanticGuide(None, None, None)
            structure = StructureMap(
                indicator_mask_to_ranges(torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool)),
                {k: HierarchicalStructureNone.ANY for k in range(num_structures)}
            )
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        out_seq_lengths = self._compute_out_seq_lengths(num_structures, samples, collate_fn.min_lengths, length_bins)

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

        assert batch.label.dtype == torch.long
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, list)
        assert all(isinstance(x, Inputs) for x in batch.inputs)
        if muddy_padded:
            assert all(x.inputids.dtype == torch.uint8 for x in batch.inputs)
        else:
            assert all(x.inputids.dtype == torch.int16 for x in batch.inputs)
        assert all(x.shape[0] == batch_size for x in batch.inputs)
        assert all(x.shape[1] == out_seq_lengths[i] for i, x in enumerate(batch.inputs))

        assert isinstance(batch.guides, list)
        assert all(isinstance(x, SemanticGuides) for x in batch.guides)
        assert all(x.parse is None and x.entropy is None and x.characteristics is None for x in batch.guides)

        assert isinstance(batch.structure, StructureMaps)

    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    @pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
    @pytest.mark.parametrize("num_structures", _NUM_STRUCTURES)
    @pytest.mark.parametrize("min_length", _MIN_LENGTHS)
    @pytest.mark.parametrize("length_bins", _LENGTH_BINS)
    @pytest.mark.parametrize("bitpack_in", [False, True])
    @pytest.mark.parametrize("bitpack_out", [False, True])
    @pytest.mark.parametrize("muddy_padded", [False, True])
    def test_bitpacked(self, batch_size: int, seq_length: int, num_structures: int, min_length: int, length_bins: Optional[tuple[int, ...]], bitpack_in: bool, bitpack_out: bool, muddy_padded: bool) -> None:
        print(f"TestCollateFnHierarchical::test_bitpacked: {batch_size=} {seq_length=} {num_structures=} {min_length=} {length_bins=} {bitpack_in=} {bitpack_out=} {muddy_padded=}")

        min_lengths = [min_length + i * 8 for i in range(num_structures)]

        collate_fn = CollateFnHierarchical(
            pin_memory=False,
            bitpack=bitpack_out,
            num_structures=num_structures,
            min_lengths=min_lengths,
            length_bins=length_bins,
            muddy_padded=muddy_padded,
        )

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            inputs = Input(inputs, torch.tensor(inputs.shape[0]), True)
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

        out_seq_lengths = self._compute_out_seq_lengths(num_structures, samples, min_lengths, length_bins)

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

        assert batch.label.dtype == torch.long
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, list)
        assert all(isinstance(x, Inputs) for x in batch.inputs)
        if muddy_padded:
            assert all(x.inputids.dtype == torch.uint8 for x in batch.inputs)
        else:
            assert all(x.inputids.dtype == torch.int16 for x in batch.inputs)
        assert all(x.shape[0] == batch_size for x in batch.inputs)
        assert all(x.shape[1] == out_seq_lengths[i] for i, x in enumerate(batch.inputs))

        assert isinstance(batch.guides, list)
        assert all(isinstance(x, SemanticGuides) for x in batch.guides)
        assert all(x.parse is None and x.entropy is None for x in batch.guides)
        assert all(isinstance(x.characteristics, Tensor) for x in batch.guides)
        if bitpack_in or bitpack_out:
            assert all(x.characteristics.dtype == torch.uint8 for x in batch.guides)
        else:
            assert all(x.characteristics.dtype == torch.bool for x in batch.guides)

        assert isinstance(batch.structure, StructureMaps)


class TestCollateFnStructural:

    @staticmethod
    def _compute_out_seq_lengths(num_structures: int, samples: list[FSample], min_lengths: list[int], length_bins: Optional[tuple[int, ...]]) -> list[int]:
        out_seq_lengths = []
        for struct_idx in range(num_structures):
            lengths = []
            for s in samples:
                print(f"Structure {struct_idx} Sample {s.file} index: {s.structure.index[struct_idx]}")
                for lo, hi in s.structure.index[struct_idx]:
                    lengths.append(hi - lo)
            print(f"Structure {struct_idx} lengths: {lengths}")
            out_seq_length = _compute_length_to_pad_sequence_to(lengths, min_lengths[struct_idx], PAD_TO_MULTIPLE_OF, length_bins)
            out_seq_lengths.append(out_seq_length)
        return out_seq_lengths

    @pytest.mark.parametrize("batch_size", _BATCH_SIZES)
    @pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
    @pytest.mark.parametrize("num_structures", _NUM_STRUCTURES)
    @pytest.mark.parametrize("min_length", _MIN_LENGTHS)
    @pytest.mark.parametrize("length_bins", _LENGTH_BINS)
    @pytest.mark.parametrize("muddy_padded", [False, True])
    def test_basic(self, batch_size: int, seq_length: int, num_structures: int, min_length: int, length_bins: Optional[tuple[int, ...]], muddy_padded: bool) -> None:
        print(f"TestCollateFnHierarchical::test_basic: {batch_size=} {seq_length=} {num_structures=} {min_length=} {length_bins=} {muddy_padded=}")

        min_lengths = [min_length + i * 8 for i in range(num_structures)]

        collate_fn = CollateFnStructural(
            pin_memory=False,
            bitpack=False,
            num_structures=num_structures,
            min_lengths=min_lengths,
            length_bins=length_bins,
            muddy_padded=muddy_padded,
        )

        samples: list[FSample] = []
        for i in range(batch_size):
            file = f"{get_random_str(16)}/{get_random_bytes(32).hex()}"
            name = Name(file.split("/")[-1])
            label = torch.randint(0, 2, (1,), dtype=torch.long)[0]
            inputs = torch.randint(0, 256, (seq_length + i,), dtype=torch.uint8)
            inputs = Input(inputs, torch.tensor(inputs.shape[0]), True)
            guides = SemanticGuide(None, None, None)
            structure = StructureMap(
                indicator_mask_to_ranges(torch.randint(0, 2, (seq_length + i, num_structures), dtype=torch.bool)),
                {k: HierarchicalStructureNone.ANY for k in range(num_structures)}
            )
            sample = FSample(file, name, label, inputs, guides, structure)
            samples.append(sample)

        out_seq_lengths = self._compute_out_seq_lengths(num_structures, samples, min_lengths, length_bins)

        batch = collate_fn(samples)
        assert isinstance(batch, SSamples)

        assert isinstance(batch.file, list)
        assert len(batch.file) == batch_size
        for f_1, f_2 in zip(batch.file, (s.file for s in samples), strict=True):
            assert f_1 == f_2

        assert isinstance(batch.name, list)
        assert len(batch.name) == batch_size
        for n_1, n_2 in zip(batch.name, (s.name for s in samples), strict=True):
            assert n_1 == n_2

        assert batch.label.dtype == torch.long
        assert batch.label.shape == (batch_size,)
        for l_1, l_2 in zip(batch.label, (s.label for s in samples), strict=True):
            assert torch.equal(l_1.to(torch.float64), l_2.to(torch.float64))

        assert isinstance(batch.inputs, list)
        assert all(isinstance(x, Inputs) for x in batch.inputs)
        if muddy_padded:
            assert all(x.inputids.dtype == torch.uint8 for x in batch.inputs)
        else:
            assert all(x.inputids.dtype == torch.int16 for x in batch.inputs)
        assert all(x.shape[0] == sum(len(s.structure.index[i]) for s in samples) for i, x in enumerate(batch.inputs))
        assert all(x.shape[1] == out_seq_lengths[i] for i, x in enumerate(batch.inputs))

        assert isinstance(batch.guides, list)
        assert all(isinstance(x, SemanticGuides) for x in batch.guides)
        assert all(x.parse is None and x.entropy is None and x.characteristics is None for x in batch.guides)

        assert isinstance(batch.structure, StructureMaps)


class TestPreprocessor:

    def test_trim_more_than_max_structures_from_index_noop(self) -> None:

        index = [
            [(101, 110), (111, 150),],       # STRUCT-A
            [(11, 40),],                     # STRUCT-B
            [],                              # STRUCT-C
            [(0, 10), (50, 100), (41,50),],  # STRUCT-D
            [],                              # STRUCT-E
        ]
        print(f"{index=}")

        unkidx = 2

        max_structures = None
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == index

        max_structures = 7
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == index

        max_structures = 6
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == index

        max_structures = 5
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        print(f"{newindex=}")
        assert newindex == [
            [],
            [(11, 40),],
            [(101, 150)],
            [(0, 10), (50, 100), (41,50),],
            [],
        ]

        max_structures = 4
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == [
            [],
            [(11, 40),],
            [(50, 150)],
            [(0, 10), (41,50),],
            [],
        ]

        max_structures = 3
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == [
            [],
            [(11, 40),],
            [(41, 150)],
            [(0, 10)],
            [],
        ]

        max_structures = 2
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == [
            [],
            [],
            [(11, 150)],
            [(0, 10)],
            [],
        ]

        max_structures = 1
        newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)
        assert newindex == [
            [],
            [],
            [(0, 150)],
            [],
            [],
        ]

        max_structures = 0
        with pytest.raises(ValueError):
            newindex = Preprocessor.trim_more_than_max_structures_from_index(index, max_structures, unkidx)

    def test_truncate_index_to_max_length_per_structure(self) -> None:
        index = [
            [(101, 110), (111, 150),],       # STRUCT-A
            [(11, 40),],                     # STRUCT-B
            [],                              # STRUCT-C
            [(0, 10), (50, 100), (41,50),],  # STRUCT-D
            [],                              # STRUCT-E
        ]
        print(f"{index=}")

        max_length_per_structure = None
        newindex = Preprocessor.truncate_index_to_max_length_per_structure(index, max_length_per_structure)
        assert newindex == index

        max_length_per_structure = -1
        with pytest.raises(ValueError):
            Preprocessor.truncate_index_to_max_length_per_structure(index, max_length_per_structure)

        max_length_per_structure = 0
        with pytest.raises(ValueError):
            Preprocessor.truncate_index_to_max_length_per_structure(index, max_length_per_structure)

        max_length_per_structure = 20
        newindex = Preprocessor.truncate_index_to_max_length_per_structure(index, max_length_per_structure)
        assert newindex == [
            [(101, 110), (111, 131)],       # STRUCT-A
            [(11, 31),],                    # STRUCT-B
            [],                             # STRUCT-C
            [(0, 10), (50, 70), (41,50),],  # STRUCT-D
            [],                             # STRUCT-E
        ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [-1, 0, 1, 2, 3])
def test__muddy_pad_add_one_(batch_size: int, device: str) -> None:
    if batch_size > 0:
        pytest.skip("Not Implemented Yet.")
        return
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
        return

    device = torch.device(device)

    x = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 0, 7, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    l = torch.tensor([6, 3], dtype=torch.long, device=device)

    _muddy_pad_add_one_(x, l, batch_size)
    y = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 1, 8, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    assert torch.equal(x, y)

    x = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0],
            [6, 0, 7, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    l = torch.tensor([6, 3], dtype=torch.long, device=device)
    _muddy_pad_add_one_(x, l, batch_size)
    y = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0],
            [7, 1, 8, 0, 0, 0, 0],
        ],
        dtype=torch.uint8,
        device=device,
    )
    assert torch.equal(x, y)


def test__compute_length_to_pad_sequence_to() -> None:

    lengths = [10, 20, 30, 40, 50]

    length = _compute_length_to_pad_sequence_to(lengths, 0, 1, None)
    assert length == 50

    length = _compute_length_to_pad_sequence_to(lengths, 0, 8, None)
    assert length == 56

    length = _compute_length_to_pad_sequence_to(lengths, 51, 1, None)
    assert length == 51

    length = _compute_length_to_pad_sequence_to(lengths, 51, 2, None)
    assert length == 52

    length = _compute_length_to_pad_sequence_to(lengths, 0, 1, [16, 32, 64, 128])
    assert length == 64

    length = _compute_length_to_pad_sequence_to(lengths, 0, 8, [16, 32, 64, 128])
    assert length == 64

    length = _compute_length_to_pad_sequence_to(lengths, 65, 8, [16, 32, 64, 128])
    assert length == 128
