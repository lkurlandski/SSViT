"""
Manage data and datasets.
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import MutableSequence
from collections import deque
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
from typing import Literal
from typing import Generic
from typing import Optional
from typing import Self
from typing import TypeVar
import warnings

import lief
import numpy as np
import torch
from torch import Tensor
from torch import BoolTensor
from torch import HalfTensor
from torch import FloatTensor
from torch import DoubleTensor
from torch import CharTensor
from torch import ByteTensor
from torch import ShortTensor
from torch import IntTensor
from torch import LongTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils._pytree import tree_map

from src.binanal import _parse_pe_and_get_size
from src.binanal import _get_size_of_liefparse
from src.binanal import LiefParse
from src.binanal import ParserGuider
from src.binanal import CharacteristicGuider
from src.binanal import StructureParser
from src.binanal import HierarchicalLevel
from src.binanal import HierarchicalStructure
from src.binanal import HierarchicalStructureCoarse
from src.binanal import HierarchicalStructureFine
from src.binanal import HierarchicalStructureMiddle
from src.binanal import HierarchicalStructureNone
from src.binentropy import compute_entropy_rolling_numpy
from src.bitpacking import packbits
from src.bitpacking import unpackbits
from src.bitpacking import slice_bitpacked_tensor
from src.simpledb import SimpleDB
from src.utils import check_tensor
from src.utils import pad_sequence


StrPath = str | os.PathLike[str]


PAD_TO_MULTIPLE_OF = 8
if PAD_TO_MULTIPLE_OF % 8 != 0:
    raise ValueError("PAD_TO_MULTIPLE_OF must be a multiple of 8.")


class Name(str):

    def __new__(cls, value: StrPath) -> Name:
        value = str(value).split("/")[-1]
        value = value.split(".")[0]
        if len(value) != 64 or not all(c in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Invalid name: {value}")
        return super().__new__(cls, value)


class _SemanticGuideOrSemanticGuides(ABC):

    # NOTE: compression/decompression with bitpacking will implicitly zero pad tensors to a multiple of 8.
    # NOTE: the boolean tensors are converted to floating point internally, to prepare for learning.
    # FIXME: some of these methods, e.g., `clone`, should return a new instance!

    parse: Optional[BoolTensor | ByteTensor | FloatTensor | HalfTensor | DoubleTensor]
    entropy: Optional[HalfTensor | FloatTensor | DoubleTensor]
    characteristics: Optional[BoolTensor | ByteTensor | HalfTensor | FloatTensor | DoubleTensor]

    def __init__(self, parse: Optional[Tensor], entropy: Optional[Tensor], characteristics: Optional[Tensor]) -> None:
        if parse is not None and parse.dtype not in (torch.bool, torch.uint8):
            raise TypeError(f"Expected parse to be bool or uint8, got {parse.dtype}")
        if entropy is not None and entropy.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"Expected entropy to be float16, float32, or float64, got {entropy.dtype}")
        if characteristics is not None and characteristics.dtype not in (torch.bool, torch.uint8):
            raise TypeError(f"Expected characteristics to be bool or uint8, got {characteristics.dtype}")

        self.parse = parse
        self.entropy = entropy
        self.characteristics = characteristics

    @property
    @abstractmethod
    def length_axis(self) -> int:
        ...

    @property
    def is_bitpacked(self) -> bool:
        is_bitpacked = [t.dtype == torch.uint8 for t in (self.parse, self.characteristics) if t is not None]
        if all(is_bitpacked):
            return True
        if not any(is_bitpacked):
            return False
        raise RuntimeError(f"Unexpected tensor dtype state: {is_bitpacked=}")

    def is_contiguous(self) -> bool:
        is_contiguous = [t.is_contiguous() for t in (self.parse, self.entropy, self.characteristics) if t is not None]
        if all(is_contiguous):
            return True
        if not any(is_contiguous):
            return False
        raise RuntimeError(f"Unexpected tensor contiguity state: {is_contiguous=}")

    @property
    def is_cuda(self) -> bool:
        is_cuda = [t.is_cuda for t in (self.parse, self.entropy, self.characteristics) if t is not None]
        if all(is_cuda):
            return True
        if not any(is_cuda):
            return False
        raise RuntimeError(f"Unexpected tensor device state: {is_cuda=}")

    def contiguous(self) -> Self:
        if self.parse is not None:
            self.parse = self.parse.contiguous()
        if self.entropy is not None:
            self.entropy = self.entropy.contiguous()
        if self.characteristics is not None:
            self.characteristics = self.characteristics.contiguous()
        return self

    def record_stream(self, s: torch.cuda.Stream) -> None:
        if self.parse is not None:
            self.parse.record_stream(s)
        if self.entropy is not None:
            self.entropy.record_stream(s)
        if self.characteristics is not None:
            self.characteristics.record_stream(s)

    def clone(self) -> Self:
        if self.parse is not None:
            self.parse = self.parse.clone()
        if self.entropy is not None:
            self.entropy = self.entropy.clone()
        if self.characteristics is not None:
            self.characteristics = self.characteristics.clone()
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        if self.parse is not None:
            self.parse = self.parse.to(device, non_blocking=non_blocking)
        if self.entropy is not None:
            self.entropy = self.entropy.to(device, non_blocking=non_blocking)
        if self.characteristics is not None:
            self.characteristics = self.characteristics.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> Self:
        if self.parse is not None:
            self.parse = self.parse.pin_memory()
        if self.entropy is not None:
            self.entropy = self.entropy.pin_memory()
        if self.characteristics is not None:
            self.characteristics = self.characteristics.pin_memory()
        return self

    def compress(self) -> Self:
        if self.parse is not None and self.parse.dtype == torch.bool:
            self.parse = packbits(self.parse, axis=self.length_axis)
        if self.entropy is not None and self.entropy.dtype != torch.float16:
            self.entropy = self.entropy.to(torch.float16)
        if self.characteristics is not None and self.characteristics.dtype == torch.bool:
            self.characteristics = packbits(self.characteristics, axis=self.length_axis)
        return self

    def decompress(self) -> Self:
        if self.parse is not None:
            if self.parse.dtype == torch.uint8:
                self.parse = unpackbits(self.parse, axis=self.length_axis)
            self.parse = self.parse.to(torch.float32)
        if self.entropy is not None and self.entropy.dtype != torch.float32:
            self.entropy = self.entropy.to(torch.float32)
        if self.characteristics is not None:
            if self.characteristics.dtype == torch.uint8:
                self.characteristics = unpackbits(self.characteristics, axis=self.length_axis)
            self.characteristics = self.characteristics.to(torch.float32)
        return self

    def trim(self, length: Optional[int]) -> Self:
        if length is None:
            return self

        if self.parse is not None:
            length_ = length
            if self.parse.dtype == torch.uint8:
                length_ = math.ceil(length / 8)
            slice_ = [slice(length_) if i == self.length_axis else slice(None) for i in range(self.parse.ndim)]
            self.parse = self.parse[tuple(slice_)]

        if self.entropy is not None:
            slice_ = [slice(length) if i == self.length_axis else slice(None) for i in range(self.parse.ndim)]
            self.parse = self.parse[tuple(slice_)]

        if self.characteristics is not None:
            length_ = length
            if self.characteristics.dtype == torch.uint8:
                length_ = math.ceil(length / 8)
            slice_ = [slice(length_) if i == self.length_axis else slice(None) for i in range(self.characteristics.ndim)]
            self.characteristics = self.characteristics[tuple(slice_)]

        return self

    def select(self, *, mask: Optional[BoolTensor] = None, idx: Optional[IntTensor] = None, ranges: Optional[list[tuple[int, int]]] = None) -> Self:
        """
        Select a subset of the data along the length axis.

        NOTE: if data is bitpacked, the selectors are assumed to correspond to the unpacked data.
        """

        def slice_normal_optional_tensor(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            if mask is not None:
                return t[mask]
            if idx is not None:
                return t[idx]
            if ranges is not None:
                if len(ranges) > 0:
                    return torch.cat([t[lo:hi] for lo, hi in ranges], dim=self.length_axis)
                return t.new_empty(t.shape[: self.length_axis] + (0,) + t.shape[self.length_axis + 1 :])
            raise RuntimeError("unreachable")

        def slice_bitpacked_optional_tensor(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            return slice_bitpacked_tensor(t, mask=mask, idx=idx, ranges=ranges, bigchunks=True, axis=self.length_axis)


        selectors = [mask is not None, idx is not None, ranges is not None]
        if sum(selectors) != 1:
            raise ValueError("exactly one of mask, idx, or ranges must be provided")

        entropy = slice_normal_optional_tensor(self.entropy)

        if not self.is_bitpacked:
            parse = slice_normal_optional_tensor(self.parse)
            characteristics = slice_normal_optional_tensor(self.characteristics)
            return self.__class__(parse, entropy, characteristics)

        # The bitpacked data may be padded to a multiple of 8, so we may need to pad the mask.
        # Entropy has already been sliced above, so we need not worry about a length mismatch.
        if mask is not None and len(mask) % 8 != 0:
            mask = torch.nn.functional.pad(mask, (0, 8 - (mask.size(0) % 8)), "constant", False)

        parse = slice_bitpacked_optional_tensor(self.parse)
        characteristics = slice_bitpacked_optional_tensor(self.characteristics)

        return self.__class__(parse, entropy, characteristics)


class SemanticGuide(_SemanticGuideOrSemanticGuides):

    @property
    def length_axis(self) -> int:
        return 0


class SemanticGuides(_SemanticGuideOrSemanticGuides):

    @property
    def length_axis(self) -> int:
        return 1

    @classmethod
    def from_singles(cls, guides: Sequence[SemanticGuide], pin_memory: bool = False, min_length: int = 0) -> SemanticGuides:
        if len(guides) == 0:
            raise ValueError("Cannot create Guides from empty list.")

        parse = None
        if guides[0].parse is not None:
            padding_value = False if guides[0].parse.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].parse.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            parse = pad_sequence([g.parse for g in guides], True, padding_value, "right", pin_memory, pad_to_multiple_of, min_length)
        entropy = None
        if guides[0].entropy is not None:
            entropy = pad_sequence([g.entropy for g in guides], True, 0.0, "right", pin_memory, PAD_TO_MULTIPLE_OF, min_length)
        characteristics = None
        if guides[0].characteristics is not None:
            padding_value = False if guides[0].characteristics.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].characteristics.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            characteristics = pad_sequence([g.characteristics for g in guides], True, padding_value, "right", pin_memory, pad_to_multiple_of, min_length)

        return cls(parse, entropy, characteristics)


class SemanticGuider:
    """
    Semantic guides to acompany a byte stream.
    """

    def __init__(self, do_parse: bool = False, do_entropy: bool = False, do_characteristics: bool = False, radius: int = 256, simple: bool = False) -> None:
        self.do_parse = do_parse
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.radius = radius
        self.simple = simple

    def __call__(self, data: Optional[LiefParse | lief.PE.Binary], size: Optional[int] = None, inputs: Optional[torch.CharTensor] = None) -> SemanticGuide:
        if not any((self.do_parse, self.do_entropy, self.do_characteristics)):
            return SemanticGuide(None, None, None)

        if data is None:
            raise ValueError("Data must be provided for semantic guides.")

        parse = None
        if self.do_parse:
            parse = ParserGuider(data, size)(simple=self.simple)
            parse = torch.from_numpy(parse)

        entropy = None
        if self.do_entropy:
            if inputs is None:
                raise RuntimeError("Inputs must be provided for entropy computation.")
            # NOTE: computing entropy with JIT-based numba is faster than torch.
            # TODO: there may be a better way to prevent additional data copying.
            inputs = inputs.numpy(force=True)
            entropy = compute_entropy_rolling_numpy(inputs, self.radius)
            entropy = torch.from_numpy(entropy).to(torch.float16)

        characteristics = None
        if self.do_characteristics:
            characteristics = CharacteristicGuider(data, size, use_packed=True)()
            characteristics = torch.from_numpy(characteristics)

        return SemanticGuide(parse, entropy, characteristics)


class _StructureMapOrStructureMaps(ABC):

    # FIXME: some of these methods, e.g., `clone`, should return a new instance!

    index: list[list[tuple[int, int]]] | list[list[list[tuple[int, int]]]]
    lexicon: Mapping[int, HierarchicalStructure]

    def __init__(self, index: list[list[tuple[int, int]]] | list[list[list[tuple[int, int]]]], lexicon: Mapping[int, HierarchicalStructure]) -> None:
        self.index = index
        self.lexicon = lexicon
        self.verify_inputs()

    @abstractmethod
    def verify_inputs(self) -> None:
        """Verifies that index and lexicon are well-formed and compatible."""
        ...

    @abstractmethod
    def trim(self, length: Optional[int]) -> Self:
        """Returns a copy excluding regions outside of length."""
        ...

    def clone(self) -> Self:
        """Return a new, deeply copied instance."""
        index = deepcopy(self.index)
        lexicon = deepcopy(self.lexicon)
        return self.__class__(index, lexicon)


class StructureMap(_StructureMapOrStructureMaps):

    index: list[list[tuple[int, int]]]  # [STRUCTURE][REGION][LO, HI]

    @staticmethod
    def verify_index(index: list[list[tuple[int, int]]]) -> bool:
        if not isinstance(index, list):
            return False
        if len(index) > 0:
            if not isinstance(index[0], list):
                return False
            if len(index[0]) > 0:
                if not isinstance(index[0][0], tuple):
                    return False
                if len(index[0][0]) != 2:
                    return False
                if not all(isinstance(x, int) for x in index[0][0]):
                    return False
        return True

    @staticmethod
    def verify_lexicon(lexicon: Mapping[int, HierarchicalStructure]) -> bool:
        if not isinstance(lexicon, Mapping):
            return False
        if len(lexicon) > 0:
            if not all(isinstance(k, int) for k in lexicon.keys()):
                return False
            if not all(isinstance(v, HierarchicalStructure) for v in lexicon.values()):
                return False
        return True

    def verify_inputs(self) -> None:
        if not StructureMap.verify_index(self.index):
            raise TypeError(f"StructureMap index must be list[list[tuple[int, int]]], got {type(self.index)}")
        if not StructureMap.verify_lexicon(self.lexicon):
            raise TypeError(f"StructureMap lexicon must be Mapping[int, HierarchicalStructure], got {type(self.lexicon)}")
        if len(self.index) != len(self.lexicon):
            raise ValueError(f"StructureMap index and lexicon must have the same length. Got {len(self.index)} and {len(self.lexicon)}.")

    @staticmethod
    def trim_index(index: list[list[tuple[int, int]]], length: int) -> list[list[tuple[int, int]]]:
        new: list[list[tuple[int, int]]] = []
        for i in range(len(index)):
            new.append([])
            for j in range(len(index[i])):
                lo, hi = index[i][j]
                if lo >= length:
                    continue
                if hi > length:
                    hi = length
                if lo < hi:
                    new[i].append((lo, hi))
        return new

    def trim(self, length: Optional[int]) -> Self:
        if length is None:
            return self.clone()

        index = StructureMap.trim_index(self.index, length)
        lexicon = deepcopy(self.lexicon)

        return self.__class__(index, lexicon)


class StructureMaps(_StructureMapOrStructureMaps):

    index: list[list[list[tuple[int, int]]]]  # [SAMPLE][STRUCTURE][REGION][LO, HI]

    def verify_inputs(self) -> None:
        if not isinstance(self.index, list) or (len(self.index) > 0 and not StructureMap.verify_index(self.index[0])):
            raise TypeError(f"StructureMaps index must be list[list[list[tuple[int, int]]]], got {type(self.index)}")
        if not StructureMap.verify_lexicon(self.lexicon):
            raise TypeError(f"StructureMaps lexicon must be Mapping[int, HierarchicalStructure], got {type(self.lexicon)}")
        if len(self.index) > 0 and len(self.index[0]) != len(self.lexicon):
            raise ValueError(f"StructureMaps index and lexicon must have the same length. Got {len(self.index[0])} and {len(self.lexicon)}.")

    @classmethod
    def from_singles(cls, maps: Sequence[StructureMap], pin_memory: bool = False, min_length: int = 0) -> _StructureMapOrStructureMaps:
        if len(maps) == 0:
            raise ValueError("Cannot create BatchedStructureMap from empty list.")
        lexicon = maps[0].lexicon
        for m in maps[1:]:
            if m.lexicon != lexicon:
                raise ValueError("All StructureMaps must have the same lexicon to be batched.")
        index = [m.index for m in maps]
        return cls(index, lexicon)

    def trim(self, length: Optional[int]) -> Self:
        if length is None:
            return self.clone()

        index = [StructureMap.trim_index(index_, length) for index_ in self.index]
        lexicon = deepcopy(self.lexicon)

        return self.__class__(index, lexicon)


class StructurePartitioner:
    """
    Partitions a binary into hierarchical structures.
    """

    def __init__(self, level: HierarchicalLevel = HierarchicalLevel.NONE) -> None:
        self.level = level

    def __call__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> StructureMap:
        if self.level == HierarchicalLevel.NONE:
            size = _get_size_of_liefparse(data) if size is None else size
            index = [[(0, size)]]
            lexicon = {0: HierarchicalStructureNone.ANY}
            return StructureMap(index, lexicon)

        if data is None:
            raise ValueError("Data must be provided for structure partitioning.")

        parser = StructureParser(data, size)

        if self.level == HierarchicalLevel.NONE:
            structures = list(HierarchicalStructureNone)
        if self.level == HierarchicalLevel.COARSE:
            structures = list(HierarchicalStructureCoarse)
        elif self.level == HierarchicalLevel.MIDDLE:
            structures = list(HierarchicalStructureMiddle)
        elif self.level == HierarchicalLevel.FINE:
            structures = list(HierarchicalStructureFine)
        else:
            raise TypeError(f"Unknown HierarchicalLevel: {self.level}. Expected one of {list(HierarchicalLevel)}.")

        index = [[] for _ in structures]
        lexicon = {}
        for i, s in enumerate(structures):
            index[i] = parser(s)
            lexicon[i] = s
        return StructureMap(index, lexicon)


class _FSampleOrSamples(ABC):

    # FIXME: some of these methods, e.g., `clone`, should return a new instance!

    file: StrPath | list[StrPath]
    name: Name | list[Name]
    label: ShortTensor | IntTensor | LongTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: _SemanticGuideOrSemanticGuides
    structure: _StructureMapOrStructureMaps

    def __init__(
        self,
        file: StrPath | list[StrPath],
        name: Name | list[Name],
        label: ShortTensor | IntTensor | LongTensor,
        inputs: ByteTensor | ShortTensor | IntTensor | LongTensor,
        guides: _SemanticGuideOrSemanticGuides,
        structure: _StructureMapOrStructureMaps,
    ) -> None:
        self.file = file
        self.name = name
        self.label = label
        self.inputs = inputs
        self.guides = guides
        self.structure = structure
        self.verify_inputs()

    def __iter__(self) -> Iterable[tuple[StrPath | list[StrPath], Name | list[Name], IntTensor, IntTensor, SemanticGuides, StructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))

    @property
    def parse(self) -> Optional[list[Tensor]]:
        return self.guides.parse

    @property
    def entropy(self) -> Optional[list[Tensor]]:
        return self.guides.entropy

    @property
    def characteristics(self) -> Optional[list[Tensor]]:
        return self.guides.characteristics

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def verify_inputs(self) -> None:
        ...

    @property
    def is_cuda(self) -> bool:
        if self.label.is_cuda and self.inputs.is_cuda and self.guides.is_cuda:
            return True
        if not self.label.is_cuda and not self.inputs.is_cuda and not self.guides.is_cuda:
            return False
        raise RuntimeError(f"Unexpected tensor device state: {self.label.is_cuda=}, {self.inputs.is_cuda=}, {self.guides.is_cuda=}")

    def record_stream(self, s: torch.cuda.Stream) -> None:
        self.label.record_stream(s)
        self.inputs.record_stream(s)
        self.guides.record_stream(s)

    def clone(self) -> Self:
        self.file = deepcopy(self.file)
        self.name = deepcopy(self.name)
        self.inputs = self.inputs.clone()
        self.label = self.label.clone()
        self.guides = self.guides.clone()
        self.structure = self.structure.clone()
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        self.label = self.label.to(device, non_blocking=non_blocking)
        self.inputs = self.inputs.to(device, non_blocking=non_blocking)
        self.guides = self.guides.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> Self:
        self.label = self.label.pin_memory()
        self.inputs = self.inputs.pin_memory()
        self.guides = self.guides.pin_memory()
        return self

    def compress(self) -> Self:
        self.label = self.label.to(torch.int16)
        self.inputs = self.inputs.to(torch.int16)
        self.guides = self.guides.compress()
        return self

    def decompress(self) -> Self:
        self.label = self.label.to(torch.int64)
        self.inputs = self.inputs.to(torch.int32)
        self.guides = self.guides.decompress()
        return self


class FSample(_FSampleOrSamples):
    file: StrPath
    name: Name
    label: ShortTensor | IntTensor | LongTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuide
    structure: StructureMap

    def __len__(self) -> int:
        return 1

    def verify_inputs(self) -> None:
        check_tensor(self.label, (), (torch.int16, torch.int32, torch.int64))
        check_tensor(self.inputs, (None,), (torch.uint8, torch.int16, torch.int32, torch.int64))


class FSamples(_FSampleOrSamples):
    file: list[StrPath]
    name: list[Name]
    label: ShortTensor | IntTensor | LongTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuides
    structure: StructureMaps

    def __len__(self) -> int:
        return len(self.file)

    def verify_inputs(self) -> None:
        check_tensor(self.label, (None,), (torch.int16, torch.int32, torch.int64))
        check_tensor(self.inputs, (self.label.shape[0], None), (torch.uint8, torch.int16, torch.int32, torch.int64))


TGuide = TypeVar("TGuide", bound=_SemanticGuideOrSemanticGuides)

class _HSampleOrSamples(Generic[TGuide], ABC):
    """
    A similar container as _FSampleOrSamples, but specifically for hierarchical models.
    The main difference is that the guides and structure are lists, one per hierarchical structure.

    NOTE: its unclear whether or not the lists of tensors can be moved efficiently across processes,
        i.e., when num_workers > 0 in DataLoader.

    FIXME: some of these methods, e.g., `clone`, should return a new instance!
    """
    file: StrPath | list[StrPath]
    name: Name | list[Name]
    label: ShortTensor | IntTensor | LongTensor
    inputs: MutableSequence[ByteTensor | ShortTensor | IntTensor | LongTensor]
    guides: MutableSequence[TGuide]
    structure: _StructureMapOrStructureMaps

    def __init__(
        self,
        file: StrPath | list[StrPath],
        name: Name | list[Name],
        label: ShortTensor | IntTensor | LongTensor,
        inputs: MutableSequence[ByteTensor | ShortTensor | IntTensor | LongTensor],
        guides: MutableSequence[TGuide],
        structure: _StructureMapOrStructureMaps,
    ) -> None:
        self.file = file
        self.name = name
        self.label = label
        self.inputs = inputs
        self.guides = guides
        self.structure = structure
        self.verify_inputs()

    def __iter__(self) -> Iterable[tuple[StrPath | list[StrPath], Name | list[Name], IntTensor, MutableSequence[IntTensor], MutableSequence[SemanticGuides], StructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))

    @property
    def num_structures(self) -> int:
        return len(self.inputs)

    @property
    def parse(self) -> list[Optional[Tensor]]:
        return [g.parse for g in self.guides]

    @property
    def entropy(self) -> list[Optional[Tensor]]:
        return [g.entropy for g in self.guides]

    @property
    def characteristics(self) -> list[Optional[Tensor]]:
        return [g.characteristics for g in self.guides]

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def verify_inputs(self) -> None:
        if len(self.guides) != len(self.inputs):
            raise ValueError(f"HSample inputs and guides must have the same length. Got {len(self.inputs)} and {len(self.guides)}.")
        if self.num_structures == 0:
            raise ValueError("HSample must have at least one structure.")
        if self.num_structures == 1:
            warnings.warn("_HSampleOrSamples has only one structure. Consider using _FSampleOrSamples instead.")

    @property
    def is_cuda(self) -> bool:
        for i in range(self.num_structures):
            if self.inputs[i].is_cuda and self.guides[i].is_cuda:
                continue
            break
        else:
            if self.label.is_cuda:
                return True

        for i in range(self.num_structures):
            if not self.inputs[i].is_cuda and not self.guides[i].is_cuda:
                continue
            break
        else:
            if not self.label.is_cuda:
                return False

        raise RuntimeError(f"Unexpected tensor device state with some components on GPU and some not.")

    def record_stream(self, s: torch.cuda.Stream) -> None:
        self.label.record_stream(s)
        for i in range(self.num_structures):
            self.inputs[i].record_stream(s)
            self.guides[i].record_stream(s)

    def clone(self) -> Self:
        self.file = deepcopy(self.file)
        self.name = deepcopy(self.name)
        self.label = self.label.clone()
        for i in range(self.num_structures):
            self.inputs[i] = self.inputs[i].clone()
            self.guides[i] = self.guides[i].clone()
        self.structure = self.structure.clone()
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        self.label = self.label.to(device, non_blocking=non_blocking)
        for i in range(self.num_structures):
            self.inputs[i] = self.inputs[i].to(device, non_blocking=non_blocking)
            self.guides[i] = self.guides[i].to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> Self:
        self.label = self.label.pin_memory()
        for i in range(self.num_structures):
            self.inputs[i] = self.inputs[i].pin_memory()
            self.guides[i] = self.guides[i].pin_memory()
        return self

    def compress(self) -> Self:
        self.label = self.label.to(torch.int16)
        for i in range(self.num_structures):
            self.inputs[i] = self.inputs[i].to(torch.int16)
            self.guides[i] = self.guides[i].compress()
        return self

    def decompress(self) -> Self:
        self.label = self.label.to(torch.int64)
        for i in range(self.num_structures):
            self.inputs[i] = self.inputs[i].to(torch.int32)
            self.guides[i] = self.guides[i].decompress()
        return self


class HSample(_HSampleOrSamples[SemanticGuide]):
    file: StrPath
    name: Name
    label: ShortTensor | IntTensor | LongTensor
    inputs: MutableSequence[ByteTensor | ShortTensor | IntTensor | LongTensor]
    guides: MutableSequence[SemanticGuide]
    structure: StructureMap

    def __len__(self) -> int:
        return 1

    def verify_inputs(self) -> None:
        super().verify_inputs()
        check_tensor(self.label, (), (torch.int16, torch.int32, torch.int64))
        for inp in self.inputs:
            check_tensor(inp, (None,), (torch.uint8, torch.int16, torch.int32, torch.int64))


class HSamples(_HSampleOrSamples[SemanticGuides]):
    file: list[StrPath]
    name: list[Name]
    label: ShortTensor | IntTensor | LongTensor
    inputs: MutableSequence[ByteTensor | ShortTensor | IntTensor | LongTensor]
    guides: MutableSequence[SemanticGuides]
    structure: StructureMaps

    def __len__(self) -> int:
        return len(self.file)

    def verify_inputs(self) -> None:
        super().verify_inputs()
        check_tensor(self.label, (None,), (torch.int16, torch.int32, torch.int64))
        for inp in self.inputs:
            check_tensor(inp, (self.label.shape[0], None), (torch.uint8, torch.int16, torch.int32, torch.int64))


FOrHSample  = FSample | HSample
FOrHSamples = FSamples | HSamples


class BinaryDataset(Dataset):  # type: ignore[misc]

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], preprocessor: Preprocessor) -> None:
        self.files = files
        self.labels = labels
        self.preprocessor = preprocessor

    def __getitem__(self, i: int) -> FSample:
        return self.preprocessor(self.labels[i], file=self.files[i])

    def __len__(self) -> int:
        return len(self.files)


class SimpleDBDataset(Dataset):  # type: ignore[misc]

    # NOTE: best not to play around with (or even access) the internal state of this class from the outside.

    def __init__(self, idx_or_names: Sequence[int | str] | Tensor, db: SimpleDB, preprocessor: Preprocessor) -> None:
        if db.is_open:  # Ensure the db is closed, so we can pickle it around process boundaries.
            raise RuntimeError("Expected a closed SimpleDB instance, but the provided instance is already open.")

        self.idx_or_names = idx_or_names  # Maps a subset of Dataset indices to SimpleDB indices or names.
        self.preprocessor = preprocessor
        self._db = deepcopy(db)
        self._pid = os.getpid()

    @property
    def db(self) -> SimpleDB:
        """
        Return an opened process-local SimpleDB instance.
        """
        if (pid := os.getpid()) != self._pid:
            self._db.close()
            self._pid = pid

        if not self._db.is_open:
            self._db = self._db.open()

        return self._db

    def __getitem__(self, i: int) -> FSample:
        idx_or_name = self.idx_or_names[i]
        if not isinstance(idx_or_name, str):
            idx_or_name = int(idx_or_name)
        sample = self.db[idx_or_name]
        label = 1 if sample.malware else 0
        inputs = sample.data
        return self.preprocessor(label, name=Name(sample.name), inputs=inputs)

    def __len__(self) -> int:
        return len(self.idx_or_names)


class Preprocessor:

    def __init__(
        self,
        do_parser: bool = True,
        do_entropy: bool = True,
        do_characteristics: bool = True,
        level: HierarchicalLevel | str = HierarchicalLevel.NONE,
        max_length: Optional[int] = None,
    ) -> None:
        self.do_parser = do_parser
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.level = HierarchicalLevel(level)
        self.max_length = max_length
        self.guider = SemanticGuider(do_parser, do_entropy, do_characteristics)
        self.partitioner = StructurePartitioner(HierarchicalLevel(level))

    @property
    def should_lief_parse(self) -> bool:
        return self.do_parser or self.do_entropy or self.do_characteristics or self.level != HierarchicalLevel.NONE

    def __call__(
        self,
        label: int,
        *,
        name: Optional[Name] = None,
        file: Optional[StrPath] = None,
        inputs: Optional[ByteTensor] = None,
    ) -> FSample:
        """
        Args:
            label: integer label for the binary.
            name: name of the binary. If None, `file` must be provided, from which the name is inferred.
            file: path to the binary file. If None, `inputs` and `name` must be provided.
            inputs: byte tensor containing the binary data. If None, `file` must be provided, from which the data is read.
        """
        if (file is None) == (inputs is None):
            raise ValueError("Exactly one of file or inputs must be provided.")
        if file is None and name is None:
            raise ValueError("If file is None, name must be provided.")

        label = torch.tensor(label)
        pe = None

        if inputs is not None:
            if name is None:
                raise ValueError("If file is None, name must be provided.")
            size = len(inputs)
            name = Name(name)
            if self.should_lief_parse:
                pe = _parse_pe_and_get_size(memoryview(inputs.numpy()))[0]
        elif file is not None:
            size = os.path.getsize(file)
            name = Name(file)
            inputs = torch.from_file(str(file), shared=False, size=size, dtype=torch.uint8)
            if self.should_lief_parse:
                pe = _parse_pe_and_get_size(file)[0]
        else:
            raise RuntimeError()

        guides = self.guider(pe, size, inputs)
        structure = self.partitioner(pe, size)

        if self.max_length is not None:
            inputs = inputs[:self.max_length]
            guides = guides.trim(self.max_length)
            structure = structure.trim(self.max_length)
            if not inputs.is_contiguous():
                warnings.warn("`inputs` is not contiguous after trimming. Making it contiguous.")
                inputs = inputs.contiguous()
            if not guides.is_contiguous():
                warnings.warn("`guides` is not contiguous after trimming. Making it contiguous.")
                guides = guides.contiguous()

        return FSample(file, name, label, inputs, guides, structure)


class CollateFn:

    def __init__(self, pin_memory: bool, bitpack: bool, min_length: int = 0) -> None:
        self.pin_memory = pin_memory
        self.bitpack = bitpack
        self.min_length = min_length
        if self.min_length % 8 != 0:
            raise ValueError(f"Due to bitpacking, we require min_length to be a multiple of 8. Got {min_length}.")

    def __call__(self, batch: Sequence[FSample]) -> FSamples:
        file = [s.file for s in batch]
        name = [s.name for s in batch]
        label = torch.stack([s.label for s in batch])
        inputs = CollateFn.get_padded_inputs([s.inputs for s in batch], self.min_length, self.pin_memory, True)
        guides = [s.guides.compress() if self.bitpack else s.guides for s in batch]
        min_length_ = int(self.min_length / 8) if (self.bitpack or guides[0].is_bitpacked) else self.min_length
        guides = SemanticGuides.from_singles(guides, self.pin_memory, min_length_)
        structure = [s.structure for s in batch]
        structure = StructureMaps.from_singles(structure, self.pin_memory, self.min_length)
        return FSamples(file, name, label, inputs, guides, structure)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pin_memory={self.pin_memory}, bitpack={self.bitpack}, min_length={self.min_length})"

    @staticmethod
    def get_padded_inputs(inputs: list[ByteTensor], min_length: int, pin_memory: bool, change_dtype_after_pad: bool = True) -> ShortTensor:
        """
        NOTE: some benchmarks indicate that Tensor.to(torch.int16) is a massive bottleneck. change_dtype_after_pad=True might resolve this.
        """
        if not change_dtype_after_pad:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(torch.int16)
                inputs[i].add_(1)
            inputs_ = pad_sequence(inputs, True, 0, "right", pin_memory, PAD_TO_MULTIPLE_OF, min_length)
            return inputs_

        lengths = [inp.size(0) for inp in inputs]

        inputs_ = pad_sequence(inputs, True, 0, "right", False, PAD_TO_MULTIPLE_OF, min_length)
        inputs_ = inputs_.to(torch.int16)
        for i, l in enumerate(lengths):
            inputs_[i, :l].add_(1)
        # inputs_.add_(1)
        # for i, l in enumerate(lengths):
        #     inputs_[i, l:] = 0

        # Memory pinning must take place after dtype conversion, hence, we used pin_memory=False above.
        if pin_memory:
            inputs_ = inputs_.pin_memory()

        return inputs_


class CollateFnHierarchical:

    def __init__(self, pin_memory: bool, bitpack: bool, num_structures: int, min_lengths: Optional[list[int]] = None) -> None:
        self.pin_memory = pin_memory
        self.bitpack = bitpack
        self.num_structures = num_structures
        self.min_lengths = [0] * num_structures if min_lengths is None else min_lengths
        if any(l % 8 != 0 for l in self.min_lengths):
            raise ValueError(f"Due to bitpacking, we require min_length to be a multiple of 8. Got {self.min_lengths}.")
        if len(self.min_lengths) != self.num_structures:
            raise ValueError(f"min_lengths must have length equal to num_structures. Got {self.min_lengths} and {self.num_structures}.")

    def __call__(self, batch: Sequence[FSample]) -> HSamples:
        file = [s.file for s in batch]
        name = [s.name for s in batch]
        label = torch.stack([s.label for s in batch])

        inputs = []
        guides = []
        for i in range(self.num_structures):
            xs = []
            gs = []
            for j in range(len(batch)):
                ranges = batch[j].structure.index[i]
                x_i_j = torch.cat([batch[j].inputs[lo:hi] for lo, hi in ranges] + [torch.tensor([], dtype=batch[j].inputs.dtype)])
                g_i_j = batch[j].guides.select(ranges=ranges)
                xs.append(x_i_j)
                gs.append(g_i_j.compress() if self.bitpack else g_i_j)
            xs = CollateFn.get_padded_inputs(xs, self.min_lengths[i], self.pin_memory)
            gs = SemanticGuides.from_singles(gs, self.pin_memory, int(self.min_lengths[i] / 8))
            inputs.append(xs)
            guides.append(gs)

        structure = [s.structure for s in batch]
        structure = StructureMaps.from_singles(structure, self.pin_memory)
        return HSamples(file, name, label, inputs, guides, structure)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pin_memory={self.pin_memory}, bitpack={self.bitpack}, num_structures={self.num_structures}, min_lengths={self.min_lengths})"


class CUDAPrefetcher:

    def __init__(self, loader: DataLoader, device: torch.device, num_streams: int) -> None:
        if num_streams > 0 and (device.type != "cuda" or not torch.cuda.is_available()):
            raise ValueError(f"{self.__class__.__name__} with num_streams > 0 requires a CUDA device.")

        self.loader = loader
        self.device = device
        self.num_streams = max(0, int(num_streams))
        self.streams = [torch.cuda.Stream(device=device) for _ in range(self.num_streams)]
        self._rr = 0
        self._buf: deque[tuple[torch.cuda.Stream, FOrHSamples]] = deque()
        self.it: Optional[Iterator[FOrHSamples]] = None
        self._primed = False

    def __contains__(self, item: object) -> bool:
        return item in self.loader

    def warmup(self, preload_batches: Optional[int] = None) -> None:
        """
        Spawn DataLoader workers early and optionally prefetch some batches to device.

        Args:
            preload_batches: how many batches to prefetch to GPU. If None, uses
            num_streams (or 0 if num_streams==0). Use 0 to only spawn workers.
        """
        if self.it is None:
            self.it = iter(self.loader)  # spawns worker processes
        self._buf.clear()
        self._rr = 0

        if preload_batches is None:
            preload_batches = self.num_streams if self.num_streams > 0 else 0

        if self.num_streams > 0 and preload_batches > 0:
            self._preload(preload_batches)
            self._primed = True
        else:
            self._primed = False  # no GPU prefetch, just workers spawned

    def __iter__(self) -> Iterator[FOrHSamples]:
        if not self._primed:
            self.it = iter(self.loader)
            self._buf.clear()
            self._rr = 0
            if self.num_streams > 0:
                self._preload(self.num_streams)
        self._primed = False
        return self

    def __next__(self) -> FOrHSamples:
        if self.it is None:
            raise StopIteration

        # Not using CUDA at all: copy on CPU
        if self.device.type != "cuda":
            try:
                cpu_batch = next(self.it)
            except StopIteration:
                raise StopIteration

            return cpu_batch.decompress()

        curr = torch.cuda.current_stream(self.device)

        # No prefetch streams: copy on current stream
        if self.num_streams == 0:
            try:
                cpu_batch = next(self.it)
            except StopIteration:
                raise StopIteration

            def _to(t: FOrHSamples) -> FOrHSamples:
                return t.to(self.device, non_blocking=True)

            dev_batch: FOrHSamples = tree_map(_to, cpu_batch)
            return dev_batch.decompress()

        # Prefetched path: pop one batch whose H2D was done on its stream
        if not self._buf:
            # Try topping up once in case we finished priming exactly
            self._preload(1)
            if not self._buf:
                raise StopIteration

        copy_stream, dev_batch = self._buf.popleft()
        curr.wait_stream(copy_stream)

        def _record(t: FOrHSamples) -> FOrHSamples:
            if getattr(t, "is_cuda", False):
                t.record_stream(curr)
            return t

        dev_batch = tree_map(_record, dev_batch)

        # Keep the pipeline full
        self._preload(1)

        return dev_batch.decompress()

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loader={self.loader.__class__.__name__}(...), device={self.device}, num_streams={self.num_streams})"

    def _preload(self, n: int) -> None:
        if self.it is None or self.num_streams == 0:
            return

        for _ in range(n):
            try:
                cpu_batch = next(self.it)
            except StopIteration:
                return

            s = self.streams[self._rr]
            self._rr = (self._rr + 1) % self.num_streams

            with torch.cuda.stream(s):
                def _to(t: FOrHSamples) -> FOrHSamples:
                    return t.to(self.device, non_blocking=True)
                dev_batch: FOrHSamples = tree_map(_to, cpu_batch)

            self._buf.append((s, dev_batch))

    @property
    def dataset(self) -> Dataset:
        return self.loader.dataset


class GroupedLengthBatchSampler(Sampler[list[int]]):  # type: ignore[misc]

    def __init__(self, chunks: Sequence[IntTensor | list[int]], first: bool, shuffle: bool, drop_last: bool) -> None:
        self.chunks = chunks
        self.first = first
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.length = self._get_length()
        self.order: Optional[list[int]] = self._get_chunk_order() if shuffle is False else None

    def __iter__(self) -> Iterator[list[int]]:
        order = self.order if not self.shuffle else self._get_chunk_order()
        if order is None:
            raise RuntimeError("Order was not established.")
        for i in order:
            indices = self.chunks[i]
            if isinstance(indices, IntTensor):
                indices = indices.tolist()
            yield indices

    def __len__(self) -> int:
        return self.length

    @classmethod
    def from_lengths(cls, batch_size: int, lengths: Sequence[int], *, first: bool = False, shuffle: bool = False, drop_last: bool = False) -> GroupedLengthBatchSampler:
        lengths = torch.tensor(lengths)
        idxsorted = torch.argsort(lengths, descending=True)
        chunks = torch.chunk(idxsorted, math.ceil(len(lengths) / batch_size))
        return cls(chunks, first, shuffle, drop_last)

    def _get_length(self) -> int:
        if not self.drop_last:
            return len(self.chunks)
        bsizes = [len(c) for c in self.chunks]
        csizes = Counter(bsizes)
        if len(csizes) == 1:
            return len(self.chunks)
        if len(csizes) == 2 and csizes.most_common(None)[-1][1] == 1:
            return len(self.chunks) - 1
        raise RuntimeError(f"Unexpected batch sizes: {csizes} when self.drop_last is {self.drop_last}.")

    def _get_chunk_order(self) -> list[int]:
        options = torch.arange(len(self.chunks))
        if self.drop_last:
            bsizes = [len(c) for c in self.chunks]
            csizes = Counter(bsizes)
            if len(csizes) == 1:
                pass
            elif len(csizes) == 2 and csizes.most_common(None)[-1][1] == 1:
                idx = torch.tensor(bsizes) != csizes.most_common(None)[-1][0]
                options = options[idx]
            else:
                raise RuntimeError(f"Unexpected batch sizes: {csizes} when self.drop_last is {self.drop_last}.")

        if self.first:
            order = options[torch.randperm(len(self) - 1) + 1]
            order = torch.cat((torch.tensor([0]), order))
        else:
            order = options[torch.randperm(len(self))]

        order_: list[int] = order.tolist()
        return order_
