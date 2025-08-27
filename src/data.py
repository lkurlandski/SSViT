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

from src.utils import check_tensor
from src.utils import packbits
from src.utils import unpackbits
from src.utils import pad_sequence
from src.utils import mask_select_packed
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

    @property
    def is_cuda(self) -> bool:
        is_cuda = [t.is_cuda for t in (self.parse, self.entropy, self.characteristics) if t is not None]
        if all(is_cuda):
            return True
        if not any(is_cuda):
            return False
        raise RuntimeError(f"Unexpected tensor device state: {is_cuda=}")

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

    def select(self, idx: BoolTensor) -> Self:
        check_tensor(idx, (None,), torch.bool)

        entropy = None
        if self.entropy is not None:
            entropy = self.entropy[idx]

        if not self.is_bitpacked:
            parse = None
            if self.parse is not None:
                parse = self.parse[idx]
            characteristics = None
            if self.characteristics is not None:
                characteristics = self.characteristics[idx]
            return self.__class__(parse, entropy, characteristics)

        # For bitpacked data, we assume idx corresponds to the unpacked data.
        if len(idx) % 8 != 0:
            idx = torch.nn.functional.pad(idx, (0, 8 - (idx.size(0) % 8)), "constant", False)

        parse = None
        if self.parse is not None:
            if idx.shape[self.length_axis] != self.parse.shape[self.length_axis] * 8:
                raise RuntimeError("Index length does not match unpacked data length.")
            parse = mask_select_packed(self.parse, idx, self.length_axis)

        characteristics = None
        if self.characteristics is not None:
            if idx.shape[self.length_axis] != self.characteristics.shape[self.length_axis] * 8:
                raise RuntimeError("Index length does not match unpacked data length.")
            characteristics = mask_select_packed(self.characteristics, idx, self.length_axis)

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

    # NOTE: compression/decompression with bitpacking will implicitly zero pad tensors to a multiple of 8.
    # NOTE: the boolean tensors are used only for slicing, so its not advisable to compress, pin, or move them.

    index: BoolTensor | CharTensor
    lexicon: Mapping[int, HierarchicalStructure]

    def __init__(self, index: BoolTensor | CharTensor, lexicon: Mapping[int, HierarchicalStructure]) -> None:
        self.index = index
        self.lexicon = lexicon
        self.verify_inputs()

    @property
    @abstractmethod
    def length_axis(self) -> int:
        ...

    @abstractmethod
    def verify_inputs(self) -> None:
        ...

    @property
    def is_cuda(self) -> bool:
        return self.index.is_cuda  # type: ignore[no-any-return]

    def record_stream(self, s: torch.cuda.Stream) -> None:
        self.index.record_stream(s)

    def clone(self) -> Self:
        self.index = self.index.clone()
        self.lexicon = deepcopy(self.lexicon)
        return self

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        self.index = self.index.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> Self:
        self.index = self.index.pin_memory()
        return self

    def compress(self) -> Self:
        self.index = packbits(self.index, self.length_axis)
        return self

    def decompress(self) -> Self:
        self.index = unpackbits(self.index, self.length_axis)
        return self

    def trim(self, length: Optional[int]) -> Self:
        if length is None:
            return self
        length_ = length
        if self.index.dtype == torch.uint8:
            length_ = math.ceil(length / 8)
        slice_ = [slice(length_) if i == self.length_axis else slice(None) for i in range(self.index.ndim)]
        self.index = self.index[tuple(slice_)]
        return self


class StructureMap(_StructureMapOrStructureMaps):

    @property
    def length_axis(self) -> int:
        return 0

    def verify_inputs(self) -> None:
        check_tensor(self.index, (None, None), (torch.bool, torch.uint8))
        if self.index.shape[1] != len(list(self.lexicon.keys())):
            raise ValueError("StructureMap index does not match lexicon keys.")


class StructureMaps(_StructureMapOrStructureMaps):

    @property
    def length_axis(self) -> int:
        return 1

    def verify_inputs(self) -> None:
        check_tensor(self.index, (None, None, None), (torch.bool, torch.uint8))
        if self.index.shape[2] != len(list(self.lexicon.keys())):
            raise ValueError("BatchedStructureMap index does not match lexicon keys.")

    @classmethod
    def from_singles(cls, maps: Sequence[StructureMap], pin_memory: bool = False, min_length: int = 0) -> _StructureMapOrStructureMaps:
        if len(maps) == 0:
            raise ValueError("Cannot create BatchedStructureMap from empty list.")
        lexicon = maps[0].lexicon
        for m in maps[1:]:
            if m.lexicon != lexicon:
                raise ValueError("All StructureMaps must have the same lexicon to be batched.")
        padding_value = False if maps[0].index.dtype == torch.bool else 0.0
        pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if maps[0].index.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
        index = pad_sequence([m.index for m in maps], True, padding_value, "right", pin_memory, pad_to_multiple_of, min_length)
        return cls(index, lexicon)


class StructurePartitioner:
    """
    Partitions a binary into hierarchical structures.
    """

    def __init__(self, level: HierarchicalLevel = HierarchicalLevel.NONE) -> None:
        self.level = level

    def __call__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> StructureMap:
        if self.level == HierarchicalLevel.NONE:
            size = _get_size_of_liefparse(data) if size is None else size
            index = torch.full((size, 1), dtype=bool, fill_value=True)
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

        index = torch.full((parser.size, len(structures)), dtype=bool, fill_value=False)
        lexicon = {}
        for i, s in enumerate(structures):
            bounds = parser(s)
            for l, u in bounds:
                index[l:u, i] = True
            lexicon[i] = s
        return StructureMap(index, lexicon)


class _FSampleOrSamples(ABC):
    file: StrPath | list[StrPath]
    name: Name | list[Name]
    label: ShortTensor | IntTensor | LongTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: _SemanticGuideOrSemanticGuides
    structure: _StructureMapOrStructureMaps

    # NOTE: the _StructureMapOrStructureMaps does not need to be involved in GPU operations.
    # TODO: some of this functionality is redundant or unnecessary, since the original design for the
        # hierarchical pipeline did not work well and we had to implement the _HSampleOrSamples class.

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
        if self.label.is_cuda and self.inputs.is_cuda and self.guides.is_cuda and not self.structure.is_cuda:
            return True
        if not self.label.is_cuda and not self.inputs.is_cuda and not self.guides.is_cuda and not self.structure.is_cuda:
            return False
        raise RuntimeError(f"Unexpected tensor device state: {self.label.is_cuda=}, {self.inputs.is_cuda=}, {self.guides.is_cuda=}, {self.structure.is_cuda=}")

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
        return self.preprocessor(self.files[i], self.labels[i])

    def __len__(self) -> int:
        return len(self.files)


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

    def __call__(self, file: StrPath, label: int) -> FSample:
        name = Name(file)
        label = torch.tensor(label)
        inputs = torch.from_file(str(file), shared=False, size=os.path.getsize(file), dtype=torch.uint8)

        if self.do_parser or self.do_entropy or self.do_characteristics or self.level != HierarchicalLevel.NONE:
            pe, size = _parse_pe_and_get_size(file)
        else:
            pe, size = None, os.path.getsize(file)

        guides = self.guider(pe, size, inputs)
        structure = self.partitioner(pe, size)

        # NOTE: this is not meant to be efficient, since its only for debugging.
        if self.max_length is not None:
            inputs = inputs[:self.max_length]
            guides = guides.trim(self.max_length)
            structure = structure.trim(self.max_length)

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
        inputs = [s.inputs.to(torch.int16) + 1 for s in batch]
        inputs = pad_sequence(inputs, True, 0, "right", self.pin_memory, PAD_TO_MULTIPLE_OF, self.min_length)
        guides = [s.guides.compress() if self.bitpack else s.guides for s in batch]
        min_length_ = int(self.min_length / 8) if (self.bitpack or guides[0].is_bitpacked) else self.min_length
        guides = SemanticGuides.from_singles(guides, self.pin_memory, min_length_)
        structure = [s.structure for s in batch]
        structure = StructureMaps.from_singles(structure, self.pin_memory, self.min_length)
        return FSamples(file, name, label, inputs, guides, structure)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pin_memory={self.pin_memory}, bitpack={self.bitpack}, min_length={self.min_length})"


class CollateFnHierarchical:

    def __init__(self, pin_memory: bool, bitpack: bool, num_structures: int, min_lengths: Optional[list[int]] = None) -> None:
        self.pin_memory = pin_memory
        self.bitpack = bitpack
        self.num_structures = num_structures
        self.min_lengths = [0] * num_structures if min_lengths is None else min_lengths
        if any(l % 8 != 0 for l in self.min_lengths):
            raise ValueError(f"Due to bitpacking, we require min_length to be a multiple of 8. Got {self.min_lengths}.")

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
                idx = batch[j].structure.index[:, i]
                x_i_j = batch[j].inputs[idx]
                g_i_j = batch[j].guides.select(idx)
                xs.append(x_i_j.to(torch.int16) + 1)
                gs.append(g_i_j.compress() if self.bitpack else g_i_j)
            xs = pad_sequence(xs, True, 0, "right", self.pin_memory, PAD_TO_MULTIPLE_OF, self.min_lengths[i])
            gs = SemanticGuides.from_singles(gs, self.pin_memory, int(self.min_lengths[i] / 8))
            inputs.append(xs)
            guides.append(gs)

        structure = [s.structure for s in batch]
        structure = StructureMaps.from_singles(structure, self.pin_memory)
        return HSamples(file, name, label, inputs, guides, structure)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pin_memory={self.pin_memory}, bitpack={self.bitpack}, num_structures={self.num_structures}, min_lengths={self.min_lengths})"


class CUDAPrefetcher:

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch: Optional[FOrHSamples] = None

    def __contains__(self, item: object) -> bool:
        return item in self.loader

    def __iter__(self) -> Iterator[FOrHSamples]:
        self.it = iter(self.loader)
        self.next_batch = None
        self._preload()
        return self

    def __next__(self) -> FOrHSamples:
        if self.next_batch is None:
            raise StopIteration

        curr = torch.cuda.current_stream(self.device)
        curr.wait_stream(self.stream)   # fence: H2D of next_batch is now done (or almost)

        # Protect CUDA storage from early reuse by other streams
        def _record(t: FOrHSamples) -> FOrHSamples:
            if t.is_cuda:
                t.record_stream(curr)
            return t
        batch: FOrHSamples = tree_map(_record, self.next_batch)

        # Immediately begin prefetch for the subsequent batch (max overlap)
        self._preload()

        # Do GPU-side transforms after the wait/record
        return batch.decompress()       # safe here; runs on curr stream

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loader={self.loader}, device={self.device})"

    def _preload(self) -> None:
        try:
            cpu_batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            def _to(t: FOrHSamples) -> FOrHSamples:
                return t.to(self.device, non_blocking=True)
            self.next_batch = tree_map(_to, cpu_batch)

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
