"""
Manage data and datasets.
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import atexit
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
from itertools import batched
import math
import os
from pathlib import Path
import random
import sys
from typing import Any
from typing import Literal
from typing import Generic
from typing import Optional
from typing import Self
from typing import TypeVar
import warnings

import lief
import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import get_worker_info
from torch.utils._pytree import tree_map

from src.binanal import LEVEL_STRUCTURE_MAP
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
from src.binentropy import _check_inputs as check_inputs_for_entropy
from src.bitpacking import packbits
from src.bitpacking import unpackbits
from src.bitpacking import slice_bitpacked_tensor
from src.simpledb import Sample as SimpleDBSample
from src.simpledb import SimpleDB
from src.simpledb import SimpleDBIterator
from src.simpledb import SimpleDBReader
from src.simpledb import MetadataDB
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
    """
    Semantic guides to acompany a byte stream.

    Both `parse` and `characteristics` are boolean tensors, where True indicates the presence of a feature
    and False indicates its absence. For compression and data transfer purposes, these can be stored as bitpacked
    uint8 Tensors. If uint8, these are treated internally very differently for a variety of operations. For
    learning, these are typically going to be converted to floating point Tensors, where False -> 0.0 and True -> 1.0.

    NOTE: compression/decompression with bitpacking will implicitly zero pad tensors to a multiple of 8.

    `entropy` is a floating point Tensor, typically float16 for compression and data transfer, and float32 for learning.
    """

    parse: Optional[Tensor]
    entropy: Optional[Tensor]
    characteristics: Optional[Tensor]

    def __init__(self, parse: Optional[Tensor], entropy: Optional[Tensor], characteristics: Optional[Tensor]) -> None:
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
    def is_cuda(self) -> Optional[bool]:
        is_cuda = [t.is_cuda for t in (self.parse, self.entropy, self.characteristics) if t is not None]
        if len(is_cuda) == 0:
            return None
        if all(is_cuda):
            return True
        if not any(is_cuda):
            return False
        raise RuntimeError(f"Unexpected tensor device state: {is_cuda=}")

    def contiguous(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            parse = parse.contiguous()
        if entropy is not None:
            entropy = entropy.contiguous()
        if characteristics is not None:
            characteristics = characteristics.contiguous()
        return self.__class__(parse, entropy, characteristics)

    def detach(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            parse = parse.detach()
        if entropy is not None:
            entropy = entropy.detach()
        if characteristics is not None:
            characteristics = characteristics.detach()
        return self.__class__(parse, entropy, characteristics)

    def record_stream(self, s: torch.cuda.Stream) -> None:
        if self.parse is not None:
            self.parse.record_stream(s)
        if self.entropy is not None:
            self.entropy.record_stream(s)
        if self.characteristics is not None:
            self.characteristics.record_stream(s)

    def clone(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            parse = parse.clone()
        if entropy is not None:
            entropy = entropy.clone()
        if characteristics is not None:
            characteristics = characteristics.clone()
        return self.__class__(parse, entropy, characteristics)

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype] = None, non_blocking: bool = False) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            if self.is_bitpacked and dtype is not None:
                warnings.warn("Converting bitpacked tensor's dtype.")
            parse = parse.to(device, dtype, non_blocking=non_blocking)
        if entropy is not None:
            entropy = entropy.to(device, dtype, non_blocking=non_blocking)
        if characteristics is not None:
            if self.is_bitpacked and dtype is not None:
                warnings.warn("Converting bitpacked tensor's dtype.")
            characteristics = characteristics.to(device, dtype, non_blocking=non_blocking)
        return self.__class__(parse, entropy, characteristics)

    def pin_memory(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            if parse.is_cuda:
                raise RuntimeError("Cannot pin memory of a CUDA tensor.")
            parse = parse.pin_memory()
        if entropy is not None:
            if entropy.is_cuda:
                raise RuntimeError("Cannot pin memory of a CUDA tensor.")
            entropy = entropy.pin_memory()
        if characteristics is not None:
            if characteristics.is_cuda:
                raise RuntimeError("Cannot pin memory of a CUDA tensor.")
            characteristics = characteristics.pin_memory()
        return self.__class__(parse, entropy, characteristics)

    def compress(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None and parse.dtype == torch.bool:
            parse = packbits(parse, axis=self.length_axis)
        if entropy is not None and entropy.dtype != torch.float16:
            entropy = entropy.to(torch.float16)
        if characteristics is not None and characteristics.dtype == torch.bool:
            characteristics = packbits(characteristics, axis=self.length_axis)
        return self.__class__(parse, entropy, characteristics)

    def decompress(self) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics
        if parse is not None:
            if parse.dtype == torch.uint8:
                parse = unpackbits(parse, axis=self.length_axis)
            parse = parse.to(torch.float32)
        if entropy is not None and entropy.dtype != torch.float32:
            entropy = entropy.to(torch.float32)
        if characteristics is not None:
            if characteristics.dtype == torch.uint8:
                characteristics = unpackbits(characteristics, axis=self.length_axis)
            characteristics = characteristics.to(torch.float32)
        return self.__class__(parse, entropy, characteristics)

    def trim(self, length: Optional[int]) -> Self:
        parse, entropy, characteristics = self.parse, self.entropy, self.characteristics

        if length is None:
            return self.__class__(parse, entropy, characteristics)

        if parse is not None:
            length_ = length
            if parse.dtype == torch.uint8:
                length_ = math.ceil(length / 8)
            slice_ = [slice(length_) if i == self.length_axis else slice(None) for i in range(parse.ndim)]
            parse = parse[tuple(slice_)]

        if entropy is not None:
            slice_ = [slice(length) if i == self.length_axis else slice(None) for i in range(entropy.ndim)]
            parse = entropy[tuple(slice_)]

        if characteristics is not None:
            length_ = length
            if characteristics.dtype == torch.uint8:
                length_ = math.ceil(length / 8)
            slice_ = [slice(length_) if i == self.length_axis else slice(None) for i in range(characteristics.ndim)]
            characteristics = characteristics[tuple(slice_)]

        return self.__class__(parse, entropy, characteristics)

    def select(self, *, mask: Optional[Tensor] = None, idx: Optional[Tensor] = None, ranges: Optional[list[tuple[int, int]]] = None) -> Self:
        """
        Select a subset of the data along the length axis.

        NOTE: if data is bitpacked, the selectors are assumed to correspond to the unpacked data.
        """
        if mask is not None:
            check_tensor(mask, (None,), torch.bool)
        if idx is not None:
            check_tensor(idx, (None,), (torch.int32, torch.int64))

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

    def build_allguides(self) -> Optional[Tensor]:
        # FIXME: this entails some massive copies, so we're going to have to optimize this later.

        avail = [t is not None for t in (self.parse, self.entropy, self.characteristics)]

        # If no guides are available, return None.
        if not any(avail):
            return None

        # If only one guide is available, return it directly without any copying.
        if sum(avail) == 1:
            if self.parse is not None:
                return self.parse
            if self.entropy is not None:
                return self.entropy
            if self.characteristics is not None:
                return self.characteristics
            raise RuntimeError()

        # Otherwise, concatenate all available guides along the guides axis.
        dim = self.length_axis + 1
        gs = [g for g in (self.parse, self.entropy, self.characteristics) if g is not None]
        gs = [g.unsqueeze(dim=dim) if g.ndim < dim + 1 else g for g in gs]
        try:
            allguides = torch.cat(gs, dim=self.length_axis + 1)
        except Exception:
            print(f"{dim=} {self.length_axis=} {len(gs)=}")
            for g in gs:
                print(g.shape, g.dtype, g.device)
            raise
        return allguides


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

        def checked(ts: list[Optional[Tensor]]) -> list[Tensor]:
            if any(t is None for t in ts):
                raise ValueError("All guides must have the same fields to be batched.")
            return ts  # type: ignore[return-value]

        parse = None
        if guides[0].parse is not None:
            padding_value = False if guides[0].parse.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].parse.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            parse = pad_sequence(checked([g.parse for g in guides]), True, padding_value, "right", pin_memory, pad_to_multiple_of, min_length)
        entropy = None
        if guides[0].entropy is not None:
            entropy = pad_sequence(checked([g.entropy for g in guides]), True, 0.0, "right", pin_memory, PAD_TO_MULTIPLE_OF, min_length)
        characteristics = None
        if guides[0].characteristics is not None:
            padding_value = False if guides[0].characteristics.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].characteristics.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            characteristics = pad_sequence(checked([g.characteristics for g in guides]), True, padding_value, "right", pin_memory, pad_to_multiple_of, min_length)

        return cls(parse, entropy, characteristics)


class SemanticGuider:
    """
    Semantic guides to acompany a byte stream.
    """

    def __init__(
        self,
        do_parse: bool = False,
        do_entropy: bool = False,
        which_characteristics: Sequence[lief.PE.Section.CHARACTERISTICS] = tuple(),
        radius: int = 256,
        stride: int = 1,
        simple: bool = False,
        use_packed: bool = True,
    ) -> None:
        self.do_parse = do_parse
        self.do_entropy = do_entropy
        self.which_characteristics = which_characteristics
        self.radius = radius
        self.stride = stride
        self.simple = simple
        self.use_packed = use_packed

    def __call__(
        self,
        data: Optional[LiefParse | lief.PE.Binary] = None,
        size: Optional[int] = None,
        inputs: Optional[Tensor] = None,
    ) -> SemanticGuide:

        parse = None
        if self.do_parse:
            if data is None:
                raise ValueError("Data must be provided for parse computation.")
            parse = self._get_parse(data, size)

        entropy = None
        if self.do_entropy:
            if inputs is None:
                raise RuntimeError("Inputs must be provided for entropy computation.")
            entropy = self._get_entropy(inputs)

        characteristics = None
        if self.which_characteristics:
            if data is None:
                raise ValueError("Data must be provided for characteristics computation.")
            characteristics = self._get_characteristics(data, size)

        return SemanticGuide(parse, entropy, characteristics)

    def _get_parse(self, data: LiefParse | lief.PE.Binary, size: Optional[int]) -> Tensor:
        parse = ParserGuider(data, size)(simple=self.simple)
        parse = torch.from_numpy(parse)
        return parse

    def _get_entropy(self, inputs: Tensor) -> Tensor:
        check_tensor(inputs, (None,), torch.uint8)
        outdtype = torch.float16
        if not check_inputs_for_entropy(len(inputs), self.radius, self.stride, errors="pass"):
            return torch.zeros_like(inputs, dtype=outdtype)
        inputs = inputs.numpy(force=True)
        entropy = compute_entropy_rolling_numpy(inputs, self.radius, self.stride)
        entropy = torch.from_numpy(entropy).to(outdtype)
        return entropy

    def _get_characteristics(self, data: LiefParse | lief.PE.Binary, size: Optional[int]) -> Tensor:
        characteristics = CharacteristicGuider(data, size, self.use_packed, self.which_characteristics)()
        characteristics = torch.from_numpy(characteristics)
        return characteristics


class _StructureMapOrStructureMaps(ABC):

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
    def from_singles(cls, maps: Sequence[StructureMap], pin_memory: bool = False, min_length: int = 0) -> StructureMaps:
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
        lexicon: Mapping[int, HierarchicalStructure]
        index: list[list[tuple[int, int]]]

        if self.level == HierarchicalLevel.NONE:
            if size is None:
                if not isinstance(data, LiefParse):
                    raise TypeError("Data must be LiefParse to infer size when level is NONE.")
                size = _get_size_of_liefparse(data)
            index = [[(0, size)]]
            lexicon = {0: HierarchicalStructureNone.ANY}
            return StructureMap(index, lexicon)

        if data is None:
            raise ValueError("Data must be provided for structure partitioning.")

        parser = StructureParser(data, size)

        index = []
        lexicon = {}
        for i, s in enumerate(LEVEL_STRUCTURE_MAP[self.level]):
            index.append(parser(s))
            lexicon[i] = s
        return StructureMap(index, lexicon)


F = TypeVar("F")                                        # file
N = TypeVar("N")                                        # name
G = TypeVar("G", bound=_SemanticGuideOrSemanticGuides)  # guides
S = TypeVar("S", bound=_StructureMapOrStructureMaps)    # structures


class _FSampleOrSamples(Generic[F, N, G, S], ABC):

    file: F
    name: N
    label: Tensor
    inputs: Tensor
    guides: G
    structure: S

    def __init__(
        self,
        file: F,
        name: N,
        label: Tensor,
        inputs: Tensor,
        guides: G,
        structure: S,
    ) -> None:
        self.file = file
        self.name = name
        self.label = label
        self.inputs = inputs
        self.guides = guides
        self.structure = structure
        self.verify_inputs()

    def __iter__(self) -> Iterator[F | N | Tensor | Tensor | G | S]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))

    @property
    def parse(self) -> Optional[Tensor]:
        return self.guides.parse

    @property
    def entropy(self) -> Optional[Tensor]:
        return self.guides.entropy

    @property
    def characteristics(self) -> Optional[Tensor]:
        return self.guides.characteristics

    @property
    def allguides(self) -> Optional[Tensor]:
        return self.guides.build_allguides()

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def verify_inputs(self) -> None:
        ...

    @property
    def is_cuda(self) -> bool:
        if self.label.is_cuda and self.inputs.is_cuda and (self.guides.is_cuda is True or self.guides.is_cuda is None):
            return True
        if not self.label.is_cuda and not self.inputs.is_cuda and (self.guides.is_cuda is False or self.guides.is_cuda is None):
            return False
        raise RuntimeError(f"Unexpected tensor device state: {self.label.is_cuda=}, {self.inputs.is_cuda=}, {self.guides.is_cuda=}")

    def record_stream(self, s: torch.cuda.Stream) -> None:
        self.label.record_stream(s)
        self.inputs.record_stream(s)
        self.guides.record_stream(s)

    def detach(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.detach()
        inputs = self.inputs.detach()
        guides = self.guides.detach()
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def clone(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        inputs = self.inputs.clone()
        label = self.label.clone()
        guides = self.guides.clone()
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(device, non_blocking=non_blocking)
        inputs = self.inputs.to(device, non_blocking=non_blocking)
        guides = self.guides.to(device, non_blocking=non_blocking)
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def pin_memory(self) -> Self:
        if self.label.is_cuda or self.inputs.is_cuda or self.guides.is_cuda:
            raise RuntimeError("Cannot pin memory of a CUDA tensor.")
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.pin_memory()
        inputs = self.inputs.pin_memory()
        guides = self.guides.pin_memory()
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def compress(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(torch.int16)
        inputs = self.inputs.to(torch.int16)
        guides = self.guides.compress()
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def decompress(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(torch.int64)
        inputs = self.inputs.to(torch.int32)
        guides = self.guides.decompress()
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def finalize(self, device: torch.device, ftype: torch.dtype) -> Self:
        new: Self = self
        new = new.to(device, non_blocking=True)
        new = new.decompress()
        new.guides = new.guides.to(None, ftype)
        return new


class FSample(_FSampleOrSamples[StrPath, Name, SemanticGuide, StructureMap]):

    def __len__(self) -> int:
        return 1

    def verify_inputs(self) -> None:
        check_tensor(self.label, (), (torch.int16, torch.int32, torch.int64))
        check_tensor(self.inputs, (None,), (torch.uint8, torch.int16, torch.int32, torch.int64))


class FSamples(_FSampleOrSamples[Sequence[StrPath], Sequence[Name], SemanticGuides, StructureMaps]):

    def __len__(self) -> int:
        return len(self.file)

    def verify_inputs(self) -> None:
        check_tensor(self.label, (None,), (torch.int16, torch.int32, torch.int64))
        check_tensor(self.inputs, (self.label.shape[0], None), (torch.uint8, torch.int16, torch.int32, torch.int64))


class _HSampleOrSamples(Generic[F, N, G, S], ABC):
    """
    A similar container as _FSampleOrSamples, but specifically for hierarchical models.
    The main difference is that the guides and structure are lists, one per hierarchical structure.

    NOTE: can lists of tensors be efficiently shared across processes?
    """
    file: F
    name: N
    label: Tensor
    inputs: Sequence[Tensor]
    guides: Sequence[G]
    structure: S

    def __init__(
        self,
        file: F,
        name: N,
        label: Tensor,
        inputs: Sequence[Tensor],
        guides: Sequence[G],
        structure: S,
    ) -> None:
        self.file = file
        self.name = name
        self.label = label
        self.inputs = inputs
        self.guides = guides
        self.structure = structure
        self.verify_inputs()

    def __iter__(self) -> Iterator[F | N | Tensor | Sequence[Tensor] | Sequence[G] | S]:
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

    @property
    def allguides(self) -> list[Optional[Tensor]]:
        return [g.build_allguides() for g in self.guides]

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
            if self.inputs[i].is_cuda and (self.guides[i].is_cuda is True or self.guides[i].is_cuda is None):
                continue
            break
        else:
            if self.label.is_cuda:
                return True

        for i in range(self.num_structures):
            if not self.inputs[i].is_cuda and (self.guides[i].is_cuda is False or self.guides[i].is_cuda is None):
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

    def detach(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.detach()
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].detach())
            guides.append(self.guides[i].detach())
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def clone(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.clone()
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].clone())
            guides.append(self.guides[i].clone())
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(device, non_blocking=non_blocking)
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].to(device, non_blocking=non_blocking))
            guides.append(self.guides[i].to(device, non_blocking=non_blocking))
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def pin_memory(self) -> Self:
        if self.label.is_cuda or any(inp.is_cuda for inp in self.inputs) or any(guide.is_cuda for guide in self.guides):
            raise RuntimeError("Cannot pin memory of a CUDA tensor.")
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.pin_memory()
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].pin_memory())
            guides.append(self.guides[i].pin_memory())
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def compress(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(torch.int16)
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].to(torch.int16))
            guides.append(self.guides[i].compress())
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def decompress(self) -> Self:
        file = deepcopy(self.file)
        name = deepcopy(self.name)
        label = self.label.to(torch.int64)
        inputs = []
        guides = []
        for i in range(self.num_structures):
            inputs.append(self.inputs[i].to(torch.int32))
            guides.append(self.guides[i].decompress())
        structure = self.structure.clone()
        return self.__class__(file, name, label, inputs, guides, structure)

    def finalize(self, device: torch.device, ftype: torch.dtype) -> Self:
        new: Self = self
        new = new.to(device, non_blocking=True)
        new = new.decompress()
        new.guides = [g.to(None, ftype) for g in new.guides]
        return new


class HSample(_HSampleOrSamples[StrPath, Name, SemanticGuide, StructureMap]):

    def __len__(self) -> int:
        return 1

    def verify_inputs(self) -> None:
        super().verify_inputs()
        check_tensor(self.label, (), (torch.int16, torch.int32, torch.int64))
        for inp in self.inputs:
            check_tensor(inp, (None,), (torch.uint8, torch.int16, torch.int32, torch.int64))


class HSamples(_HSampleOrSamples[Sequence[StrPath], Sequence[Name], SemanticGuides, StructureMaps]):

    def __len__(self) -> int:
        return len(self.file)

    def verify_inputs(self) -> None:
        super().verify_inputs()
        check_tensor(self.label, (None,), (torch.int16, torch.int32, torch.int64))
        for inp in self.inputs:
            check_tensor(inp, (self.label.shape[0], None), (torch.uint8, torch.int16, torch.int32, torch.int64))


FOrHSample  = FSample | HSample
FOrHSamples = FSamples | HSamples


class IterableSimpleDBDataset(IterableDataset[FSample]):

    def __init__(
        self,
        db: SimpleDB,
        metadb: MetadataDB,
        preprocessor: Preprocessor,
        shards: Optional[list[int]] = None,
        *,
        shuffle: bool = False,
        poolsize: int = 16,
    ) -> None:
        self.db = db
        self.metadb = metadb
        self.preprocessor = preprocessor
        self.shards = list(range(db.num_shards)) if shards is None else deepcopy(shards)
        self.shuffle = shuffle
        self.poolsize = poolsize

    def __iter__(self) -> Iterator[FSample]:
        reader = SimpleDBIterator(self.db)

        shards = self._get_local_shards()

        # If not shuffling, just yield samples in order.
        if not self.shuffle:
            for sample in reader.iter(shards):
                meta = self.metadb.get(sample.name, reader.curshardidx)
                yield self.preprocess(sample, meta)
            return

        # Otherwise, randomize the order of the shards deterministically.
        seed = self._get_local_seed()
        rng = random.Random(int(seed))
        if self.shuffle:
            rng.shuffle(shards)

        # Yield samples randomly using a reservoir sampling.
        pool: list[tuple[SimpleDBSample, pl.DataFrame]] = []
        for sample in reader.iter(shards):
            meta = self.metadb.get(sample.name, reader.curshardidx)
            if len(pool) < self.poolsize:
                pool.append((sample, meta))
                continue
            i = random.randint(0, len(pool) - 1)
            yield self.preprocess(*pool[i])
            pool[i] = (sample, meta)

        # Yield the remaining samples in the pool.
        while len(pool) > 0:
            i = random.randint(0, len(pool) - 1)
            yield self.preprocess(*pool[i])
            pool.pop(i)

    def __len__(self) -> int:
        return sum(self.db.num_samples(self._get_local_shards()))

    def preprocess(self, sample: SimpleDBSample, meta: pl.DataFrame) -> FSample:
        return self.preprocessor(
            sample.name,
            1 if sample.malware else 0,
            sample.data,
            meta,
        )

    def _get_local_shards(self) -> list[int]:
        if (worker_info := get_worker_info()) is None:
            return self.shards
        if len(self.shards) < worker_info.num_workers:
            raise ValueError(f"Number of shards ({len(self.shards)}) is less than number of workers ({worker_info.num_workers}).")
        per_worker = math.ceil(len(self.shards) / worker_info.num_workers)
        return list(list(batched(self.shards, per_worker))[worker_info.id])

    def _get_local_seed(self) -> int:
        if (worker_info := get_worker_info()) is None:
            return torch.initial_seed()
        if not hasattr(worker_info, "seed"):
            return torch.initial_seed() + worker_info.id
        return worker_info.seed


class Preprocessor:

    def __init__(
        self,
        do_entropy: bool = False,
        which_characteristics: Sequence[lief.PE.Section.CHARACTERISTICS] = tuple(),
        level: HierarchicalLevel | str = HierarchicalLevel.NONE,
        max_length: Optional[int] = None,
        unsafe: bool = False,
    ) -> None:
        self.do_entropy = do_entropy
        self.which_characteristics = which_characteristics
        self.level = HierarchicalLevel(level)
        self.max_length = max_length
        self.unsafe = unsafe
        self.guider = SemanticGuider(do_entropy=do_entropy, which_characteristics=which_characteristics)
        self.partitioner = StructurePartitioner(HierarchicalLevel(level))

    def __call__(self, name: str, label: int, data: bytes, meta: pl.DataFrame) -> FSample:
        mv = memoryview(data)[0:self.max_length]
        size = len(mv)

        buffer: bytearray | memoryview
        if self.unsafe:
            warnings.filterwarnings("ignore", message=r"The given buffer is not writable.*", category=UserWarning)
            buffer = mv
        else:
            buffer = bytearray(mv)

        file = ""
        name = Name(name)
        label = torch.tensor(label, dtype=torch.int64)
        inputs = torch.frombuffer(buffer, dtype=torch.uint8)
        guides = SemanticGuide(None, None, None)
        if self.do_entropy:
            guides.entropy = self.guider._get_entropy(inputs)
        if self.which_characteristics:
            guides.characteristics = self.get_characteristics(meta, size)
        structure = self.get_structure(meta, size)

        return FSample(
            file=file,
            name=name,
            label=label,
            inputs=inputs,
            guides=guides,
            structure=structure,
        )

    def get_characteristics(self, meta: pl.DataFrame, size: int) -> torch.Tensor:
        df_sec = meta.filter(
            pl.col("group") == "partitions",
            pl.col("level") == "COARSE",
            pl.col("label") == "SECTION",
        )
        df_chr = meta.filter(
            pl.col("group") == "characteristics",
        )

        n_sec = len(df_sec)
        n_chr = len(self.which_characteristics)

        offsets = np.empty(n_sec, dtype=np.int64)
        sizes   = np.empty(n_sec, dtype=np.int64)
        flags   = np.zeros((n_sec, n_chr), dtype=np.uint8)

        for i, row in enumerate(df_sec.iter_rows(named=True)):
            s = row["start"]
            e = row["end"]
            offsets[i] = s
            sizes[i]   = e - s

            df = df_chr.filter(
                pl.col("start") == s,
                pl.col("end") == e,
            )
            for row2 in df.iter_rows(named=True):
                c = row2["label"]
                l = getattr(lief.PE.Section.CHARACTERISTICS, c)
                if l not in self.which_characteristics:
                    continue
                j = self.which_characteristics.index(l)
                flags[i, j] = 1

        x = CharacteristicGuider._get_bit_mask_(size, offsets, sizes, flags)
        return torch.from_numpy(x)

    def get_structure(self, meta: pl.DataFrame, size: int) -> StructureMap:
        df = meta.filter(
            pl.col("group") == "partitions",
            pl.col("level") == self.level.name,
        )
        lexicon = {}
        index: list[list[tuple[int, int]]] = []
        for i, structure in enumerate(LEVEL_STRUCTURE_MAP[self.level]):
            lexicon[i] = structure
            index.append([])
            for row in df.filter(pl.col("label") == structure.name).iter_rows(named=True):
                s = row["start"]
                e = row["end"]
                index[i].append((s, e))

        return StructureMap(index, lexicon)


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
    def get_padded_inputs(inputs: list[Tensor], min_length: int, pin_memory: bool, change_dtype_after_pad: bool = True) -> Tensor:
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
                # TODO: remove these checks.
                if g_i_j.parse is not None:
                    expected = x_i_j.size(0) if not g_i_j.is_bitpacked else math.ceil(x_i_j.size(0) / 8)
                    if g_i_j.parse.size(0) != expected:
                        raise RuntimeError(f"{i=} {j=} {ranges=} {expected=} x_i_j.shape={tuple(x_i_j.shape)} g_i_j.parse.shape={tuple(g_i_j.parse.shape)}")
                if g_i_j.characteristics is not None:
                    expected = x_i_j.size(0) if not g_i_j.is_bitpacked else math.ceil(x_i_j.size(0) / 8)
                    if g_i_j.characteristics.size(0) != expected:
                        raise RuntimeError(f"{i=} {j=} {ranges=} {expected=} x_i_j.shape={tuple(x_i_j.shape)} g_i_j.characteristics.shape={tuple(g_i_j.characteristics.shape)}")
                if g_i_j.entropy is not None:
                    expected = x_i_j.size(0)
                    if g_i_j.entropy.size(0) != expected:
                        raise RuntimeError(f"{i=} {j=} {ranges=} {expected=} x_i_j.shape={tuple(x_i_j.shape)} g_i_j.entropy.shape={tuple(g_i_j.entropy.shape)}")
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
    """
    A DataLoader wrapper that prefetches batches to a CUDA device using multiple streams.
    """

    def __init__(self, loader: DataLoader[FOrHSamples], device: torch.device, num_streams: int) -> None:
        if num_streams > 0 and (device.type != "cuda" or not torch.cuda.is_available()):
            raise ValueError(f"{self.__class__.__name__} with num_streams > 0 requires a CUDA device.")
        if os.environ.get("TORCH_NCCL_BLOCKING_WAIT") == "1":
            warnings.warn("TORCH_NCCL_BLOCKING_WAIT=1 may cause unexpected crashes with CUDAPrefetcher. Consider unsetting it.")

        self.loader = loader
        self.device = device
        self.num_streams = max(0, int(num_streams))
        self.streams = [torch.cuda.Stream(device=device) for _ in range(self.num_streams)]  # type: ignore[no-untyped-call]
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
    def dataset(self) -> Dataset[Any]:
        return self.loader.dataset


class StreamlessCUDAPrefetcher:
    """
    A simpler prefetcher that does not use multiple streams, useful for debugging.
    """

    def __init__(self, loader: DataLoader[FOrHSamples], device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.it: Optional[Iterator[FOrHSamples]] = None

    def __contains__(self, item: object) -> bool:
        return item in self.loader

    def warmup(self, preload_batches: Optional[int] = None) -> None:
        """
        Spawn DataLoader workers early.
        """
        if preload_batches is not None:
            warnings.warn(f"{self.__class__.__name__}.warmup does not support preloading batches. Ignoring preload_batches={preload_batches}.")
        if self.it is None:
            self.it = iter(self.loader)

    def __iter__(self) -> Iterator[FOrHSamples]:
        self.it = iter(self.loader)
        return self

    def __next__(self) -> FOrHSamples:
        if self.it is None:
            raise StopIteration

        try:
            cpu_batch: FOrHSamples = next(self.it)
        except StopIteration:
            raise StopIteration

        return cpu_batch.to(self.device).decompress()

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loader={self.loader.__class__.__name__}(...), device={self.device})"

    @property
    def dataset(self) -> Dataset[Any]:
        return self.loader.dataset
