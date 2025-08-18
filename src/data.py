"""
Manage data and datasets.
"""

from __future__ import annotations
from abc import ABC
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
import mmap
import os
from pathlib import Path
import sys
from typing import Literal
from typing import Optional
from typing import Self

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
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from src.utils import check_tensor
from src.binanal import _parse_pe_and_get_size
from src.binanal import _get_size_of_liefparse
from src.binanal import LiefParse
from src.binanal import ParserGuider
from src.binanal import CharacteristicGuider
from src.binanal import EntropyGuider
from src.binanal import StructureParser
from src.binanal import HierarchicalLevel
from src.binanal import HierarchicalStructure
from src.binanal import HierarchicalStructureCoarse
from src.binanal import HierarchicalStructureFine
from src.binanal import HierarchicalStructureMiddle
from src.binanal import HierarchicalStructureNone


StrPath = str | os.PathLike[str]


class Name(str):

    def __new__(cls, value: StrPath) -> Name:
        value = str(value).split("/")[-1]
        value = value.split(".")[0]
        if len(value) != 64 or not all(c in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Invalid name: {value}")
        return super().__new__(cls, value)


@dataclass(frozen=True, slots=True)
class _SemanticGuideOrSemanticGuides(ABC):
    parse: Optional[BoolTensor] = None
    entropy: Optional[HalfTensor | FloatTensor | DoubleTensor] = None
    characteristics: Optional[BoolTensor] = None

    def clone(self) -> Self:
        return replace(
            self,
            parse=self.parse.clone() if self.parse is not None else None,
            entropy=self.entropy.clone() if self.entropy is not None else None,
            characteristics=self.characteristics.clone() if self.characteristics is not None else None,
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        return replace(
            self,
            parse=self.parse.to(device, non_blocking=non_blocking) if self.parse is not None else None,
            entropy=self.entropy.to(device, non_blocking=non_blocking) if self.entropy is not None else None,
            characteristics=self.characteristics.to(device, non_blocking=non_blocking) if self.characteristics is not None else None,
        )

    def pin_memory(self) -> Self:
        return replace(
            self,
            parse=self.parse.pin_memory() if self.parse is not None else None,
            entropy=self.entropy.pin_memory() if self.entropy is not None else None,
            characteristics=self.characteristics.pin_memory() if self.characteristics is not None else None,
        )


class SemanticGuide(_SemanticGuideOrSemanticGuides):

    def __post_init__(self) -> None:
        lengths = [x.shape[0] for x in (self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"All non-None guides must have the same length. Got lengths: {lengths}")


class SemanticGuides(_SemanticGuideOrSemanticGuides):

    def __post_init__(self) -> None:
        lengths = [x.shape[1] for x in (self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"All non-None guides must have the same length. Got lengths: {lengths}")

    @classmethod
    def from_singles(cls, guides: Sequence[SemanticGuide], pin_memory: bool = False) -> SemanticGuides:
        if len(guides) == 0:
            raise ValueError("Cannot create Guides from empty list.")

        parse = None
        if guides[0].parse is not None:
            parse = pad_sequence([g.parse for g in guides], True, False, "right", pin_memory)
        entropy = None
        if guides[0].entropy is not None:
            entropy = pad_sequence([g.entropy for g in guides], True, 0.0, "right", pin_memory)
        characteristics = None
        if guides[0].characteristics is not None:
            characteristics = pad_sequence([g.characteristics for g in guides], True, False, "right", pin_memory)
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
            return SemanticGuide()

        if data is None:
            raise ValueError("Data must be provided for semantic guides.")

        parse = None
        if self.do_parse:
            parse = ParserGuider(data, size)(simple=self.simple)
            parse = torch.from_numpy(parse)

        entropy = None
        if self.do_entropy:
            # TODO: clean this up.
            inputs = inputs.clone().numpy() if inputs is not None else data
            entropy = EntropyGuider(inputs)(radius=self.radius)      # 64-bit is fastest for computation.
            entropy = torch.from_numpy(entropy).to(torch.float16)  # 16-bit is fastest for GPU transfer.

        characteristics = None
        if self.do_characteristics:
            characteristics = CharacteristicGuider(data, size)()
            characteristics = torch.from_numpy(characteristics)

        return SemanticGuide(parse, entropy, characteristics)


@dataclass(frozen=True, slots=True)
class _StructureMapOrStructureMaps(ABC):
    index: BoolTensor
    lexicon: Mapping[int, HierarchicalStructure]

    def clone(self) -> Self:
        return replace(
            self,
            index=self.index.clone(),
            lexicon=deepcopy(self.lexicon),
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
       return replace(
            self,
            index=self.index.to(device, non_blocking=non_blocking),
            lexicon=deepcopy(self.lexicon),
        )

    def pin_memory(self) -> Self:
        return replace(
            self,
            index=self.index.pin_memory(),
            lexicon=deepcopy(self.lexicon),
        )


class StructureMap(_StructureMapOrStructureMaps):

    def __post_init__(self) -> None:
        if self.index.dim() != 2:
            raise ValueError("StructureMap index must be a 2D tensor.")
        if self.index.shape[1] != len(list(self.lexicon.keys())):
            raise ValueError("StructureMap index does not match lexicon keys.")


class StructureMaps(_StructureMapOrStructureMaps):

    def __post_init__(self) -> None:
        if self.index.dim() != 3:
            raise ValueError("BatchedStructureMap index must be a 3D tensor.")
        if self.index.shape[2] != len(list(self.lexicon.keys())):
            raise ValueError("BatchedStructureMap index does not match lexicon keys.")

    @classmethod
    def from_singles(cls, maps: Sequence[StructureMap], pin_memory: bool = False) -> _StructureMapOrStructureMaps:
        if len(maps) == 0:
            raise ValueError("Cannot create BatchedStructureMap from empty list.")
        lexicon = maps[0].lexicon
        for m in maps[1:]:
            if m.lexicon != lexicon:
                raise ValueError("All StructureMaps must have the same lexicon to be batched.")
        index = pad_sequence([m.index for m in maps], True, False, "right", pin_memory)
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


@dataclass(frozen=True, slots=True)
class _SampleOrSamples(ABC):
    file: StrPath | list[StrPath]
    name: Name | list[Name]
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: _SemanticGuideOrSemanticGuides
    structure: _StructureMapOrStructureMaps

    def __iter__(self) -> Iterable[tuple[StrPath, Name, IntTensor, IntTensor, SemanticGuides, StructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))

    def clone(self) -> Self:
        return replace(
            self,
            file=deepcopy(self.file),
            name=deepcopy(self.name),
            label=self.label.clone(),
            inputs=self.inputs.clone(),
            guides=self.guides.clone(),
            structure=self.structure.clone(),
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        return replace(
            self,
            file=self.file,
            name=self.name,
            label=self.label.to(device, non_blocking=non_blocking),
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            guides=self.guides.to(device, non_blocking=non_blocking),
            structure=self.structure.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Self:
        return replace(
            self,
            file=self.file,
            name=self.name,
            label=self.label.pin_memory(),
            inputs=self.inputs.pin_memory(),
            guides=self.guides.pin_memory(),
            structure=self.structure.pin_memory(),
        )


class Sample(_SampleOrSamples):
    file: StrPath
    name: Name
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuide
    structure: StructureMap

    def __post__init__(self) -> None:
        check_tensor(self.label, (), torch.int)
        check_tensor(self.inputs, (None,), torch.int)


class Samples(_SampleOrSamples):
    file: list[StrPath]
    name: list[Name]
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuides
    structure: StructureMaps

    def __post__init__(self) -> None:
        check_tensor(self.label, (None,), torch.int)
        check_tensor(self.inputs, (self.label.shape[0], None), torch.int)


class BinaryDataset(Dataset):

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], preprocessor: Preprocessor) -> None:
        self.files = files
        self.labels = labels
        self.preprocessor = preprocessor

    def __getitem__(self, i: int) -> Sample:
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
    ) -> None:
        self.do_parser = do_parser
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.level = HierarchicalLevel(level)
        self.guider = SemanticGuider(do_parser, do_entropy, do_characteristics)
        self.partitioner = StructurePartitioner(HierarchicalLevel(level))

    def __call__(self, file: StrPath, label: int) -> Sample:
        name = Name(file)
        label = torch.tensor(label)

        # TODO: experiment with using torch.from_file instead of mmap.
        # inputs = torch.from_file(str(file), shared=False, size=os.path.getsize(file), dtype=torch.uint8)

        with open(file, "rb") as fp:
            with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mv = memoryview(mm)
                try:
                    inputs = torch.frombuffer(mv, dtype=torch.uint8)
                    inputs = inputs.clone()  # Detach reference to the memoryview.
                finally:
                    mv.release()

        if self.do_parser or self.do_entropy or self.do_characteristics or self.level != HierarchicalLevel.NONE:
            pe, size = _parse_pe_and_get_size(file)
        else:
            pe, size = None, os.path.getsize(file)

        guides = self.guider(pe, size, inputs)
        structure = self.partitioner(pe, size)

        return Sample(file, name, label, inputs, guides, structure)


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float | int | bool = 0.0,
    padding_side: Literal["right", "left"] = "right",
    pin_memory: bool = False,
) -> Tensor:
    if not pin_memory:
        return _pad_sequence(sequences, batch_first, padding_value, padding_side)

    if padding_side != "right":
        raise NotImplementedError("pad_sequence with pin_memory=True requires padding_side='right'.")
    if not batch_first:
        raise NotImplementedError("pad_sequence with pin_memory=True requires batch_first=True.")

    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty list of sequences.")
    for s in sequences:
        if s.device.type != "cpu":
            raise ValueError("All sequences must be on CPU when pin_memory=True.")
        if s.shape[1:] != sequences[0].shape[1:]:
            raise ValueError("All sequences must have the same shape except for the first dimension.")
        if s.dtype != sequences[0].dtype:
            raise ValueError("All sequences must have the same dtype.")

    size = tuple([len(sequences), max(s.shape[0] for s in sequences)] + list(sequences[0].shape[1:]))
    dtype = sequences[0].dtype

    padded = torch.full(size, fill_value=padding_value, dtype=dtype, pin_memory=True)
    for i, s in enumerate(sequences):
        s = s.contiguous() if not s.is_contiguous() else s
        padded[i, :s.shape[0]].copy_(s)
    return padded


class CollateFn:

    def __init__(self, pin_memory: bool) -> None:
        self.pin_memory = pin_memory

    def __call__(self, batch: Sequence[Sample]) -> Samples:
        return Samples(
            file=[s.file for s in batch],
            name=[s.name for s in batch],
            label=torch.stack([s.label for s in batch]),
            inputs=pad_sequence([s.inputs.to(torch.int16) + 1 for s in batch], True, 0, "right", self.pin_memory),
            guides=SemanticGuides.from_singles([s.guides for s in batch], pin_memory=self.pin_memory),
            structure=StructureMaps.from_singles([s.structure for s in batch], pin_memory=self.pin_memory),
        )


class CUDAPrefetcher:

    def __init__(self, loader: Iterable[Samples], device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch: Optional[Samples] = None

    def __iter__(self) -> Iterator[Samples]:
        self.it = iter(self.loader)
        self.next_batch = None
        self._preload()
        return self

    def __next__(self) -> Samples:
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)
