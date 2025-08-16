"""
Manage data and datasets.
"""

from __future__ import annotations
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
from typing import Optional

import torch
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
from torch.nn.utils.rnn import pad_sequence

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
class SemanticGuides:
    """
    Semantic guides to acompany a byte stream.

    Attrs:
        parse (Optional[BoolTensor]): A boolean tensor of shape (T, *).
        entropy (Optional[DoubleTensor]): A float tensor of shape (T,).
        characteristics (Optional[BoolTensor]): A boolean tensor of shape (T, *).
    """
    parse: Optional[BoolTensor] = None
    entropy: Optional[HalfTensor | FloatTensor | DoubleTensor] = None
    characteristics: Optional[BoolTensor] = None

    def __post_init__(self) -> None:
        lengths = [x.shape[0] for x in (self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"All non-None guides must have the same length. Got lengths: {lengths}")

    def clone(self) -> SemanticGuides:
        return SemanticGuides(
            self.parse.clone() if self.parse is not None else None,
            self.entropy.clone() if self.entropy is not None else None,
            self.characteristics.clone() if self.characteristics is not None else None,
        )

    def to(self, device: torch.device) -> SemanticGuides:
        return SemanticGuides(
            self.parse.to(device) if self.parse is not None else None,
            self.entropy.to(device) if self.entropy is not None else None,
            self.characteristics.to(device) if self.characteristics is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BatchedSemanticGuides(SemanticGuides):

    def __post_init__(self) -> None:
        lengths = [x.shape[1] for x in (self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"All non-None guides must have the same length. Got lengths: {lengths}")

    @classmethod
    def from_singles(cls, guides: Sequence[SemanticGuides]) -> BatchedSemanticGuides:
        if len(guides) == 0:
            raise ValueError("Cannot create BatchedSemanticMap from empty list.")

        parse = None
        if guides[0].parse is not None:
            parse = pad_sequence([g.parse for g in guides], batch_first=True, padding_value=False)
        entropy = None
        if guides[0].entropy is not None:
            entropy = pad_sequence([g.entropy for g in guides], batch_first=True, padding_value=0.0)
        characteristics = None
        if guides[0].characteristics is not None:
            characteristics = pad_sequence([g.characteristics for g in guides], batch_first=True, padding_value=False)
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

    def __call__(self, data: LiefParse) -> SemanticGuides:
        parse = None
        if self.do_parse:
            parse = ParserGuider(data)(simple=self.simple)
            parse = torch.from_numpy(parse)

        entropy = None
        if self.do_entropy:
            entropy = EntropyGuider(data)(radius=self.radius)
            entropy = torch.from_numpy(entropy).to(torch.float16)

        characteristics = None
        if self.do_characteristics:
            characteristics = CharacteristicGuider(data)()
            characteristics = torch.from_numpy(characteristics)

        return SemanticGuides(parse, entropy, characteristics)


@dataclass(frozen=True, slots=True)
class StructureMap:
    """
    Maps hierarchical structures to byte positions in a binary.

    Attrs:
        index (BoolTensor): A binary matrix of shape (T,*)
            indicating which bytes belong to which structures.
        lexicon (Mapping[int, HierarchicalStructure]): A mapping indicating which column
            in the index corresponds to which structure.
    """

    index: BoolTensor
    lexicon: Mapping[int, HierarchicalStructure]

    def __post_init__(self) -> None:
        if self.index.dim() != 2:
            raise ValueError("StructureMap index must be a 2D tensor.")
        if self.index.shape[1] != len(list(self.lexicon.keys())):
            raise ValueError("StructureMap index does not match lexicon keys.")

    def clone(self) -> StructureMap:
        cls = type(self)
        return cls(self.index.clone(), self.lexicon)

    def to(self, device: torch.device) -> StructureMap:
        cls = type(self)
        return cls(self.index.to(device), self.lexicon)


@dataclass(frozen=True, slots=True)
class BatchedStructureMap(StructureMap):

    def __post_init__(self) -> None:
        if self.index.dim() != 3:
            raise ValueError("BatchedStructureMap index must be a 3D tensor.")
        if self.index.shape[2] != len(list(self.lexicon.keys())):
            raise ValueError("BatchedStructureMap index does not match lexicon keys.")

    @classmethod
    def from_singles(cls, maps: Sequence[StructureMap]) -> BatchedStructureMap:
        if len(maps) == 0:
            raise ValueError("Cannot create BatchedStructureMap from empty list.")
        lexicon = maps[0].lexicon
        for m in maps[1:]:
            if m.lexicon != lexicon:
                raise ValueError("All StructureMaps must have the same lexicon to be batched.")
        index = pad_sequence([m.index for m in maps], batch_first=True, padding_value=False)
        return cls(index, lexicon)


class StructurePartitioner:
    """
    Partitions a binary into hierarchical structures.
    """

    def __init__(self, level: HierarchicalLevel = HierarchicalLevel.NONE) -> None:
        self.level = level

    def __call__(self, data: LiefParse) -> StructureMap:
        parser = StructureParser(data)

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
class Sample:
    file: StrPath
    name: Name
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuides
    structure: StructureMap

    def __iter__(self) -> Iterable[tuple[StrPath, Name, IntTensor, IntTensor, SemanticGuides, StructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))


@dataclass(frozen=True, slots=True)
class BatchedSamples:
    file: list[StrPath]
    name: list[Name]
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: BatchedSemanticGuides
    structure: BatchedStructureMap

    def __iter__(self) -> Iterable[tuple[list[StrPath], list[Name], IntTensor, IntTensor, BatchedSemanticGuides, BatchedStructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))


class BinaryDataset(Dataset):

    def __init__(
        self,
        files: Sequence[StrPath],
        labels: Sequence[int],
        preprocessor: Preprocessor,
    ) -> None:
        self.files = files
        self.labels = labels
        self.preprocessor = preprocessor

    def __getitem__(self, i: int) -> Sample:
        file = self.files[i]
        name = Name(file)
        label = torch.tensor(self.labels[i])
        # inputs, guides, structure = self.preprocessor(Path(file).read_bytes(), file)

        with open(file, "rb") as fp:
            with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mv = memoryview(mm)
                try:
                    inputs, guides, structure = self.preprocessor(mv, file)
                    # Detach references to the memoryview object. This is necessary to prevent Segmentation Faults.
                    inputs = inputs.clone()
                    # guides = SemanticGuides(
                    #     guides.parse.clone() if guides.parse is not None else None,
                    #     guides.entropy.clone() if guides.entropy is not None else None,
                    #     guides.characteristics.clone() if guides.characteristics is not None else None,
                    # )
                    # structure = StructureMap(
                    #     structure.index.clone(),
                    #     deepcopy(structure.lexicon),
                    # )
                    # TODO: Can we move straight to GPU memory?
                finally:
                    mv.release()

        return Sample(file, name, label, inputs, guides, structure)

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
        self.guider = SemanticGuider(do_parser, do_entropy, do_characteristics)
        self.partitioner = StructurePartitioner(HierarchicalLevel(level))

    def __call__(self, mv: memoryview, file: StrPath) -> tuple[IntTensor, SemanticGuides, StructureMap]:
        inputs = torch.frombuffer(mv, dtype=torch.uint8)
        guides = self.guider(file)
        structure = self.partitioner(file)
        return inputs, guides, structure


class CollateFn:

    def __init__(
        self,
        do_parser: bool = True,
        do_entropy: bool = True,
        do_characteristics: bool = True,
    ) -> None:
        self.do_parser = do_parser
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics

    def __call__(self, batch: Sequence[Sample]) -> BatchedSamples:
        return BatchedSamples(
            file=[s.file for s in batch],
            name=[s.name for s in batch],
            label=torch.stack([s.label for s in batch]),
            inputs=pad_sequence([s.inputs.to(torch.int16) + 1 for s in batch], batch_first=True, padding_value=0),
            guides=BatchedSemanticGuides.from_singles([s.guides for s in batch]),
            structure=BatchedStructureMap.from_singles([s.structure for s in batch]),
        )
