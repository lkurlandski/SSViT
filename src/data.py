"""
Manage data and datasets.
"""

from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
from typing import Optional

import torch
from torch import IntTensor
from torch.nn.utils.rnn import pad_sequence

from src.binanal import HierarchicalLevel
from src.binanal import SemanticGuider
from src.binanal import SemanticGuides
from src.binanal import BatchedSemanticGuides
from src.binanal import StructurePartitioner
from src.binanal import StructureMap


StrPath = str | os.PathLike[str]


class Name(str):

    def __new__(cls, value: StrPath) -> Name:
        value = str(value).split("/")[-1]
        value = value.split(".")[0]
        if len(value) != 64 or not all(c in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Invalid name: {value}")
        return super().__new__(cls, value)


@dataclass(frozen=True, slots=True)
class Sample:
    file: StrPath
    name: Name
    label: IntTensor
    inputs: IntTensor
    guides: SemanticGuides
    structure: StructureMap


@dataclass(frozen=True, slots=True)
class BatchedSamples:
    file: list[StrPath]
    name: list[Name]
    label: IntTensor
    inputs: IntTensor
    guides: BatchedSemanticGuides
    structure: list[StructureMap]


class BinaryDataset:

    def __init__(
        self,
        files: Sequence[StrPath],
        labels: Sequence[int],
        preprocessor: Preprocessor,
        max_length: Optional[int] = None,
    ) -> None:
        self.files = files
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length

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

    def __call__(self, mv: memoryview | bytes, file: StrPath) -> tuple[IntTensor, SemanticGuides, StructureMap]:
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
            inputs=pad_sequence([s.inputs for s in batch], batch_first=True, padding_value=0),
            guides=BatchedSemanticGuides(
                parse=pad_sequence([s.guides.parse for s in batch], batch_first=True, padding_value=False) if self.do_parser else None,
                entropy=pad_sequence([s.guides.entropy for s in batch], batch_first=True, padding_value=0.0) if self.do_entropy else None,
                characteristics=pad_sequence([s.guides.characteristics for s in batch], batch_first=True, padding_value=False) if self.do_characteristics else None,
            ),
            structure=[s.structure for s in batch],
        )
