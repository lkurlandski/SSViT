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
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import replace
import math
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
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from src.utils import check_tensor
from src.utils import packbits
from src.utils import unpackbits
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


class SemanticGuide(_SemanticGuideOrSemanticGuides):

    @property
    def length_axis(self) -> int:
        return 0


class SemanticGuides(_SemanticGuideOrSemanticGuides):

    @property
    def length_axis(self) -> int:
        return 1

    @classmethod
    def from_singles(cls, guides: Sequence[SemanticGuide], pin_memory: bool = False) -> SemanticGuides:
        if len(guides) == 0:
            raise ValueError("Cannot create Guides from empty list.")

        parse = None
        if guides[0].parse is not None:
            padding_value = False if guides[0].parse.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].parse.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            parse = pad_sequence([g.parse for g in guides], True, padding_value, "right", pin_memory, pad_to_multiple_of)
        entropy = None
        if guides[0].entropy is not None:
            entropy = pad_sequence([g.entropy for g in guides], True, 0.0, "right", pin_memory, PAD_TO_MULTIPLE_OF)
        characteristics = None
        if guides[0].characteristics is not None:
            padding_value = False if guides[0].characteristics.dtype == torch.bool else 0
            pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if guides[0].characteristics.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
            characteristics = pad_sequence([g.characteristics for g in guides], True, padding_value, "right", pin_memory, pad_to_multiple_of)

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
            characteristics = CharacteristicGuider(data, size)()
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
    def from_singles(cls, maps: Sequence[StructureMap], pin_memory: bool = False) -> _StructureMapOrStructureMaps:
        if len(maps) == 0:
            raise ValueError("Cannot create BatchedStructureMap from empty list.")
        lexicon = maps[0].lexicon
        for m in maps[1:]:
            if m.lexicon != lexicon:
                raise ValueError("All StructureMaps must have the same lexicon to be batched.")
        padding_value = False if maps[0].index.dtype == torch.bool else 0.0
        pad_to_multiple_of = PAD_TO_MULTIPLE_OF // 8 if maps[0].index.dtype == torch.uint8 else PAD_TO_MULTIPLE_OF
        index = pad_sequence([m.index for m in maps], True, padding_value, "right", pin_memory, pad_to_multiple_of)
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

    # TODO: this class needs is getting sloppy and needs to be re-written.

    # NOTE: the _StructureMapOrStructureMaps does not get moved to the GPU, since it is only used for indexing.
    # Therefore, it is intentionally ignored in the to(), pin_memory(), compress(), and decompress() methods.

    def __iter__(self) -> Iterable[tuple[StrPath, Name, IntTensor, IntTensor, SemanticGuides, StructureMap]]:
        return iter((self.file, self.name, self.label, self.inputs, self.guides, self.structure))

    @abstractmethod
    def __len__(self) -> int:
        ...

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
            structure=self.structure,
        )

    def pin_memory(self) -> Self:
        return replace(
            self,
            file=self.file,
            name=self.name,
            label=self.label.pin_memory(),
            inputs=self.inputs.pin_memory(),
            guides=self.guides.pin_memory(),
            structure=self.structure,
        )

    def compress(self) -> Self:
        return replace(
            self,
            file=self.file,
            name=self.name,
            label=self.label.to(torch.int16),
            inputs=self.inputs.to(torch.uint16),
            guides=self.guides.compress(),
            structure=self.structure,
        )

    def decompress(self) -> Self:
        return replace(
            self,
            file=self.file,
            name=self.name,
            label=self.label.to(torch.int64),
            inputs=self.inputs.to(torch.int32),
            guides=self.guides.decompress(),
            structure=self.structure,
        )


class Sample(_SampleOrSamples):
    file: StrPath
    name: Name
    label: IntTensor
    inputs: ByteTensor | ShortTensor | IntTensor | LongTensor
    guides: SemanticGuide
    structure: StructureMap

    def __len__(self) -> int:
        return 1

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

    def __len__(self) -> int:
        return len(self.file)

    def __post__init__(self) -> None:
        check_tensor(self.label, (None,), torch.int)
        check_tensor(self.inputs, (self.label.shape[0], None), torch.int)


class BinaryDataset(Dataset):  # type: ignore[misc]

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
        max_length: Optional[int] = None,
    ) -> None:
        self.do_parser = do_parser
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.level = HierarchicalLevel(level)
        self.max_length = max_length
        self.guider = SemanticGuider(do_parser, do_entropy, do_characteristics)
        self.partitioner = StructurePartitioner(HierarchicalLevel(level))

    def __call__(self, file: StrPath, label: int) -> Sample:
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

        return Sample(file, name, label, inputs, guides, structure)


def pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float | int | bool = 0.0,
    padding_side: Literal["right", "left"] = "right",
    pin_memory: bool = False,
    pad_to_multiple_of: int = 1,
) -> Tensor:

    if len(sequences) == 0:
        raise ValueError("Cannot pad an empty list of sequences.")
    if pad_to_multiple_of < 1:
        raise ValueError(f"pad_to_multiple_of must be a positive integer. Got {pad_to_multiple_of}.")

    if not pin_memory and pad_to_multiple_of == 1:
        return _pad_sequence(sequences, batch_first, padding_value, padding_side)

    if padding_side != "right":
        raise NotImplementedError("pad_sequence with pin_memory=True requires padding_side='right'.")
    if not batch_first:
        raise NotImplementedError("pad_sequence with pin_memory=True requires batch_first=True.")

    for s in sequences:
        if pin_memory and s.device.type != "cpu":
            raise ValueError("All sequences must be on CPU when pin_memory=True.")
        if s.shape[1:] != sequences[0].shape[1:]:
            raise ValueError("All sequences must have the same shape except for the first dimension.")
        if s.dtype != sequences[0].dtype:
            raise ValueError("All sequences must have the same dtype.")

    batch_size = len(sequences)
    seq_length = math.ceil(max(s.shape[0] for s in sequences) / pad_to_multiple_of) * pad_to_multiple_of
    other_dims = sequences[0].shape[1:]
    size = (batch_size, seq_length) + tuple(other_dims)

    padded = torch.full(size, fill_value=padding_value, dtype=sequences[0].dtype, pin_memory=pin_memory)
    for i, s in enumerate(sequences):
        s = s.contiguous() if not s.is_contiguous() else s
        padded[i, :s.shape[0]].copy_(s)
    return padded


class CollateFn:

    def __init__(self, pin_memory: bool, bitpack: bool) -> None:
        self.pin_memory = pin_memory
        self.bitpack = bitpack

    def __call__(self, batch: Sequence[Sample]) -> Samples:
        return Samples(
            file=[s.file for s in batch],
            name=[s.name for s in batch],
            label=torch.stack([s.label for s in batch]),
            inputs=pad_sequence([s.inputs.to(torch.int16) + 1 for s in batch], True, 0, "right", self.pin_memory, PAD_TO_MULTIPLE_OF),
            guides=SemanticGuides.from_singles([s.guides.compress() if self.bitpack else s.guides for s in batch], pin_memory=self.pin_memory),
            structure=StructureMaps.from_singles([s.structure for s in batch], pin_memory=self.pin_memory),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pin_memory={self.pin_memory}, bitpack={self.bitpack})"


class CUDAPrefetcher:

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch: Optional[Samples] = None

    def __contains__(self, item: object) -> bool:
        return item in self.loader

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

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loader={self.loader}, device={self.device})"

    def _preload(self) -> None:
        try:
            batch: Samples = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(self.device, non_blocking=True)

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
