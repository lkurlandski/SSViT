"""
Manage data and datasets.
"""

from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from collections.abc import Sequence
import math
import os
import random
from typing import Optional

import torch
from torch import IntTensor
from torch import LongTensor
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from torch.utils.data import Sampler

from src.fileio import read_files_asynch_lazy, read_file


BATCHED_CHUNK_SIZE = 1024


StrPath = str | os.PathLike[str]


class Name(str):

    def __new__(cls, value: StrPath) -> Name:
        value = str(value).split("/")[-1]
        value = value.split(".")[0]
        return super().__new__(cls, value)


Sample = tuple[Name, LongTensor, LongTensor]


class BinaryDataset(ABC):

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None) -> None:
        self.files = files
        self.labels = labels
        self.max_length = max_length

    @abstractmethod
    def __getitem__(self, i: int) -> Sample:
        ...

    def __len__(self) -> int:
        return len(self.files)


class MapBinaryDataset(Dataset, BinaryDataset):
    """
    Standard dataset. Loads binaries into memory when initialized.
    """

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None) -> None:
        super().__init__(files, labels, max_length)
        self.x: list[bytes] = list(read_files_asynch_lazy(self.files, max_length))

    def __getitem__(self, i: int) -> Sample:
        n = Name(self.files[i])
        x = torch.frombuffer(self.x[i], dtype=torch.uint8).to(torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return n, x, y


class MapBinaryDatasetBatchedLoader(Dataset, BinaryDataset):
    """
    Loads large chunks of binaries once in-memory cache of data is drained.
    Needs to paired with a sampler that will select indices in contiguous chunks.
    Otherwise, this is simply a much less efficient version of the MapBinaryDataset.
    """

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None, chunk_size: int = BATCHED_CHUNK_SIZE) -> None:
        super().__init__(files, labels, max_length)
        self.chunk_size = chunk_size
        self.x: list[Optional[bytes]] = [None for _ in range(len(self))]

    def __getitem__(self, i: int) -> Sample:
        if self.x[i] is None:
            self.x = [None for _ in range(len(self))]
            files = self.files[i:i+self.chunk_size]
            data = list(read_files_asynch_lazy(files, self.max_length))
            self.x[i : i + self.chunk_size] = data

        n = Name(self.files[i])
        x = torch.frombuffer(self.x[i], dtype=torch.uint8).to(torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return n, x, y


class MapBinaryDatasetMemoryMapped(Dataset, BinaryDataset):
    """
    Uses torch's memory-mapped tensors to avoid loading all data into memory at once.
    """

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None) -> None:
        super().__init__(files, labels, max_length)

        def get_tensor(f: StrPath) -> IntTensor:
            size = min(os.path.getsize(f), max_length) if max_length is not None else os.path.getsize(f)
            return torch.from_file(str(f), shared=False, size=size, dtype=torch.uint8)

        self.x: list[IntTensor] = [get_tensor(f) for f in self.files]

    def __getitem__(self, i: int) -> Sample:
        n = Name(self.files[i])
        x = self.x[i].to(torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return n, x, y


class IterableBinaryDataset(IterableDataset, BinaryDataset):

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None) -> None:
        super().__init__(files, labels, max_length)
        # Variables unique to each process when num_workers > 1 (after process forking).
        self.my_length: int = -1
        self.my_files: Sequence[StrPath] = []
        self.my_labels: Sequence[int] = []
        self.my_idx: int = -1

    def __iter__(self) -> Iterator[Sample]:
        self.set_my_attributes()
        return self

    def __next__(self) -> Sample:
        if self.my_idx >= len(self.my_files):
            raise StopIteration()

        n = Name(self.files[self.my_idx])
        b = read_file(self.files[self.my_idx], self.max_length)
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.long)
        y = torch.tensor(self.labels[self.my_idx], dtype=torch.long)

        self.my_idx += 1
        return n, x, y

    def set_my_attributes(self) -> None:
        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self)
        else:
            per_worker = int(math.ceil((len(self) - 0) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = 0 + worker_id * per_worker
            end = min(start + per_worker, len(self))

        self.my_length = end - start
        self.my_files = self.files[start:end]
        self.my_labels = self.labels[start:end]
        self.my_x = [None for _ in range(self.my_length)]
        self.my_idx = 0


class IterableBinaryDatasetBatchedLoader(IterableDataset, BinaryDataset):

    def __init__(self, files: Sequence[StrPath], labels: Sequence[int], max_length: Optional[int] = None, chunk_size: int = BATCHED_CHUNK_SIZE) -> None:
        super().__init__(files, labels, max_length)
        self.chunk_size = chunk_size
        # Variables unique to each process when num_workers > 1 (after process forking).
        self.my_length: int = -1
        self.my_files: Sequence[StrPath] = []
        self.my_x: list[Optional[bytes]] = []
        self.my_labels: Sequence[int] = []
        self.my_idx: int = -1

    def __iter__(self) -> Iterator[Sample]:
        self.set_my_attributes()
        return self

    def __next__(self) -> Sample:
        if self.my_idx >= len(self.my_files):
            raise StopIteration()

        if self.my_x[self.my_idx] is None:
            self.my_x = [None for _ in range(self.my_length)]
            files = self.my_files[self.my_idx : self.my_idx + self.chunk_size]
            data = list(read_files_asynch_lazy(files, self.max_length))
            self.my_x[self.my_idx : self.my_idx + self.chunk_size] = data

        n = Name(self.my_files[self.my_idx])
        x = torch.frombuffer(self.my_x[self.my_idx], dtype=torch.uint8).to(torch.long)
        y = torch.tensor(self.my_labels[self.my_idx], dtype=torch.long)

        self.my_idx += 1
        return n, x, y

    def set_my_attributes(self) -> None:
        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self)
        else:
            per_worker = int(math.ceil((len(self) - 0) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = 0 + worker_id * per_worker
            end = min(start + per_worker, len(self))

        self.my_length = end - start
        self.my_files = self.files[start:end]
        self.my_labels = self.labels[start:end]
        self.my_x = [None for _ in range(self.my_length)]
        self.my_idx = 0


class ContiguousSampler(Sampler[int]):

    def __init__(self, num_samples: int, chunk_size: int, shuffle: bool = False) -> None:
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.leads = list(range(0, num_samples, self.chunk_size))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            random.shuffle(self.leads)
        return self

    def __next__(self) -> int:
        for l in self.leads:
            for i in range(l, min(l + self.chunk_size, self.num_samples)):
                return i
        raise StopIteration()


class CollateFn:

    def __call__(self, batch: list[Sample]) -> tuple[LongTensor, LongTensor]:
        names, xs, ys = zip(*batch)
        x = torch.nn.utils.rnn.pad_sequence([x + 1 for x in xs], batch_first=True, padding_value=0)
        y = torch.stack(ys, dim=0)
        return x, y
