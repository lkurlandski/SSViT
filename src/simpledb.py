"""
A simple wrapper over millions of files for faster I/O on slow filesystems.
"""

from __future__ import annotations
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import NamedTuple
from typing import Optional
from typing import Self
import warnings

import numpy as np
import pandas as pd
import polars as pl
import zstandard as zstd


PAGESIZE = 4096


class Sample(NamedTuple):
    """
    Structure representing one sample in the SimpleDB.

    name (str): The hash of the sample.
    data (bytes): The raw bytes of the sample.
    malware (bool): Whether the sample is malware.
    timestamp (int): The UNIX timestamp of the sample, or -1 if not available.
    family (str): The malware family of the sample, or "" if not available/applicable.
    """
    name: str
    data: bytes
    malware: bool
    timestamp: int
    family: str


class SimpleDB:
    """
    A simple database that provides fast(-ish) access to a large number of files.

    Currently, the database can handle a maximum of 10^8 shard files, each of which can contain
    an arbitrary number of individual entries. Balancing the size of the shard vs the number of
    samples within each shard is a tradeoff between system performance and shuffling granularity.

    Structure:
        root
        ├── data
            ├── data-0000000.bin[.zst]
            ├── data-*******.bin[.zst]
            └── data-9999999.bin[.zst]
        ├── size
            ├── size-0000000.csv
            ├── size-*******.csv
            └── size-9999999.csv
        └── meta
            ├── meta-0000000.csv
            ├── meta-*******.csv
            └── meta-9999999.csv

    Each data-*.bin file contains (possibly compressed) concatenated binary blobs of each entry.
    Each size-*.csv file contains the size of each entry, essential for indexing.
    Each meta-*.csv file contains non-essential metadata for each entry.

    data-*******.bin[.zst]
    ---------------------
    This is a simple binary file containing concatenated binary blobs of each entry.
    Each sample is padded to a multiple of `PAGESIZE` bytes for paging efficiency.

    size-*******.csv
    ----------------
    This is a CSV file containing the offset and size of each entry in the corresponding data-*******.bin file.
    It is absolutely critical to the operation of the pseudo database and should not be modified.
    It has the following columns:
        - idx (int): the unique integer ID of the entry.
        - name (str): the name of the entry.
        - shard (int): the shard number (coresponding to the '*******'). This is redundant but useful.
        - offset (int): the offset of the entry in the data-*******.bin file.
        - size (int): the size of the entry in bytes.

    meta-*******.csv
    ----------------
    This is a CSV file containing non-essential metadata for each entry.
    Its less critical to the operation of the pseudo database and can be modified as needed.
    It has the following columns:
        - idx (int): the unique integer ID of the entry.
        - name (str): the name of the entry.
        - shard (int): the shard number (coresponding to the '*'). This is redundant but useful.
        - timestamp (int): the UNIX timestamp of the entry (as a UNIX timestamp). If not available, is -1.
        - malware (int): whether the entry is malware (1) or not (0). If not available, is -1.
        - family (str): the malware family of the entry. If not available, is empty string.
    """

    def __init__(self, dir_root: Path, check: bool = True) -> None:
        self.dir_root = dir_root
        self.dir_data = dir_root / "data"
        self.dir_size = dir_root / "size"
        self.dir_meta = dir_root / "meta"
        if check:
            self._check()

    def _check(self) -> None:
        if not (len(self.files_data) == len(self.files_size) == len(self.files_meta)):
            raise RuntimeError("Number of data, size, and meta files do not match.")
        idx = np.array([], dtype=int)
        names = np.array([], dtype=str)
        for i in range(self.num_shards):
            size_df = self.get_size_df(i)
            meta_df = self.get_meta_df(i)
            if len(size_df) != len(meta_df):
                raise RuntimeError(f"Number of entries in size and meta files do not match for shard {i}.")
            if not (size_df["idx"] == meta_df["idx"]).all():
                raise RuntimeError(f"Indices in size and meta files do not match for shard {i}.")
            if not (size_df["name"] == meta_df["name"]).all():
                raise RuntimeError(f"Names in size and meta files do not match for shard {i}.")
            if not (size_df["shard"] == i).all():
                raise RuntimeError(f"Shard numbers in size files do not match for shard {i}.")
            if not (meta_df["shard"] == i).all():
                raise RuntimeError(f"Shard numbers in meta files do not match for shard {i}.")
            idx = np.concatenate([idx, size_df["idx"].to_numpy(dtype=int)])
            names = np.concatenate([names, size_df["name"].to_numpy(dtype=str)])

        expected = np.arange(idx.max() + 1)
        if not np.array_equal(idx, expected):
            diff = np.setdiff1d(expected, idx).tolist()
            warnings.warn(
                f"Missing indices were detected in the database ({self.dir_root.as_posix()}). "
                f"It is recommended to rebuild the database indices from [0, num_samples]. "
                f"The ({len(diff)}) missing indices are: {diff}"
            )
        if not np.unique(names).shape[0] == names.shape[0]:
            raise RuntimeError("Names across shards are not unique.")

    @cached_property
    def files_data(self) -> list[Path]:
        return sorted(self.dir_data.glob("data-*.bin*"), key=lambda p: p.stem.split("-")[-1])

    @cached_property
    def files_size(self) -> list[Path]:
        return sorted(self.dir_size.glob("size-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @cached_property
    def files_meta(self) -> list[Path]:
        return sorted(self.dir_meta.glob("meta-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @cached_property
    def num_shards(self) -> int:
        return len(self.files_data)

    @cached_property
    def num_samples_per_shard(self) -> list[int]:
        return [len(self.get_size_df(i).index) for i in range(self.num_shards)]

    def num_samples(self, idx: Optional[int | list[int]]) -> list[int]:
        if idx is None:
            idx = list(range(self.num_shards))
        if isinstance(idx, int):
            idx = [idx]
        num_samples = self.num_samples_per_shard
        return [num_samples[i] for i in idx]

    def _get_size_df(self, idx: int) -> pd.DataFrame:
        return pd.read_csv(self.files_size[idx])

    def _get_meta_df(self, idx: int) -> pd.DataFrame:
        return pd.read_csv(self.files_meta[idx], converters={"family": lambda s: s if s else ""})

    def _get_dfs(self, get_df: Callable[[int], pd.DataFrame], idx: Optional[list[int] | int] = None) -> Generator[pd.DataFrame, None, None]:
        if idx is None:
            idx = list(range(self.num_shards))
        elif isinstance(idx, int):
            idx = [idx]
        for i in idx:
            yield get_df(i)

    def get_size_dfs(self, idx: Optional[list[int] | int] = None) -> Generator[pd.DataFrame, None, None]:
        yield from self._get_dfs(self._get_size_df, idx)

    def get_meta_dfs(self, idx: Optional[list[int] | int] = None) -> Generator[pd.DataFrame, None, None]:
        yield from self._get_dfs(self._get_meta_df, idx)

    def get_size_df(self, idx: Optional[list[int] | int] = None) -> pd.DataFrame:
        return pd.concat(list(self.get_size_dfs(idx)), ignore_index=True)

    def get_meta_df(self, idx: Optional[list[int] | int] = None) -> pd.DataFrame:
        return pd.concat(list(self.get_meta_dfs(idx)), ignore_index=True)

    def get_name_to_idx_map(self) -> dict[str, int]:
        d = {}
        for i in range(self.num_shards):
            for _, row in self.get_size_df(i).iterrows():
                if row["name"] in d:
                    raise RuntimeError(f"Duplicate name found: {row['name']}")
                d[row["name"]] = row["idx"]
        return d

    def get_shardix_from_idx(self, idx: int) -> int:
        left, right = 0, self.num_shards - 1
        while left <= right:
            mid = (left + right) // 2
            size_df = self.get_size_df(mid)
            if idx < size_df["idx"].min():
                right = mid - 1
            elif idx > size_df["idx"].max():
                left = mid + 1
            else:
                return mid
        raise ValueError(f"Index {idx} not found in any shard.")


class SimpleDBReader:
    """
    A simple and extremely inefficient reader over a SimpleDB.
    """

    def __init__(self, db: SimpleDB) -> None:
        self.db = db
        self.name_to_idx_map = self.db.get_name_to_idx_map()

    def get(self, idx_or_name: int | str) -> Sample:
        idx = idx_or_name if isinstance(idx_or_name, int) else self.name_to_idx_map[idx_or_name]
        shardidx = self.db.get_shardix_from_idx(idx)

        blob = self.db.files_data[shardidx].read_bytes()
        if blob.startswith(b'\x28\xb5\x2f\xfd'):
            blob = zstd.decompress(blob)

        size_df = self.db.get_size_df(shardidx)
        meta_df = self.db.get_meta_df(shardidx)

        size_row = size_df[size_df["idx"] == idx].iloc[0]
        meta_row = meta_df[meta_df["idx"] == idx].iloc[0]

        name = str(size_row["name"])
        data = blob[size_row["offset"]:size_row["offset"] + size_row["size"]]
        malware = bool(int(meta_row["malware"]) == 1)
        timestamp = int(meta_row["timestamp"])
        family = str(meta_row["family"])

        return Sample(
            name=name,
            data=data,
            malware=malware,
            timestamp=timestamp,
            family=family,
        )


class SimpleDBIterator:
    """
    An iterator over a SimpleDB for optimized data loading.
    """

    def __init__(self, db: SimpleDB) -> None:
        self.db = db
        self.curshardidx = -1

    def __iter__(self) -> Iterator[Sample]:
        yield from self.iter()

    def iter(self, idx: Optional[list[int] | int] = None) -> Iterator[Sample]:
        if idx is None:
            idx = list(range(self.db.num_shards))

        if isinstance(idx, list):
            for i in idx:
                yield from self.iter(i)
            return

        self.curshardidx = idx

        blob = self.db.files_data[idx].read_bytes()
        if blob.startswith(b'\x28\xb5\x2f\xfd'):
            blob = zstd.decompress(blob)

        size_df = self.db.get_size_df(idx)
        meta_df = self.db.get_meta_df(idx)

        for (_, size_row), (_, meta_row) in zip(size_df.iterrows(), meta_df.iterrows(), strict=True):

            name = str(size_row["name"])
            data = blob[size_row["offset"]:size_row["offset"] + size_row["size"]]
            malware = bool(int(meta_row["malware"]) == 1)
            timestamp = int(meta_row["timestamp"])
            family = str(meta_row["family"])

            yield Sample(
                name=name,
                data=data,
                malware=malware,
                timestamp=timestamp,
                family=family,
            )

        self.curshardidx = -1


def roundup(x: int, to: int) -> int:
    """Rounds up x to the nearest multiple of to."""
    return ((x + to - 1) // to) * to


class CreateSimpleDB:
    """
    Create a SimpleDB from files and metadata.
    """

    def __init__(self, root: Path, shardsize: int = 2 ** 30, samples_per_shard: int = -1, exist_ok: bool = False) -> None:
        self.root = root
        self.dir_data = root / "data"
        self.dir_size = root / "size"
        self.dir_meta = root / "meta"
        self.root.mkdir(parents=True, exist_ok=exist_ok)
        self.dir_data.mkdir(exist_ok=exist_ok)
        self.dir_size.mkdir(exist_ok=exist_ok)
        self.dir_meta.mkdir(exist_ok=exist_ok)
        self.shardsize = shardsize
        self.samples_per_shard = samples_per_shard
        if (shardsize > 0) == (samples_per_shard > 0):
            raise ValueError("Exactly one of `shardsize` or `samples_per_shard` must be greater than zero.")

        # ---- internal state for incremental use ----
        self._cur_data: bytearray | None = None
        self._cur_size_df: pd.DataFrame | None = None
        self._cur_meta_df: pd.DataFrame | None = None
        self._cur_count: int = 0
        self._shard_idx: int = 0
        self._sample_idx: int = 0
        self._closed: bool = False

    @property
    def max_capacity(self) -> int:
        if self.samples_per_shard is not None:
            raise NotImplementedError("max_capacity is not defined when samples_per_shard is used.")
        return self.shardsize * int(10 ** len('*******'))

    # ---------- Batch Build API ----------
    def __call__(self, samples: Iterable[Sample]) -> Self:
        for s in samples:
            self.add(s)
        self.close()
        return self

    # ---------- Incremental Build API ----------
    def _ensure_containers(self) -> None:
        if self._cur_data is None:
            self._cur_data = bytearray()
            self._cur_size_df = pd.DataFrame(columns=["idx", "name", "shard", "offset", "size"])
            self._cur_meta_df = pd.DataFrame(columns=["idx", "name", "shard", "timestamp", "malware", "family"])
            self._cur_count = 0

    def add(self, sample: Sample) -> None:
        if self._closed:
            raise RuntimeError("CreateSimpleDB is already closed.")
        self._ensure_containers()

        # If adding this sample would overflow current shard, dump current shard first.
        assert self._cur_data is not None and self._cur_size_df is not None and self._cur_meta_df is not None
        if self._should_dump_shard_containers(self._cur_data, self._cur_size_df, self._cur_meta_df, self._cur_count, sample):
            self._dump_shard_containers(self._cur_data, self._cur_size_df, self._cur_meta_df, self._cur_count, self._shard_idx)
            # reset containers and advance shard id
            self._cur_data = bytearray()
            self._cur_size_df = pd.DataFrame(columns=["idx", "name", "shard", "offset", "size"])
            self._cur_meta_df = pd.DataFrame(columns=["idx", "name", "shard", "timestamp", "malware", "family"])
            self._cur_count = 0
            self._shard_idx += 1

        # Append sample to current shard
        data_bytes = bytearray(sample.data)
        size_df_ = pd.DataFrame({
            "idx": [self._sample_idx],
            "name": [sample.name],
            "shard": [self._shard_idx],
            "offset": [len(self._cur_data)],
            "size": [len(data_bytes)],
        })
        meta_df_ = pd.DataFrame({
            "idx": [self._sample_idx],
            "name": [sample.name],
            "shard": [self._shard_idx],
            "timestamp": [sample.timestamp],
            "malware": [1 if sample.malware else 0],
            "family": [sample.family if sample.family is not None else ""],
        })

        # pad + append
        npad = roundup(len(data_bytes), PAGESIZE) - len(data_bytes)
        self._cur_data.extend(data_bytes)
        if npad:
            self._cur_data.extend(b"\x00" * npad)

        self._cur_size_df = pd.concat([self._cur_size_df, size_df_], ignore_index=True)
        self._cur_meta_df = pd.concat([self._cur_meta_df, meta_df_], ignore_index=True)

        self._cur_count += 1
        self._sample_idx += 1

    def close(self) -> CreateSimpleDB:
        if self._closed:
            return self
        if self._cur_count > 0:
            assert self._cur_data is not None and self._cur_size_df is not None and self._cur_meta_df is not None
            self._dump_shard_containers(self._cur_data, self._cur_size_df, self._cur_meta_df, self._cur_count, self._shard_idx)
        self._closed = True
        return self

    def _should_dump_shard_containers(self, data: bytearray, size_df: pd.DataFrame, meta_df: pd.DataFrame, num_samples: int, sample: Sample) -> bool:
        if self.samples_per_shard > 0:
            return num_samples >= self.samples_per_shard
        if num_samples == 0:
            return False
        if len(sample.data) > self.shardsize:
            warnings.warn(
                f"Sample {sample.name} is larger than the shard size {self.shardsize}. "
                f"It will be stored in its own shard of size {roundup(len(sample.data), PAGESIZE)}."
            )
            return True
        if roundup(len(data) + len(sample.data), PAGESIZE) > self.shardsize:
            return True
        return False

    def _dump_shard_containers(self, data: bytearray, size_df: pd.DataFrame, meta_df: pd.DataFrame, num_samples: int, shard_idx: int, *, compress: bool = False, level: int = 22, threads: int = 0) -> None:
        if num_samples == 0:
            return
        if shard_idx >= 10 ** len('*******'):
            raise RuntimeError(f"Exceeded maximum number of shards ({10 ** len('*******')}).")
        prefix = f"{(len('*******') - len(str(shard_idx))) * '0'}{shard_idx}"
        outfile = self.dir_data / f"data-{prefix}.bin"
        data = bytes(data)
        if compress:
            data = zstd.ZstdCompressor(level=level, threads=threads).compress(data)
            outfile = outfile.with_name(f"{outfile.name}.zst")
        outfile.write_bytes(data)
        size_df.to_csv(self.dir_size / f"size-{prefix}.csv", index=False)
        meta_df.to_csv(self.dir_meta / f"meta-{prefix}.csv", index=False)


class MetadataDB:
    """
    Utility to access metadata stored in parquets in shard-by-shard access patterns.
    """

    def __init__(self, root: Path) -> None:
        """
        Args:
            root: Path to the root directory containing the metadata files.
        """
        self.files = sorted(root.glob("*.parquet"), key=lambda p: p.stem.split("_")[-1])
        self.shard = -1
        self.df = pl.DataFrame()

    def get(self, name: str, shard: int) -> pl.DataFrame:
        """
        Args:
            name: Name of the sample to retrieve metadata for.
            shard: Shard index to look for the sample in.
        """
        if shard != self.shard:
            self.df = pl.read_parquet(self.files[shard])
            self.shard = shard

        return self.df.filter(pl.col("sha") == name)
