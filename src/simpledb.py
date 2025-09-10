"""
A simple wrapper over millions of file for faster I/O on slow filesystems.
"""

from collections.abc import Iterable
from collections.abc import Iterator
import contextlib
import gc
from pathlib import Path
import os
from typing import Any
from typing import Literal
from typing import Optional
from typing import NamedTuple
from typing import Self
import warnings

import numpy as np
import pandas as pd
import torch
from torch import ByteTensor
from torch import UntypedStorage


PADDING = 4096


class SimpleDBSample(NamedTuple):
    """A subset of information returned by the SimpleDB when indexed."""
    name: str
    data: ByteTensor
    malware: bool
    timestamp: int
    family: Optional[str] = None


class SimpleDB:
    """
    A simple database that provides fast, random access to packed files.

    Structure:
        root
        ├── data
            ├── data-0000000.bin
            ├── data-*******.bin
            └── data-9999999.bin
        ├── size
            ├── size-0000000.csv
            ├── size-*******.csv
            └── size-9999999.csv
        └── meta
            ├── meta-0000000.csv
            ├── meta-*******.csv
            └── meta-9999999.csv

    Each data-*.bin file contains concatenated binary blobs of each entry.
    Each size-*.csv file contains the size of each entry, essential for indexing.
    Each meta-*.csv file contains non-essential metadata for each entry.

    data-*.bin
    -------------
    This is a simple binary file containing concatenated binary blobs of each entry.
    Each sample is padded to a multiple of `PADDING` bytes for paging efficiency.

    size-*.csv
    -------------
    This is a CSV file containing the offset and size of each entry in the corresponding data-****.bin file.
    It is absolutely critical to the operation of the pseudo database and should not be modified.
    It has the following columns:
        - idx (int): the unique integer ID of the entry.
        - name (str): the name of the entry.
        - shard (int): the shard number (coresponding to the '*'). This is redundant but useful.
        - offset (int): the offset of the entry in the data-****.bin file.
        - size (int): the size of the entry in bytes.

    meta-*.csv
    -------------
    This is a CSV file containing non-essential metadata for each entry.
    Its less critical to the operation of the pseudo database and can be modified as needed.
    It has the following columns:
        - idx (int): the unique integer ID of the entry.
        - name (str): the name of the entry.
        - shard (int): the shard number (coresponding to the '*'). This is redundant but useful.
        - timestamp (int): the UNIX timestamp of the entry (as a UNIX timestamp). If not available, is -1.
        - malware (int): whether the entry is malware (1) or not (0). If not available, is -1.
        - family (str): the malware family of the entry. If not available, is empty string.

    The size-*.csv and meta-*.csv files are separated because size is critical to the operation
    of the pseudo database, while meta is not.
    """

    def __init__(self, dir_root: Path, reader: Literal["pread", "torch"] = "torch", allow_name_indexing: bool = False) -> None:
        self.dir_root = dir_root
        self.reader = reader
        self.allow_name_indexing = allow_name_indexing
        self.dir_data = dir_root / "data"
        self.dir_size = dir_root / "size"
        self.dir_meta = dir_root / "meta"
        self.storages: list[UntypedStorage] = []
        self.handles: list[int] = []
        self.size_df: pd.DataFrame = self.get_size_df()
        self.meta_df: pd.DataFrame = self.get_meta_df()
        self.name_map: dict[str, int] = self.get_name_map()
        if not (len(self.files_data) == len(self.files_size) == len(self.files_meta)):
            raise RuntimeError("Number of data, size, and meta files do not match.")
        if self.reader not in ("pread", "torch"):
            raise ValueError("Reader must be 'pread' or 'torch'.")
        if len(self.size_df) != len(self.meta_df):
            raise RuntimeError(f"Size and meta data have different number of rows ({len(self.size_df)} vs {len(self.meta_df)}).")

    @property
    def files_data(self) -> list[Path]:
        return sorted(self.dir_data.glob("data-*.bin"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_size(self) -> list[Path]:
        return sorted(self.dir_size.glob("size-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_meta(self) -> list[Path]:
        return sorted(self.dir_meta.glob("meta-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @property
    def is_open(self) -> bool:
        return len(self.storages) > 0 or len(self.handles) > 0

    def _get_idx_from_idx_or_name(self, idx_or_name: int | str) -> int:
        if isinstance(idx_or_name, int):
            return idx_or_name

        if isinstance(idx_or_name, str):
            if not self.allow_name_indexing:
                raise ValueError("Name indexing is not allowed. Set allow_name_indexing=True to enable it.")
            return self.name_map[idx_or_name]

        raise TypeError("Index must be an integer or string.")

    def _read_as_tensor_torch(self, shard: int, offset: int, length: int) -> ByteTensor:
        storage = self.storages[shard]
        data = torch.empty(0, dtype=torch.uint8)
        data.set_(storage, storage_offset=offset, size=(length,), stride=(1,))
        return data

    def _read_as_tensor_pread(self, shard: int, offset: int, length: int) -> ByteTensor:
        fd = self.handles[shard]
        data = torch.empty(length, dtype=torch.uint8)
        mv = memoryview(data.numpy()).cast("B")
        got, off = 0, offset
        while got < length:
            n = os.preadv(fd, [mv[got:]], off)
            if n == 0:
                raise EOFError(f"EOF at shard={shard} off={off}, need {length-got} more")
            got += n
            off += n
        return data

    def __getitem__(self, idx_or_name: int | str) -> SimpleDBSample:
        idx = self._get_idx_from_idx_or_name(idx_or_name)

        size_dict: dict[str, Any] = self.size_df.loc[idx].to_dict()
        meta_dict: dict[str, Any] = self.meta_df.loc[idx].to_dict()

        if any(size_dict[key] != meta_dict[key] for key in ("idx", "name", "shard")):
            raise RuntimeError(f"Size and meta data mismatch for idx {idx}.")

        name = size_dict["name"]
        shard = size_dict["shard"]
        offset = size_dict["offset"]
        length = size_dict["size"]
        malware = meta_dict["malware"] == 1
        timestamp = meta_dict["timestamp"]
        family = meta_dict["family"]

        if self.reader == "torch":
            data = self._read_as_tensor_torch(shard, offset, length)
        elif self.reader == "pread":
            data = self._read_as_tensor_pread(shard, offset, length)

        return SimpleDBSample(
            name=name,
            data=data,
            malware=malware,
            timestamp=timestamp,
            family=family,
        )

    def get_size_df(self) -> pd.DataFrame:
        size_dfs = [pd.read_csv(f) for f in self.files_size]
        for f, df in zip(self.files_data, size_dfs):
            if (df["offset"] + df["size"]).max() > os.path.getsize(f):
                raise RuntimeError("Size file contains an entry that exceeds the size its data file.")
        size_df = pd.concat(size_dfs, ignore_index=True)
        size_df = size_df.sort_values(by="idx").set_index("idx", drop=False)
        return size_df

    def get_meta_df(self) -> pd.DataFrame:
        meta_dfs = [pd.read_csv(f, converters={"family": lambda s: s if s else ""}) for f in self.files_meta]
        meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_df = meta_df.sort_values(by="idx").set_index("idx", drop=False)
        return meta_df

    def get_name_map(self) -> dict[str, int]:
        if not self.allow_name_indexing:
            return {}
        if not self.meta_df["name"].is_unique:
            raise RuntimeError("Names are not unique, so name indexing is not possible.")
        name_map = {str(row["name"]): int(row["idx"]) for _, row in self.meta_df.iterrows()}
        return name_map

    def open(self) -> Self:
        """
        Prepares the pseudo database for operations and conducts basic integrity checks.
        """
        if self.is_open:
            self.close()

        # NOTE: it is critical that storages and handles refer to the same list.
        if self.reader == "torch":
            for f in self.files_data:
                self.storages.append(UntypedStorage.from_file(str(f), shared=False, nbytes=os.path.getsize(f)))
        elif self.reader == "pread":
            for f in self.files_data:
                self.handles.append(os.open(str(f), os.O_RDONLY | os.O_CLOEXEC))

        return self

    def close(self) -> Self:
        """
        Safely closes the pseudo database, releasing all resources.

        This is critical to ensure the database can safely be pickled across processes.
        """
        self.storages.clear()
        for h in self.handles:
            os.close(h)
        self.handles.clear()
        gc.collect()
        return self

    @contextlib.contextmanager
    def open_slice_as_path(self, idx_or_name: int | str) -> Iterator[str]:
        idx = self._get_idx_from_idx_or_name(idx_or_name)

        shard  = int(self.size_df.loc[idx, "shard"])
        offset = int(self.size_df.loc[idx, "offset"])
        length = int(self.size_df.loc[idx, "size"])
        src    = self.files_data[shard]

        use_cfr = True
        chunksz = 2 ** 20

        src_fd = os.open(src, os.O_RDONLY)
        try:
            memfd = os.memfd_create(f"simpledb-{shard}-{offset}", flags=os.MFD_CLOEXEC)
            try:
                remaining = length
                off_in    = offset
                off_out   = 0

                while remaining:
                    nreq = min(chunksz, remaining)
                    n = 0

                    if use_cfr:
                        try:
                            n = os.copy_file_range(src_fd, memfd, nreq, offset_src=off_in, offset_dst=off_out)
                            if n == 0:
                                warnings.warn("`copy_file_range` returned 0 bytes copied. Falling back to userspace copy.")
                                use_cfr = False
                                continue
                        except OSError as err:
                            warnings.warn(f"`copy_file_range` failed with {str(err)}. Falling back to userspace copy.")
                            use_cfr = False
                            continue

                    if (not use_cfr) and (n == 0):
                        chunk = os.pread(src_fd, nreq, off_in)
                        if not chunk:
                            break
                        n = os.pwrite(memfd, chunk, off_out)

                    remaining -= n
                    off_in    += n
                    off_out   += n

                yield f"/proc/self/fd/{memfd}"
            finally:
                os.close(memfd)
        finally:
            os.close(src_fd)


class CreateSimpleDBSample(NamedTuple):
    name: str
    data: bytes
    malware: bool
    timestamp: int
    family: Optional[str] = None


def roundup(x: int, to: int) -> int:
    """Rounds up x to the nearest multiple of to."""
    return ((x + to - 1) // to) * to


class CreateSimpleDB:
    """
    Create a SimpleDB from files and metadata.

    TODO: pad each individual sample to a multiple of PADDING for better paging performance.
    """

    def __init__(self, root: Path, shardsize: int = 2 ** 30) -> None:
        """
        Args:
            root (Path): The root directory of the SimpleDB.
            shardsize (int): The maximum size of each shard in bytes.
                Default is 1 GiB, which can hold about 1PB of binary data.
        """
        self.root = root
        self.dir_data = root / "data"
        self.dir_size = root / "size"
        self.dir_meta = root / "meta"
        self.root.mkdir(parents=True, exist_ok=False)
        self.dir_data.mkdir()
        self.dir_size.mkdir()
        self.dir_meta.mkdir()
        self.shardsize = shardsize

    @property
    def max_capacity(self) -> int:
        return self.shardsize * int(10 ** len('*******'))

    def __call__(self, samples: Iterable[CreateSimpleDBSample]) -> Self:
        """
        Create the SimpleDB from an iterable of samples.

        Args:
            samples (Iterable[CreateSimpleDBSample]): An iterable of samples to add to the SimpleDB.
        """
        # Data for the current shard.
        data, size_df, meta_df, num_samples = self._initialize_shard_containers()

        # Global counters for all shards
        shard_idx = 0
        sample_idx = 0

        for sample in samples:
            # If we should dump the current shard, do so, and reset the containers for this shard.
            if self._should_dump_shard_containers(data, size_df, meta_df, num_samples, sample):
                self._dump_shard_containers(data, size_df, meta_df, num_samples, shard_idx)
                data, size_df, meta_df, num_samples = self._initialize_shard_containers()
                shard_idx += 1
            # Get the data for this sample.
            data_ = bytearray(sample.data)
            size_df_ = pd.DataFrame({
                "idx": [sample_idx],
                "name": [sample.name],
                "shard": [shard_idx],
                "offset": [len(data)],
                "size": [len(data_)],
            })
            meta_df_ = pd.DataFrame({
                "idx": [sample_idx],
                "name": [sample.name],
                "shard": [shard_idx],
                "timestamp": [sample.timestamp],
                "malware": [1 if sample.malware else 0],
                "family": [sample.family if sample.family is not None else ""],
            })
            # Append the data for this sample to the current shard.
            data.extend(data_)
            size_df = pd.concat([size_df, size_df_], ignore_index=True)
            meta_df = pd.concat([meta_df, meta_df_], ignore_index=True)
            # Increment counters.
            sample_idx += 1
            num_samples += 1

        # Dump any remaining data.
        if num_samples > 0:
            self._dump_shard_containers(data, size_df, meta_df, num_samples, shard_idx)

        return self

    def _initialize_shard_containers(self) -> tuple[bytearray, pd.DataFrame, pd.DataFrame, int]:
        data = bytearray()
        size_df = pd.DataFrame(columns=["idx", "name", "shard", "offset", "size"])
        meta_df = pd.DataFrame(columns=["idx", "name", "shard", "timestamp", "malware", "family"])
        num_samples = 0
        return data, size_df, meta_df, num_samples

    def _should_dump_shard_containers(self, data: bytearray, size_df: pd.DataFrame, meta_df: pd.DataFrame, num_samples: int, sample: CreateSimpleDBSample) -> bool:
        # Never dump if we have no samples.
        if num_samples == 0:
            return False
        # Dump if the current sample is larger than the shard size.
        if len(sample.data) > self.shardsize:
            warnings.warn(
                f"Sample {sample.name} is larger than the shard size {self.shardsize}. "
                f"It will be stored in its own shard of size {roundup(len(sample.data), PADDING)}."
            )
            return True
        # Dump if adding the current sample would exceed the shard size.
        if roundup(len(data) + len(sample.data), PADDING) > self.shardsize:
            return True
        # Otherwise, do not dump.
        return False

    def _dump_shard_containers(self, data: bytearray, size_df: pd.DataFrame, meta_df: pd.DataFrame, num_samples: int, shard_idx: int) -> None:
        # Never dump if we have no samples.
        if num_samples == 0:
            return
        # Get the prefix for the shard.
        if shard_idx >= 10 ** len('*******'):
            raise RuntimeError(f"Exceeded maximum number of shards ({10 ** len('*******')}).")
        prefix = f"{(len('*******') - len(str(shard_idx))) * '0'}{shard_idx}"
        # Pad the binary data to a multiple of PADDING.
        padding = roundup(len(data), PADDING) - len(data)
        data.extend(b"\x00" * padding)
        Path(self.dir_data / f"data-{prefix}.bin").write_bytes(data)
        # Save the size and meta dataframes.
        size_df.to_csv(self.dir_size / f"size-{prefix}.csv", index=False)
        meta_df.to_csv(self.dir_meta / f"meta-{prefix}.csv", index=False)
