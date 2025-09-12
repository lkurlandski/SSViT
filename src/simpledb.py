"""
A simple wrapper over millions of files for faster I/O on slow filesystems.
"""

from __future__ import annotations
from collections.abc import Iterable
from collections.abc import Iterator
import ctypes
import ctypes.util
from enum import Enum
import gc
import os
from pathlib import Path
import random
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Self
from types import TracebackType
import warnings

import pandas as pd
import torch
from torch import ByteTensor
from torch import UntypedStorage


PAGESIZE = 4096


_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_POSIX_FADV_NORMAL     = 0
_POSIX_FADV_RANDOM     = 1
_POSIX_FADV_SEQUENTIAL = 2
_POSIX_FADV_WILLNEED   = 3
_HAS_FADVICE   = hasattr(_LIBC, "posix_fadvise")
_HAS_READAHEAD = hasattr(_LIBC, "readahead")


class FADViseMode(Enum):
    SEQ = "seq"  # SEQUENTIAL
    RND = "rnd"  # RANDOM
    OFF = "off"  # NORMAL


def _posix_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
    if not _HAS_FADVICE:
        warnings.warn("posix_fadvise called but not available on this system.")
        return
    _ = _LIBC.posix_fadvise(ctypes.c_int(fd), ctypes.c_long(offset), ctypes.c_long(length), ctypes.c_int(advice))


def _readahead(fd: int, offset: int, length: int) -> None:
    if not _HAS_READAHEAD:
        warnings.warn("readahead called but not available on this system.")
        return
    _ = _LIBC.readahead(ctypes.c_int(fd), ctypes.c_long(offset), ctypes.c_size_t(length))


def _pread_into_tensor(fd: int, offset: int, nbytes: int, *, pin: bool = False) -> torch.Tensor:
    """
    Read exactly `nbytes` from `fd` at `offset` into a fresh 1-D uint8 tensor.
    """
    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
    mv = memoryview(buf.numpy()).cast("B")
    got, off = 0, offset
    while got < nbytes:
        n = os.preadv(fd, [mv[got:]], off)
        if n == 0:
            raise EOFError(f"EOF: need {nbytes - got} more bytes at off={off}")
        got += n
        off += n
    return buf

def _align_down(x: int, a: int) -> int:
    return (x // a) * a


class SimpleDBSample(NamedTuple):
    """
    Information returned from the SimpleDB for each sample.
    """
    name: str
    data: ByteTensor
    bview: memoryview
    malware: bool
    timestamp: int
    family: str


class SimpleDB:
    """
    A simple database that provides fast(-ish) access to a large number of files.

    Usage:
        >>> db = SimpleDB(Path("/path/to/db"))
        >>> db.is_open
        False
        >>> with db.open() as openeddb:
        ...     openeddb.is_open
        True

    Currently, the database can handle a maximum of 10^8 shard files, each of which can contain
    an arbitrary number of individual entries. Balancing the size of the shard vs the number of
    samples within each shard is a tradeoff between system performance and shuffling granularity.    

    At the moment, the database opens every shard file simulateously, so the maximum number of
    shards is limited by the maximum number of open file descriptors. Use `ulimit -n` to check
    and set this limit.

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

    data-*******.bin
    -------------
    This is a simple binary file containing concatenated binary blobs of each entry.
    Each sample is padded to a multiple of `PAGESIZE` bytes for paging efficiency.

    size-*******.csv
    -------------
    This is a CSV file containing the offset and size of each entry in the corresponding data-*******.bin file.
    It is absolutely critical to the operation of the pseudo database and should not be modified.
    It has the following columns:
        - idx (int): the unique integer ID of the entry.
        - name (str): the name of the entry.
        - shard (int): the shard number (coresponding to the '*******'). This is redundant but useful.
        - offset (int): the offset of the entry in the data-*******.bin file.
        - size (int): the size of the entry in bytes.

    meta-*******.csv
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
    """

    def __init__(self, dir_root: Path, *, allow_name_indexing: bool) -> None:
        """
        Args:
            dir_root (Path): The root directory of the SimpleDB.
            allow_name_indexing (bool): Whether to allow indexing by name, as opposed to index.
        """
        self.dir_root = dir_root
        self.allow_name_indexing = allow_name_indexing
        self.dir_data = dir_root / "data"
        self.dir_size = dir_root / "size"
        self.dir_meta = dir_root / "meta"
        self.size_df: pd.DataFrame = self.get_size_df()
        self.meta_df: pd.DataFrame = self.get_meta_df()
        self.name_map: dict[str, int] = self.get_name_map()
        if not (len(self.files_data) == len(self.files_size) == len(self.files_meta)):
            raise RuntimeError("Number of data, size, and meta files do not match.")
        if len(self.size_df) != len(self.meta_df):
            raise RuntimeError(f"Size and meta data have different number of rows ({len(self.size_df)} vs {len(self.meta_df)}).")
        self._is_open = False

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, exc_type: type[Exception], exc: Exception, tb: TracebackType) -> Literal[False]:
        try:
            self.close()
        except Exception:
            pass
        return False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def files_data(self) -> list[Path]:
        return sorted(self.dir_data.glob("data-*.bin"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_size(self) -> list[Path]:
        return sorted(self.dir_size.glob("size-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_meta(self) -> list[Path]:
        return sorted(self.dir_meta.glob("meta-*.csv"), key=lambda p: p.stem.split("-")[-1])

    def get_size_df(self) -> pd.DataFrame:
        size_dfs = [pd.read_csv(f) for f in self.files_size]
        for f, df in zip(self.files_data, size_dfs):
            if (df["offset"] + df["size"]).max() > os.path.getsize(f):
                raise RuntimeError("Size file contains an entry that exceeds the size its data file.")
        size_df = pd.concat(size_dfs, ignore_index=True)
        size_df = size_df.sort_values(by="idx").set_index("idx", drop=False)
        size_df.index.name = None
        return size_df

    def get_meta_df(self) -> pd.DataFrame:
        meta_dfs = [pd.read_csv(f, converters={"family": lambda s: s if s else ""}) for f in self.files_meta]
        meta_df = pd.concat(meta_dfs, ignore_index=True)
        meta_df = meta_df.sort_values(by="idx").set_index("idx", drop=False)
        meta_df.index.name = None
        return meta_df

    def get_idx_from_idx_or_name(self, idx_or_name: int | str) -> int:
        if isinstance(idx_or_name, int):
            return idx_or_name
        if isinstance(idx_or_name, str):
            if not self.allow_name_indexing:
                raise ValueError("Name indexing is not allowed. Set allow_name_indexing=True to enable it.")
            return self.name_map[idx_or_name]
        raise TypeError("Index must be an integer or string.")

    def get_name_map(self) -> dict[str, int]:
        if not self.allow_name_indexing:
            return {}
        if not self.meta_df["name"].is_unique:
            raise RuntimeError("Names are not unique, so name indexing is not possible.")
        name_map = {str(row["name"]): int(row["idx"]) for _, row in self.meta_df.iterrows()}
        return name_map

    def open(self) -> Self:
        """
        Perform any initialization required to open the database.
        """
        self._is_open = True
        return self

    def close(self) -> Self:
        """
        Perform any cleanup required to close the database.
        """
        self._is_open = False
        return self


class RandomMappingSimpleDB(SimpleDB):
    """
    SimpleDB optimized for random accesses across shards.
    """

    def __init__(self, dir_root: Path, *, allow_name_indexing: bool = False, num_open: int = 1, backend: Literal["mmap", "pread"] = "mmap") -> None:
        """
        Args:
            num_open (int): The maximum number of shard files to keep open at once.
            backend (str): The backend to use for reading data. Either "mmap" or "pread".
        """
        super().__init__(dir_root, allow_name_indexing=allow_name_indexing)
        self.num_open = min(num_open, len(self.files_data))
        if num_open < 1:
            raise ValueError("num_open must be at least 1.")
        self.backend = backend
        if backend not in ("mmap", "pread"):
            raise ValueError("backend must be either 'mmap' or 'pread'.")
        self._storages: list[Optional[UntypedStorage]] = [None for _ in self.files_data]
        self._fds: list[Optional[int]] = [None for _ in self.files_data]

    def __getitem__(self, idx_or_name: int | str) -> SimpleDBSample:
        if not self.is_open:
            raise RuntimeError("Database is not open. Call open() before accessing data.")

        idx = self.get_idx_from_idx_or_name(idx_or_name)

        size_dict: dict[str, Any] = self.size_df.loc[idx].to_dict()
        meta_dict: dict[str, Any] = self.meta_df.loc[idx].to_dict()

        shard = size_dict["shard"]
        offset = size_dict["offset"]
        length = size_dict["size"]
        shardfile = str(self.files_data[shard])

        if self.backend == "mmap":
            if self._storages[shard] is None:
                opened = [i for i, s in enumerate(self._storages) if s is not None]
                assert len(opened) == self.num_open, "There should always be `num_open` storages open."
                i = random.choice(opened)
                self._storages[i] = None
                self._storages[shard] = UntypedStorage.from_file(shardfile, shared=False, nbytes=os.path.getsize(shardfile))
            data = torch.empty(0, dtype=torch.uint8)
            data.set_(self._storages[shard], storage_offset=offset, size=(length,), stride=(1,))

        elif self.backend == "pread":
            if self._fds[shard] is None:
                opened = [i for i, s in enumerate(self._fds) if s is not None]
                assert len(opened) == self.num_open, "There should always be `num_open` storages open."
                i = random.choice(opened)
                os.close(self._fds[i])  # type: ignore[arg-type]
                self._fds[i] = None
                self._fds[shard] = os.open(shardfile, os.O_RDONLY | os.O_CLOEXEC)
                _posix_fadvise(self._fds[shard], 0, 0, _POSIX_FADV_RANDOM)
            data = _pread_into_tensor(self._fds[shard], offset, length, pin=False)

        bview = memoryview(data.numpy()).cast("B")

        return SimpleDBSample(
            name=size_dict["name"],
            data=data,
            bview=bview,
            malware=meta_dict["malware"] == 1,
            timestamp=meta_dict["timestamp"],
            family=meta_dict["family"],
        )

    def open(self) -> Self:
        super().open()
        if self.backend == "mmap":
            for i, f in enumerate(self.files_data[0:self.num_open]):
                self._storages[i] = UntypedStorage.from_file(str(f), shared=False, nbytes=os.path.getsize(f))
        elif self.backend == "pread":
            for i, f in enumerate(self.files_data[0:self.num_open]):
                fd = os.open(str(f), os.O_RDONLY | os.O_CLOEXEC)
                _posix_fadvise(fd, 0, 0, _POSIX_FADV_RANDOM)
                self._fds[i] = fd
        return self

    def close(self) -> Self:
        super().close()
        if self.backend == "mmap":
            for i in range(len(self._storages)):
                self._storages[i] = None
        elif self.backend == "pread":
            for i in range(len(self._fds)):
                if self._fds[i] is not None:
                    os.close(self._fds[i])  # type: ignore[arg-type]
                self._fds[i] = None
        return self


class ChunkedMappingSimpleDB(SimpleDB):
    """
    SimpleDB optimized for random accesses within a shard.
    """

    def __init__(self, dir_root: Path, *, allow_name_indexing: bool = False) -> None:
        super().__init__(dir_root, allow_name_indexing=allow_name_indexing)
        self._curshard = -1
        self._storage: Optional[UntypedStorage] = None

    def __getitem__(self, idx_or_name: int | str) -> SimpleDBSample:
        if not self.is_open:
            raise RuntimeError("Database is not open. Call open() before accessing data.")

        idx = self.get_idx_from_idx_or_name(idx_or_name)

        size_dict: dict[str, Any] = self.size_df.loc[idx].to_dict()
        meta_dict: dict[str, Any] = self.meta_df.loc[idx].to_dict()

        shard = size_dict["shard"]
        offset = size_dict["offset"]
        length = size_dict["size"]
        shardfile = str(self.files_data[shard])

        if self._curshard != shard:
            self._curshard = shard
            self._storage = UntypedStorage.from_file(shardfile, shared=False, nbytes=os.path.getsize(shardfile))

        data = torch.empty(0, dtype=torch.uint8)
        data.set_(self._storage, storage_offset=offset, size=(length,), stride=(1,))
        bview = memoryview(data.numpy()).cast("B")

        return SimpleDBSample(
            name=size_dict["name"],
            data=data,
            bview=bview,
            malware=meta_dict["malware"] == 1,
            timestamp=meta_dict["timestamp"],
            family=meta_dict["family"],
        )

    def close(self) -> Self:
        super().close()
        self._curshard = -1
        self._storage = None
        return self


class IterableSimpleDB(SimpleDB):
    """
    SimpleDB optimized for sequential accesses.
    """

    def __init__(
        self,
        dir_root: Path,
        *,
        allow_name_indexing: bool = False,
        pread_block_bytes: int = 64 * 2 ** 20,
        merge_slack_bytes: int = 0,
        prefetch_next_window: bool = True,
        use_readahead: bool = False,
    ) -> None:
        """
        Args:
            pread_block_bytes (int): The block size to use for pread. Larger blocks
                can improve performance, but increase memory usage. Default is 64 MiB.
            merge_slack_bytes (int): The maximum number of bytes to extend a read window
                beyond the current block to include additional samples. This can reduce
                the number of read calls, but increases memory usage. Default is 0.
            prefetch_next_window (bool): Whether to prefetch the next window of data.
                This can improve performance when reading sequentially, but increases
                memory usage. Default is True.
            use_readahead (bool): Whether to use readahead to prefetch the next window
                of data. This can improve performance when reading sequentially, but
                increases memory usage. Default is False.
        """
        super().__init__(dir_root, allow_name_indexing=allow_name_indexing)
        self.block_bytes = pread_block_bytes
        self.merge_slack = merge_slack_bytes
        self.prefetch_next_window = prefetch_next_window
        self.use_readahead = use_readahead
        self._fd = -1

    def __iter__(self) -> Iterator[SimpleDBSample]:
        for f in self.files_data:
            shard_idx = int(f.stem.split("-")[-1])
            yield from self.iter_one_shard(shard_idx)

    def iter_one_shard(self, shard_idx: int) -> Iterator[SimpleDBSample]:
        if not self.is_open:
            raise RuntimeError("Database is not open. Call open() before accessing data.")
        sub_df = self.size_df[self.size_df["shard"] == shard_idx][["idx", "name", "shard", "offset", "size"]]
        rows = sub_df.sort_values(["offset", "idx"]).reset_index(drop=True)
        if rows.empty:
            return

        shardfile = str(self.files_data[shard_idx])
        self._fd = os.open(shardfile, os.O_RDONLY | os.O_CLOEXEC)
        _posix_fadvise(self._fd, 0, 0, _POSIX_FADV_SEQUENTIAL)
        fsz = os.path.getsize(shardfile)

        block = int(self.block_bytes)
        slack = int(self.merge_slack)

        i = 0
        while i < len(rows):
            r0 = rows.iloc[i]
            first_off  = int(r0["offset"])
            first_size = int(r0["size"])
            # Window start aligned to block; ensure at least one full block or the whole sample
            base = _align_down(first_off, block)
            end  = base + max(block, (first_off + first_size) - base)

            # Greedily extend while consecutive samples fit within [base, end] + slack
            j = i + 1
            while j < len(rows):
                rj = rows.iloc[j]
                o  = int(rj["offset"]); s = int(rj["size"])
                if o + s <= end + slack and end < fsz:
                    end = max(end, o + s)
                    j += 1
                else:
                    break

            if end > fsz:
                end = fsz
            nbytes = end - base
            if nbytes <= 0:
                break

            # Hint current and (optionally) next window
            _posix_fadvise(self._fd, base, nbytes, _POSIX_FADV_WILLNEED)
            if self.prefetch_next_window:
                nb_base = end
                if nb_base < fsz:
                    nb_len = min(nbytes, fsz - nb_base)
                    _posix_fadvise(self._fd, nb_base, nb_len, _POSIX_FADV_WILLNEED)
                    if self.use_readahead and _HAS_READAHEAD:
                        try:
                            _readahead(self._fd, nb_base, nb_len)
                        except Exception:
                            pass

            # Read the whole window in one preadv loop
            win = _pread_into_tensor(self._fd, base, nbytes, pin=False)

            # Views/slices per row i..j-1
            base_mv = memoryview(win.numpy()).cast("B")
            for k in range(i, j):
                rk = rows.iloc[k]
                idx = int(rk["idx"])
                name = str(rk["name"])
                off = int(rk["offset"]); sz = int(rk["size"])
                start = off - base

                tview = win.narrow(0, start, sz)
                bview = base_mv[start:start + sz]

                mrow = self.meta_df.loc[idx]
                yield SimpleDBSample(
                    name=name,
                    data=tview,
                    bview=bview,
                    malware=bool(int(mrow["malware"]) == 1),
                    timestamp=int(mrow["timestamp"]),
                    family=str(mrow["family"]),
                )

            i = j  # advance to first not-yet-emitted row

        os.close(self._fd)
        self._fd = -1

    def close(self) -> Self:
        super().close()
        if self._fd != -1:
            os.close(self._fd)
            self._fd = -1
        return self


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
            # Append the data and padding bytes for this sample to the current shard.
            npad = roundup(len(data_), PAGESIZE) - len(data_)
            bpad = b"\x00" * npad
            data.extend(data_)
            data.extend(bpad)
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
                f"It will be stored in its own shard of size {roundup(len(sample.data), PAGESIZE)}."
            )
            return True
        # Dump if adding the current sample would exceed the shard size.
        if roundup(len(data) + len(sample.data), PAGESIZE) > self.shardsize:
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
        # Write the data, size, and meta files for this shard.
        Path(self.dir_data / f"data-{prefix}.bin").write_bytes(data)
        size_df.to_csv(self.dir_size / f"size-{prefix}.csv", index=False)
        meta_df.to_csv(self.dir_meta / f"meta-{prefix}.csv", index=False)
