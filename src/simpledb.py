"""
A simple wrapper over millions of files for faster I/O on slow filesystems.
"""

from __future__ import annotations
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from concurrent.futures import Future
import ctypes
import ctypes.util
from enum import Enum
import gc
import os
from pathlib import Path
import random
import shutil
from typing import Any
from typing import Generator
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Self
from types import TracebackType
import warnings

import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch import ByteTensor
from torch import UntypedStorage
from tqdm import tqdm
import zstandard as zstd

from typing import Union, Optional
from os import PathLike
import zstandard as zstd


PAGESIZE = 4096


_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_POSIX_FADV_NORMAL     = 0
_POSIX_FADV_RANDOM     = 1
_POSIX_FADV_SEQUENTIAL = 2
_POSIX_FADV_WILLNEED   = 3
_POSIX_FADV_DONTNEED   = 4
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

    Each data-*.bin file contains concatenated binary blobs of each entry.
    Each size-*.csv file contains the size of each entry, essential for indexing.
    Each meta-*.csv file contains non-essential metadata for each entry.

    data-*******.bin[.zst]
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

    def __init__(self, dir_root: Path, *, allow_name_indexing: bool = False) -> None:
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
        return sorted(self.dir_data.glob("data-*.bin*"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_size(self) -> list[Path]:
        return sorted(self.dir_size.glob("size-*.csv"), key=lambda p: p.stem.split("-")[-1])

    @property
    def files_meta(self) -> list[Path]:
        return sorted(self.dir_meta.glob("meta-*.csv"), key=lambda p: p.stem.split("-")[-1])

    def compute_shardwise_stats(self) -> pd.DataFrame:
        """
        Computes a DataFrame with the distribution of malware and families per shard.

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - shard (int): The shard number.
                - num_samples (int): The number of samples in the shard.
                - num_malware (int): The number of malware samples in the shard.
                - num_benign (int): The number of benign samples in the shard.
                - malware_fraction (float): The fraction of malware samples in the shard.
        """
        stats = []
        for f in self.files_meta:
            df = pd.read_csv(f, converters={"family": lambda s: s if s else ""})
            shard_idx = int(f.stem.split("-")[-1])
            num_samples = len(df)
            num_malware = int((df["malware"] == 1).sum())
            num_benign = int((df["malware"] == 0).sum())
            malware_fraction = num_malware / num_samples if num_samples > 0 else 0.0
            stats.append({
                "shard": shard_idx,
                "num_samples": num_samples,
                "num_malware": num_malware,
                "num_benign": num_benign,
                "malware_fraction": malware_fraction,
            })
        return pd.DataFrame(stats).sort_values("shard").reset_index(drop=True)

    def number_of_samples_in_shards(self) -> npt.NDArray[np.int64]:
        """
        Returns an array whose i-th element is the number of samples in the i-th shard.
        """
        return self.size_df.groupby("shard").size().to_numpy().astype(np.int64)  # type: ignore[no-any-return]

    def get_size_df(self) -> pd.DataFrame:
        size_dfs = [pd.read_csv(f) for f in self.files_size]
        for f, df in zip(self.files_data, size_dfs):
            if f.suffix != ".zst" and (df["offset"] + df["size"]).max() > os.path.getsize(f):
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
        fastdir: Optional[str | Path] = None,
        pread_block_bytes: int = 32 * 2 ** 20,
        merge_slack_bytes: int = 0,
        prefetch_next_window: bool = True,
        prefetch_max_bytes: int = 1 * 2 ** 20,
        use_readahead: bool = False,
    ) -> None:
        """
        Args:
            fastdir (str | Path | None):
                If not None, a directory to which shards will be copied in the background
                before being read. This can help if the original data is on a slow or
                heavily contended filesystem. The fastdir directory should be on a fast
                local disk with enough space to hold at least one shard.
            pread_block_bytes (int):
                Target size of each contiguous window read from a shard with a single pread.
                Larger windows reduce syscall overhead and improve throughput, at the cost of
                higher peak RAM per worker and more page-cache churn with many workers.
            merge_slack_bytes (int):
                Allowance to extend a window past its nominal end to include the next sample
                if it would otherwise straddle the boundary. Reduces the number of small tail
                reads near block boundaries. Typical values 0-4 MiB.
            prefetch_next_window (bool):
                If True, issue posix_fadvise(WILLNEED) on the next window so the kernel can
                start pulling it into the page cache while you process the current window.
                Cheap and usually beneficial; default True.
            use_readahead (bool):
                If True, additionally call readahead(fd, next_base, next_len).
                This actively queues IO (stronger than a hint). Often unnecessary on NVMe/SSD,
                but can help on spinning disks or network filesystems. Default False.
        """
        super().__init__(dir_root, allow_name_indexing=allow_name_indexing)
        self.fastdir = Path(fastdir) if fastdir is not None else None
        if self.fastdir is not None:
            self.fastdir.mkdir(exist_ok=True)
        self.block_bytes = pread_block_bytes
        self.merge_slack = merge_slack_bytes
        self.prefetch_next_window = prefetch_next_window
        self.prefetch_max_bytes = prefetch_max_bytes
        self.use_readahead = use_readahead
        self._fd = -1

    def __iter__(self) -> Iterator[SimpleDBSample]:
        yield from self.iter_shards(range(len(self.files_data)))

    def iter_shards(self, shard_indices: Sequence[int]) -> Iterator[SimpleDBSample]:
        shardfiles = [self.files_data[i] for i in shard_indices]

        if self.fastdir is None or len(shardfiles) == 1:
            for shard_idx, shardfile in zip(shard_indices, shardfiles):
                yield from self.iter_one_shard(shard_idx, str(shardfile))
            return

        assert self.fastdir is not None
        if not isinstance(self.fastdir, Path):
            raise TypeError("fastdir must be a Path or None.")
        if not self.fastdir.exists():
            raise FileNotFoundError(f"Fastdir {self.fastdir} does not exist.")

        copier = ShardCopier(self.fastdir, max_workers=2)
        try:
            copier.prefetch(shardfiles[1])
            for k, (shard_idx, shardfile) in enumerate(zip(shard_indices, shardfiles)):
                shardfile = copier.staged_or_src(shardfile, wait=1.0)
                if k + 1 < len(shardfiles):
                    copier.prefetch(shardfiles[k + 1])
                yield from self.iter_one_shard(shard_idx, str(shardfile))
        finally:
            copier.close()

    def iter_one_shard(self, shard_idx: int, shardfile: Optional[str] = None) -> Iterator[SimpleDBSample]:
        if not self.is_open:
            raise RuntimeError("Database is not open. Call open() before accessing data.")
        sub_df = self.size_df[self.size_df["shard"] == shard_idx][["idx", "name", "shard", "offset", "size"]]
        rows = sub_df.sort_values(["offset", "idx"]).reset_index(drop=True)
        if rows.empty:
            return

        shardfile = str(self.files_data[shard_idx]) if shardfile is None else shardfile
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
                    nb_len = min(nbytes, fsz - nb_base, self.prefetch_max_bytes)
                    _posix_fadvise(self._fd, nb_base, nb_len, _POSIX_FADV_WILLNEED)
                    if self.use_readahead:
                        _readahead(self._fd, nb_base, nb_len)

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
            _posix_fadvise(self._fd, base, nbytes, _POSIX_FADV_DONTNEED)

        os.close(self._fd)
        self._fd = -1

    def close(self) -> Self:
        super().close()
        if self._fd != -1:
            os.close(self._fd)
            self._fd = -1
        return self


class ShardCopier:
    """
    Background copier for pre-staging shards to a fast local disk.
    """

    def __init__(self, dir: str | Path, *, max_workers: int = 1):
        self.dir = Path(dir)
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="precopy")
        self.futs: dict[Path, Future[Path]] = {}

    def _copy_streaming(self, src: Path) -> Path:
        """Copy `src` and return the destination path when finished."""
        dst = self.dir / src.name
        tmp = dst.with_suffix(dst.suffix + ".part")
        self.dir.mkdir(parents=True, exist_ok=True)

        # If dst already exists and looks complete, skip copy.
        if dst.exists():
            s = src.stat()
            d = dst.stat()
            if d.st_size == s.st_size and d.st_mtime >= s.st_mtime:
                return dst

        # Otherwise, perform a cross-filesystem safe copy.
        with open(src, "rb", buffering=1024 * 1024) as r, open(tmp, "wb", buffering=1024 * 1024) as w:
            shutil.copyfileobj(r, w, length=8 * 1024 * 1024)  # type: ignore[misc]
            w.flush()
            os.fsync(w.fileno())
        os.replace(tmp, dst)

        return dst

    def prefetch(self, src: Path) -> None:
        """If not already in progress, start copying `src` in the background."""
        if src not in self.futs:
            self.futs[src] = self.pool.submit(self._copy_streaming, src)

    def staged_or_src(self, src: Path, *, wait: float = 0.0) -> Path:
        """If the copy of `src` is ready, return the copied path; else return `src`."""
        fut = self.futs.pop(src, None)
        if not fut:
            return src
        if fut.done():
            return fut.result()
        if wait > 0:
            try:
                return fut.result(timeout=wait)
            except TimeoutError:
                pass
        return src

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


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
    def __call__(self, samples: Iterable["CreateSimpleDBSample"]) -> "CreateSimpleDB":
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

    def add(self, sample: CreateSimpleDBSample) -> None:
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

    def _should_dump_shard_containers(self, data: bytearray, size_df: pd.DataFrame, meta_df: pd.DataFrame, num_samples: int, sample: "CreateSimpleDBSample") -> bool:
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


def split_simple_db(
    dir_root: Path,
    dirs_out: list[Path],
    indices: list[Sequence[int] | npt.NDArray[np.integer[Any]] | torch.Tensor],
    shardsize: int = 2 ** 30,
    *,
    shuffle: bool = True,
    poolsize: int = 16384,
    seed: Optional[int] = None,
) -> None:
    """
    Split a SimpleDB into multiple SimpleDBs.

    NOTE: This algorithm's randomization capabalities are somewhat limited across shards unless
    poolsize is large relative to the shardsize of the source database. Recommend you use SimpleDB's
    `compute_shardwise_stats` method to examind each shard's data distribution before and after splitting.
    """
    if len(dirs_out) != len(indices):
        raise ValueError("dirs_out and indices must have the same length.")

    rng = random.Random(seed)

    # For each output, convert indices to a set for fast lookup.
    idxsets: list[set[int]] = []
    for idx in indices:
        if isinstance(idx, (np.ndarray, torch.Tensor)):
            idx = idx.tolist()
        idxsets.append(set(idx))

    builders = [CreateSimpleDB(out_dir, shardsize=shardsize) for out_dir in dirs_out]
    pools: list[list[CreateSimpleDBSample]] = [[] for _ in dirs_out]

    src = IterableSimpleDB(dir_root)
    shard_ids = [int(p.stem.split("-")[-1]) for p in src.files_data]
    if shuffle:
        rng.shuffle(shard_ids)

    try:
        src.open()
        for shard in tqdm(shard_ids, desc="Processing shards", unit="shard"):
            # Build the "rows" view to recover idx order used by iter_one_shard.
            sub_df = src.size_df[src.size_df["shard"] == shard][["idx", "offset"]]
            rows = sub_df.sort_values(["offset", "idx"]).reset_index(drop=True)
            idx_in_order = rows["idx"].astype(int).tolist()

            # Stream samples for this shard and zip with the true idxs.
            for idx_val, sample in zip(idx_in_order, src.iter_one_shard(shard)):
                # Add sample to each new database that wants it.
                for out_j, idxset in enumerate(idxsets):
                    if idx_val not in idxset:
                        continue
                    # Create the Sample to add.
                    rec = CreateSimpleDBSample(
                        name=sample.name,
                        data=bytes(sample.bview),
                        malware=bool(sample.malware),
                        timestamp=int(sample.timestamp),
                        family=(sample.family if sample.family else None),
                    )
                    # If not shuffling, add directly to builder.
                    if not shuffle:
                        builders[out_j].add(rec)
                        continue
                    # Add to reservoir pool or randomly evict from the pool to builder.
                    pool = pools[out_j]
                    if len(pool) < poolsize:
                        pool.append(rec)
                    else:
                        j = rng.randrange(len(pool))
                        builders[out_j].add(pool[j])
                        pool[j] = rec

        # Drain all pools in randomized order.
        for out_j, pool in enumerate(pools):
            while pool:
                j = rng.randrange(len(pool))
                builders[out_j].add(pool.pop(j))
    finally:
        src.close()
        for b in builders:
            try:
                b.close()
            except Exception:
                pass
