"""
Tests.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Generator
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from src.simpledb import PADDING
from src.simpledb import roundup
from src.simpledb import SimpleDB
from src.simpledb import CreateSimpleDB
from src.simpledb import CreateSimpleDBSample

# ---------- helpers ----------


def _mk_bytes(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def _samples(count: int, *, min_len: int = 100, max_len: int = 10000, seed: int = 0) -> Generator[CreateSimpleDBSample, None, None]:
    rng = random.Random(seed)
    for i in range(count):
        name = f"file-{i:08d}.exe"
        n = rng.randint(min_len, max_len)
        data = _mk_bytes(n, seed=seed + i)
        malware = bool(rng.getrandbits(1))
        ts = 1_600_000_000 + i
        fam = None if (i % 3 == 0) else f"fam{i%7}"
        yield CreateSimpleDBSample(name=name, data=data, malware=malware, timestamp=ts, family=fam)

# ---------- unit tests ----------

def test_roundup_basic() -> None:
    assert roundup(0, 4096) == 0
    assert roundup(1, 4096) == 4096
    assert roundup(4095, 4096) == 4096
    assert roundup(4096, 4096) == 4096
    assert roundup(4097, 4096) == 8192


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_build_one_shard_and_read(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    builder = CreateSimpleDB(root, shardsize=2 * 1024 * 1024)  # small-ish shard
    samples = list(_samples(25, min_len=5000, max_len=12000, seed=123))
    builder(samples)

    # Expect exactly one data/size/meta shard
    data_files = sorted((root / "data").glob("data-*.bin"))
    size_files = sorted((root / "size").glob("size-*.csv"))
    meta_files = sorted((root / "meta").glob("meta-*.csv"))
    assert len(data_files) == len(size_files) == len(meta_files) == 1

    # PADDING is applied to the end of each shard
    sz = os.path.getsize(data_files[0])
    assert sz % PADDING == 0

    # Read back with SimpleDB and verify content & metadata
    db = SimpleDB(root, reader=reader, allow_name_indexing=True)
    assert db.files_data == data_files
    assert db.files_size == size_files
    assert db.files_meta == meta_files
    try:
        db = db.open()
        for idx, s in enumerate(samples):
            rec = db[idx]
            assert rec.name == s.name
            assert isinstance(rec.data, torch.Tensor)
            assert rec.data.dtype == torch.uint8
            assert rec.data.numel() == len(s.data)
            # bytes equality
            assert bytes(rec.data.tolist()) == s.data
            # meta roundtrip
            assert rec.malware == s.malware
            assert rec.timestamp == s.timestamp
            if s.family is None:
                assert rec.family in ("", None)
            else:
                assert rec.family == s.family

            # name indexing works
            by_name = db[s.name]
            assert by_name.name == rec.name
            assert bytes(by_name.data.tolist()) == s.data
    finally:
        db.close()


def test_shards_rollover(tmp_path: Path) -> None:
    root = tmp_path / "db"
    # Force small shards so we roll over multiple times
    builder = CreateSimpleDB(root, shardsize=128 * 1024)  # 128 KiB
    big_samples = list(_samples(40, min_len=2000, max_len=5000, seed=7))
    builder(big_samples)

    data_files = sorted((root / "data").glob("data-*.bin"))
    size_files = sorted((root / "size").glob("size-*.csv"))
    meta_files = sorted((root / "meta").glob("meta-*.csv"))
    assert len(data_files) == len(size_files) == len(meta_files) >= 2

    # Filenames are zero-padded and monotonic
    names = [p.name for p in data_files]
    assert names == sorted(names)
    assert all(name.startswith("data-") and name.endswith(".bin") for name in names)

    # Verify offsets and sizes fit within each shard (consistency)
    for size_csv, data_bin in zip(size_files, data_files):
        df = pd.read_csv(size_csv)
        total = os.path.getsize(data_bin)
        for _, row in df.iterrows():
            off, ln = int(row["offset"]), int(row["size"])
            assert 0 <= off
            assert off + ln <= total


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_name_indexing_toggle(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("x.exe", b"\x01\x02", True, 42, None)])
    db = SimpleDB(root, reader=reader, allow_name_indexing=False).open()
    try:
        with pytest.raises(ValueError):
            _ = db["x.exe"]
    finally:
        db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_open_slice_as_path(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("x.exe", b"\x01\x02", True, 42, None)])
    db = SimpleDB(root, reader=reader, allow_name_indexing=True).open()
    try:
        with db.open_slice_as_path(0) as file:
            assert isinstance(file, (Path, str))
            assert os.path.isfile(file)
            assert os.path.exists(file)
            assert Path(file).read_bytes() == b"\x01\x02"
    finally:
        db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_close_is_idempotent_and_clears(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("x.exe", b"\x00", False, 0, None)])
    db = SimpleDB(root, reader=reader).open()
    # Should not raise, and should clear storages
    db.close()
    assert db.storages == []
    # Call again to ensure idempotent
    db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_oversized_sample_warns_and_is_present(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    big = b"\xff" * (256 * 1024)  # 256 KiB
    small = b"\x01\x02"
    builder = CreateSimpleDB(root, shardsize=64 * 1024)  # 64 KiB shard -> big is oversized
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        builder([
            CreateSimpleDBSample("a.exe", small, False, 1, None),
            CreateSimpleDBSample("big.exe", big, True, 2, "fam"),
            CreateSimpleDBSample("b.exe", small, False, 3, None),
        ])
        assert any("larger than the shard size" in str(x.message) for x in w)

    db = SimpleDB(root, reader=reader, allow_name_indexing=True).open()
    try:
        rec = db["big.exe"]
        assert rec.data.numel() == len(big)
        assert rec.malware is True
        assert rec.family == "fam"
    finally:
        db.close()


def test_sorted_file_mapping_and_storage_size(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root, shardsize=512 * 1024)([
        CreateSimpleDBSample("a", _mk_bytes(1024, 1), True, 1),
        CreateSimpleDBSample("b", _mk_bytes(2048, 2), False, 2),
        CreateSimpleDBSample("c", _mk_bytes(4096, 3), True, 3),
    ])
    db = SimpleDB(root).open()
    try:
        # storage sizes match on-disk sizes (prevents zero-sized mapping regressions)
        for f, st in zip(db.files_data, db.storages):
            assert len(st) == os.path.getsize(f)
        # shard indices in CSV line up with storage order (sorted)
        size_paths = db.files_size
        for shard_idx, size_csv in enumerate(size_paths):
            df = pd.read_csv(size_csv)
            assert (df["shard"] == shard_idx).all()
    finally:
        db.close()


def test_boundary_reads_and_last_entry_before_padding(tmp_path: Path) -> None:
    root = tmp_path / "db"
    # Create entries that end just before padding boundary
    entry1 = _mk_bytes(PADDING - 7, 10)  # will force padding after shard end
    entry2 = _mk_bytes(33, 11)
    CreateSimpleDB(root, shardsize=PADDING + len(entry2))( [
        CreateSimpleDBSample("first", entry1, True, 1),
        CreateSimpleDBSample("second", entry2, False, 2),
    ])
    db = SimpleDB(root, allow_name_indexing=True).open()
    try:
        r1 = db["first"].data
        r2 = db["second"].data
        assert r1.numel() == len(entry1)
        assert r2.numel() == len(entry2)
        # first/last bytes check
        assert r1[0].item() == entry1[0]
        assert r1[-1].item() == entry1[-1]
        assert r2[0].item() == entry2[0]
        assert r2[-1].item() == entry2[-1]
    finally:
        db.close()


def test_padding_region_is_zeroed(tmp_path: Path) -> None:
    root = tmp_path / "db"
    e1 = _mk_bytes(1000, 1)
    e2 = _mk_bytes(2000, 2)
    CreateSimpleDB(root, shardsize=4096)([
        CreateSimpleDBSample("x", e1, True, 0),
        CreateSimpleDBSample("y", e2, False, 0),
    ])
    data_files = sorted((root / "data").glob("data-*.bin"))
    assert len(data_files) == 1
    shard_size = os.path.getsize(data_files[0])
    assert shard_size % PADDING == 0
    raw = (root / "data" / data_files[0].name).read_bytes()
    # Everything after last record should be zeros
    sizes = pd.read_csv(sorted((root / "size").glob("size-*.csv"))[0])
    last = sizes.iloc[-1]
    pad_start = int(last["offset"]) + int(last["size"])
    assert all(b == 0 for b in raw[pad_start:])


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_nan_and_empty_metadata_variants(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    # family=None, empty string, and missing family column worth testing
    CreateSimpleDB(root)([
        CreateSimpleDBSample("n1", b"\x00\x01", True, 123, None),
        CreateSimpleDBSample("n2", b"\x02\x03", False, 456, ""),
    ])
    # Manually edit meta CSV to inject NaN in family for one row
    meta_csv = sorted((root / "meta").glob("meta-*.csv"))[0]
    df = pd.read_csv(meta_csv)
    df.loc[df["name"] == "n2", "family"] = float("nan")
    df.to_csv(meta_csv, index=False)

    db = SimpleDB(root, reader=reader, allow_name_indexing=True).open()
    try:
        r1 = db["n1"]
        r2 = db["n2"]
        assert r1.family in ("", None)  # None stored → empty string or None on read
        # NaN should come back as None/empty string; no crash
        assert r2.family in ("", None)
    finally:
        db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_duplicate_names_behavior(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("dup", b"\x00", True, 1, None),
        CreateSimpleDBSample("dup", b"\x01", False, 2, None),
    ])
    # If name indexing is enabled, confirm whether it errors or picks last.
    # Choose a policy — here we assert it raises; if you choose "last wins",
    # change to assert specific content.
    with pytest.raises(RuntimeError):
        SimpleDB(root, reader=reader, allow_name_indexing=True).open()
    SimpleDB(root, reader=reader, allow_name_indexing=False).open()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_mismatched_size_and_meta_rows(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("ok", b"\x00\x01", True, 1, None),
    ])
    # Remove the meta row so idxs don't match
    meta_csv = sorted((root / "meta").glob("meta-*.csv"))[0]
    df = pd.read_csv(meta_csv)
    df = df[df["name"] != "ok"]
    df.to_csv(meta_csv, index=False)
    db = SimpleDB(root, reader=reader)
    with pytest.raises(Exception):
        db.open()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_corrupted_offset_raises(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("bad", b"\x00\x01\x02", False, 0, None),
    ])
    # Corrupt size CSV to point past EOF
    size_csv = sorted((root / "size").glob("size-*.csv"))[0]
    df = pd.read_csv(size_csv)
    df.loc[df["name"] == "bad", "offset"] = 10_000_000
    df.to_csv(size_csv, index=False)
    db = SimpleDB(root, reader=reader, allow_name_indexing=True)
    try:
        with pytest.raises(Exception):
            db.open()
    finally:
        db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_zero_length_entry(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("empty", b"", True, 0, None),
    ])
    db = SimpleDB(root, reader=reader, allow_name_indexing=True).open()
    try:
        rec = db["empty"]
        assert isinstance(rec.data, torch.Tensor)
        assert rec.data.dtype == torch.uint8
        assert rec.data.numel() == 0
    finally:
        db.close()


@pytest.mark.parametrize("reader", ["pread", "torch"])
def test_open_twice_behavior(tmp_path: Path, reader: Literal["pread", "torch"]) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("x", b"\x01", True, 1, None),
    ])
    db = SimpleDB(root, reader=reader)
    db.open()
    # Either support idempotent open() or assert a specific error; this expects idempotency.
    db.open()
    db.close()
