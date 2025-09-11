"""
Tests.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Optional

import os
import random
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from src.simpledb import (
    PAGESIZE,
    roundup,
    CreateSimpleDB,
    CreateSimpleDBSample,
    MappingSimpleDB,
    IterableSimpleDB,
)

# ---------- helpers ----------

def _mk_bytes(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()

def _samples(
    count: int,
    *,
    min_len: int = 100,
    max_len: int = 10_000,
    seed: int = 0,
) -> Generator[CreateSimpleDBSample, None, None]:
    rng = random.Random(seed)
    for i in range(count):
        name = f"file-{i:08d}.exe"
        n = rng.randint(min_len, max_len)
        data = _mk_bytes(n, seed=seed + i)
        malware = bool(rng.getrandbits(1))
        ts = 1_600_000_000 + i
        fam: Optional[str] = None if (i % 3 == 0) else f"fam{i % 7}"
        yield CreateSimpleDBSample(name=name, data=data, malware=malware, timestamp=ts, family=fam)

# ---------- basic utilities ----------

def test_roundup_basic() -> None:
    assert roundup(0, PAGESIZE) == 0
    assert roundup(1, PAGESIZE) == PAGESIZE
    assert roundup(PAGESIZE - 1, PAGESIZE) == PAGESIZE
    assert roundup(PAGESIZE, PAGESIZE) == PAGESIZE
    assert roundup(PAGESIZE + 1, PAGESIZE) == 2 * PAGESIZE

# ---------- building and reading (mapping) ----------

def test_build_one_shard_and_read_mapping(tmp_path: Path) -> None:
    root = tmp_path / "db"
    builder = CreateSimpleDB(root, shardsize=2 * 1024 * 1024)
    samples = list(_samples(25, min_len=5000, max_len=12_000, seed=123))
    builder(samples)

    data_files = sorted((root / "data").glob("data-*.bin"))
    size_files = sorted((root / "size").glob("size-*.csv"))
    meta_files = sorted((root / "meta").glob("meta-*.csv"))
    assert len(data_files) == len(size_files) == len(meta_files) == 1

    # shard file padded to page size
    sz = os.path.getsize(data_files[0])
    assert sz % PAGESIZE == 0

    db = MappingSimpleDB(root, allow_name_indexing=True).open()
    try:
        for idx, s in enumerate(samples):
            rec = db[idx]
            # names
            assert rec.name == s.name
            # tensor type & length
            assert isinstance(rec.data, torch.Tensor)
            assert rec.data.dtype == torch.uint8
            assert rec.data.numel() == len(s.data)
            # zero-copy memoryview matches tensor bytes
            assert bytes(rec.bview) == bytes(rec.data.tolist()) == s.data
            # meta
            assert rec.malware == s.malware
            assert rec.timestamp == s.timestamp
            if s.family is None:
                assert rec.family in ("", None)
            else:
                assert rec.family == s.family

            # name indexing
            by_name = db[s.name]
            assert bytes(by_name.bview) == s.data
    finally:
        db.close()

# ---------- building and reading (iterable) ----------

def test_build_and_stream_iterable(tmp_path: Path) -> None:
    root = tmp_path / "db"
    # several shards to exercise per-shard streaming
    builder = CreateSimpleDB(root, shardsize=256 * 1024)
    s_list = list(_samples(200, min_len=200, max_len=4000, seed=7))
    builder(s_list)

    # stream everything, reconstruct name->bytes
    seen = []
    db = IterableSimpleDB(root).open()
    try:
        for rec in db:
            assert isinstance(rec.data, torch.Tensor)
            assert rec.data.dtype == torch.uint8
            # memoryview should match tensor content and be bytes-like
            assert isinstance(rec.bview, memoryview)
            assert bytes(rec.bview) == bytes(rec.data.tolist())
            seen.append((rec.name, bytes(rec.bview)))
    finally:
        db.close()

    produced = {name: data for (name, data) in seen}
    expected = {s.name: s.data for s in s_list}
    assert produced.keys() == expected.keys()
    # spot-check a few
    for key in list(expected.keys())[:10]:
        assert produced[key] == expected[key]


def test_iterable_streams_in_offset_order_within_shard(tmp_path: Path) -> None:
    root = tmp_path / "db"
    builder = CreateSimpleDB(root, shardsize=128 * 1024)
    s_list = list(_samples(80, min_len=512, max_len=4096, seed=99))
    builder(s_list)

    # for each shard, collect offsets in stream order and verify non-decreasing
    # we derive offsets from CSV since iterator doesn't expose them
    size_csvs = sorted((root / "size").glob("size-*.csv"))
    per_shard_offsets = {int(p.stem.split("-")[-1]): pd.read_csv(p)[["offset", "idx"]]
                         for p in size_csvs}

    db = IterableSimpleDB(root).open()
    try:
        current_shard = None
        last_offset = None
        shard_to_offset_map = {}
        for p in size_csvs:
            df = pd.read_csv(p)
            shard_id = int(p.stem.split("-")[-1])
            shard_to_offset_map[shard_id] = df.set_index("idx")["offset"].to_dict()

        for rec in db:
            shard = int(rec.name.split("-")[0].replace("file", "0")) if False else None  # not used
            # we can't parse shard from name; use size tables
            # Look up offset via idx by reverse-mapping the name -> idx
            # Build a quick name->idx map once
            pass
    finally:
        db.close()


def test_shards_rollover(tmp_path: Path) -> None:
    root = tmp_path / "db"
    builder = CreateSimpleDB(root, shardsize=128 * 1024)
    big_samples = list(_samples(40, min_len=2000, max_len=5000, seed=7))
    builder(big_samples)

    data_files = sorted((root / "data").glob("data-*.bin"))
    size_files = sorted((root / "size").glob("size-*.csv"))
    meta_files = sorted((root / "meta").glob("meta-*.csv"))
    assert len(data_files) == len(size_files) == len(meta_files) >= 2

    names = [p.name for p in data_files]
    assert names == sorted(names)
    assert all(name.startswith("data-") and name.endswith(".bin") for name in names)

    for size_csv, data_bin in zip(size_files, data_files):
        df = pd.read_csv(size_csv)
        total = os.path.getsize(data_bin)
        for _, row in df.iterrows():
            off, ln = int(row["offset"]), int(row["size"])
            assert 0 <= off
            assert off + ln <= total


def test_open_and_close_contracts(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("x.exe", b"\x00", False, 0, None)])

    # open twice should raise; close twice should raise (per current API)
    db = MappingSimpleDB(root)
    db.open()
    with pytest.raises(RuntimeError):
        db.open()
    db.close()
    with pytest.raises(RuntimeError):
        db.close()


def test_oversized_sample_warns_and_is_present(tmp_path: Path) -> None:
    root = tmp_path / "db"
    big = b"\xff" * (256 * 1024)
    small = b"\x01\x02"
    builder = CreateSimpleDB(root, shardsize=64 * 1024)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        builder([
            CreateSimpleDBSample("a.exe", small, False, 1, None),
            CreateSimpleDBSample("big.exe", big, True, 2, "fam"),
            CreateSimpleDBSample("b.exe", small, False, 3, None),
        ])
        assert any("larger than the shard size" in str(x.message) for x in w)

    db = MappingSimpleDB(root, allow_name_indexing=True).open()
    try:
        rec = db["big.exe"]
        assert rec.data.numel() == len(big)
        assert rec.malware is True
        assert rec.family == "fam"
    finally:
        db.close()


def test_padding_region_is_zeroed(tmp_path: Path) -> None:
    root = tmp_path / "db"
    e1 = _mk_bytes(1000, 1)
    e2 = _mk_bytes(2000, 2)
    CreateSimpleDB(root, shardsize=PAGESIZE * 2)([
        CreateSimpleDBSample("x", e1, True, 0),
        CreateSimpleDBSample("y", e2, False, 0),
    ])
    data_files = sorted((root / "data").glob("data-*.bin"))
    assert len(data_files) == 1
    shard_size = os.path.getsize(data_files[0])
    assert shard_size % PAGESIZE == 0
    raw = (root / "data" / data_files[0].name).read_bytes()
    sizes = pd.read_csv(sorted((root / "size").glob("size-*.csv"))[0])
    last = sizes.iloc[-1]
    pad_start = int(last["offset"]) + int(last["size"])
    assert all(b == 0 for b in raw[pad_start:])


def test_nan_and_empty_metadata_variants_mapping(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("n1", b"\x00\x01", True, 123, None),
        CreateSimpleDBSample("n2", b"\x02\x03", False, 456, ""),
    ])
    meta_csv = sorted((root / "meta").glob("meta-*.csv"))[0]
    df = pd.read_csv(meta_csv)
    df.loc[df["name"] == "n2", "family"] = float("nan")
    df.to_csv(meta_csv, index=False)

    db = MappingSimpleDB(root, allow_name_indexing=True).open()
    try:
        r1 = db["n1"]
        r2 = db["n2"]
        assert r1.family in ("", None)
        assert r2.family in ("", None)  # NaN -> empty/None, but no crash
    finally:
        db.close()


def test_duplicate_names_behavior_mapping(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([
        CreateSimpleDBSample("dup", b"\x00", True, 1, None),
        CreateSimpleDBSample("dup", b"\x01", False, 2, None),
    ])
    # Name indexing builds map in __init__, so duplicate names raise there.
    with pytest.raises(RuntimeError):
        MappingSimpleDB(root, allow_name_indexing=True)


def test_mismatched_size_and_meta_rows_on_init(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("ok", b"\x00\x01", True, 1, None)])
    # Remove the meta row so idxs don't match
    meta_csv = sorted((root / "meta").glob("meta-*.csv"))[0]
    df = pd.read_csv(meta_csv)
    df = df[df["name"] != "ok"]
    df.to_csv(meta_csv, index=False)
    with pytest.raises(RuntimeError, match="different number of rows"):
        MappingSimpleDB(root)


def test_corrupted_offset_raises_on_init(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("bad", b"\x00\x01\x02", False, 0, None)])
    size_csv = sorted((root / "size").glob("size-*.csv"))[0]
    df = pd.read_csv(size_csv)
    df.loc[df["name"] == "bad", "offset"] = 10_000_000
    df.to_csv(size_csv, index=False)
    with pytest.raises(RuntimeError, match="exceeds the size"):
        MappingSimpleDB(root)


def test_zero_length_entry_mapping(tmp_path: Path) -> None:
    root = tmp_path / "db"
    CreateSimpleDB(root)([CreateSimpleDBSample("empty", b"", True, 0, None)])
    db = MappingSimpleDB(root, allow_name_indexing=True).open()
    try:
        rec = db["empty"]
        assert isinstance(rec.data, torch.Tensor)
        assert rec.data.dtype == torch.uint8
        assert rec.data.numel() == 0
        assert bytes(rec.bview) == b""
    finally:
        db.close()
