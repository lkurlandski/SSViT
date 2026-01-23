"""
Tests.
"""

from collections.abc import Generator
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Literal
from typing import Optional
import threading
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from src.simpledb import PAGESIZE
from src.simpledb import roundup
from src.simpledb import Sample
from src.simpledb import SimpleDB
from src.simpledb import CreateSimpleDB
from src.simpledb import SimpleDBReader
from src.simpledb import SimpleDBIterator
from src.simpledb import MetadataDB


def _make_bytes(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def _create_samples(
    count: int,
    *,
    min_len: int = 100,
    max_len: int = 10_000,
    seed: int = 0,
) -> Generator[Sample, None, None]:
    rng = random.Random(seed)
    for i in range(count):
        name = f"file-{i:08d}.exe"
        n = rng.randint(min_len, max_len)
        data = _make_bytes(n, seed=seed + i)
        malware = bool(rng.getrandbits(1))
        ts = 1_600_000_000 + i
        fam = "" if (i % 3 == 0) else f"fam{i % 7}"
        yield Sample(name=name, data=data, malware=malware, timestamp=ts, family=fam)


@pytest.mark.parametrize("num_samples", [1, 1, 2, 3, 10, 100, 1000])
@pytest.mark.parametrize("shardsize", [-1, 0, 2**10, 2**12, 2**14, 2**16])
@pytest.mark.parametrize("samples_per_shard", [-1, 0, 1, 2, 5, 10, 100, 1000, 2000])
def test(tmp_path: Path, num_samples: int, shardsize: int, samples_per_shard: int) -> None:
    """
    This essentially tests the SimpleDB, the SimpleDBCreator, and the SimpleDBReader.
    """

    root = tmp_path / "db"

    # Generate synthetic samples.
    samples = list(_create_samples(num_samples))

    # Build a SimpleDB.
    if (shardsize <= 0) == (samples_per_shard <= 0):
        with pytest.raises(ValueError):
            CreateSimpleDB(root, shardsize=shardsize, samples_per_shard=samples_per_shard)
        return
    builder = CreateSimpleDB(root, shardsize=shardsize, samples_per_shard=samples_per_shard)
    builder(samples)

    # Verify shard files created as expected.
    data_files = sorted((root / "data").glob("data-*.bin"))
    size_files = sorted((root / "size").glob("size-*.csv"))
    meta_files = sorted((root / "meta").glob("meta-*.csv"))
    if samples_per_shard > 0:
        num = math.ceil(num_samples / samples_per_shard)
    elif shardsize > 0:
        num = 1
        running = 0
        for sample in samples:
            size = roundup(len(sample.data), PAGESIZE)
            if running + size > shardsize:
                if running != 0:
                    num += 1
                    running = 0
            running += size
        num = max(1, num)
    else:
        raise ValueError("Either shardsize or samples_per_shard must be positive.")
    assert len(data_files) == num
    assert len(size_files) == num
    assert len(meta_files) == num

    # Read the data back and verify integrity.
    db = SimpleDB(root)
    reader = SimpleDBReader(db)
    for idx, sample in enumerate(samples):
        rec = reader.get(idx)
        assert rec.name == sample.name
        assert rec.data == sample.data
        assert rec.malware == sample.malware
        assert rec.timestamp == sample.timestamp
        assert rec.family == sample.family

        rec = reader.get(sample.name)
        assert rec.name == sample.name
        assert rec.data == sample.data
        assert rec.malware == sample.malware
        assert rec.timestamp == sample.timestamp
        assert rec.family == sample.family
