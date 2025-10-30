"""
Preprocess large binary dataset (parsing, splitting, sharding, etc.).

Usage
-----
To use, run:
```
python scripts/preprocess.py \
    --outdir OUTDIR \
    --tr_index TR_INDEX \
    --vl_index VL_INDEX \
    --ts_index TS_INDEX
```
The processes data will be placed in `OUTDIR`. Index files, e.g., `TR_INDEX`,
are plaintext files with two columns separated by whitespace. Each row should
contain a hash (sha-256) and label (0 for benign, 1 for malware), respectively.
Each row corresponds to one sample belonging to the respective split (tr, vl, ts).

For more options (compression, etc.), run:
```
python scripts/preprocess.py --help
```
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable
from collections import defaultdict
import hashlib
from itertools import batched
import math
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean
from statistics import median
import sys
import time
from typing import Any
from typing import Optional

import lief
import numpy as np
import polars as pl
from tqdm import tqdm
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import _parse_pe_and_get_size
from src.binanal import CharacteristicGuider
from src.binanal import StructureParser
from src.binanal import HierarchicalLevel
from src.binanal import LEVEL_STRUCTURE_MAP
from src.fileio import read_files_asynch
from src.simpledb import CreateSimpleDB
from src.simpledb import CreateSimpleDBSample
from src.simpledb import SimpleDB


lief.logging.disable()


def preprocess_portable_executable(buffer: bytes) -> dict[str, Any]:
    """
    Extract metadata dictionary from a PE file.
    """
    sha = hashlib.sha256(buffer).hexdigest()

    pe, size = _parse_pe_and_get_size(buffer)

    guider = CharacteristicGuider(pe, size)
    characteristics = guider._get_characteristic_offsets()
    characteristics = {c.name: offs for c, offs in characteristics.items()}

    parser = StructureParser(pe, size)
    partitions: defaultdict[str, dict[str, list[tuple[int, int]]]] = defaultdict(dict)
    for level, structures in LEVEL_STRUCTURE_MAP.items():
        if level == HierarchicalLevel.NONE:
            continue
        for structure in structures:
            ranges = parser(structure)
            partitions[f"{level.name}"][f"{structure.name}"] = ranges
    partitions = dict(partitions)

    return {"sha": sha, "characteristics": characteristics, "partitions": partitions}


def build_dataframe(meta: dict[str, Any]) -> pl.DataFrame:
    """
    Normalize metadata into a long-form Polars DataFrame (~3KB).
    """
    rows: list[dict[str, Optional[str | int]]] = []

    sha: str = meta["sha"]
    characteristics: dict[str, list[tuple[int, int]]] = meta["characteristics"]
    partitions: dict[str, dict[str, list[tuple[int, int]]]] = meta["partitions"]

    for label, ranges in characteristics.items():
        for (start, end) in ranges:
            rows.append(
                {
                    "sha": sha,
                    "group": "characteristics",
                    "level": None,
                    "label": label,
                    "start": int(start),
                    "end": int(end),
                }
            )

    for level, subdict in partitions.items():
        for label, ranges in subdict.items():
            for (start, end) in ranges:
                rows.append(
                    {
                        "sha": sha,
                        "group": "partitions",
                        "level": level,
                        "label": label,
                        "start": int(start),
                        "end": int(end),
                    }
                )

    df = pl.DataFrame(
        rows,
        schema={
            "sha": pl.Utf8,
            "group": pl.Utf8,
            "level": pl.Utf8,
            "label": pl.Utf8,
            "start": pl.UInt32,
            "end": pl.UInt32,
        },
    )

    return df


def preprocess_executables(inputs: Iterable[bytes | Path]) -> pl.DataFrame:
    """
    Preprocess a list of executables into a single Polars DataFrame.
    """
    dfs: list[pl.DataFrame] = []
    for inp in tqdm(inputs, disable=True):
        if not isinstance(inp, bytes):
            with open(inp, "rb") as fp:
                buffer = fp.read()
        else:
            buffer = inp

        try:
            meta = preprocess_portable_executable(buffer)
        except Exception as err:
            sha = hashlib.sha256(buffer).hexdigest()
            print(f"ERROR: failed to preprocess sample {sha} ({err}).")
            continue
        df = build_dataframe(meta)
        dfs.append(df)
    return pl.concat(dfs)


def preprocess_executables_to_parquet(inputs: Iterable[bytes | Path], outfile: Path, compress: bool = True, level: int = 22) -> None:
    """
    Preprocess a list of executables and write dataframe to a Parquet file.
    """
    df = preprocess_executables(inputs)
    df.write_parquet(outfile, compression="zstd" if compress else "uncompressed", compression_level=level)


def read_buffers_from_cloud(shas: list[str]) -> list[bytes]:
    """
    Read binary files from cloud storage given their SHA256 hashes.
    """
    raise NotImplementedError()  # (Scott)


def read_buffers_from_disk(shas: list[str]) -> list[bytes]:
    """
    Read binary files from disk given their SHA256 hashes.
    """
    files = []
    for sha in shas:
        file = Path("./data/sor") / sha[0] / sha[1] / sha
        if file.exists():
            files.append(file)
            continue
        file = Path("./data/ass") / sha[0] / sha[1] / sha
        if file.exists():
            files.append(file)
            continue
        raise FileNotFoundError(f"File for SHA {sha} not found on disk.")
    return read_files_asynch(list(map(str, files)))


def process_one_shard(
        outdir: Path,
        split: str,
        shardidx: int,
        sha_batch: Iterable[str],
        lab_batch: Iterable[int],
        samples_per_shard: int,
        max_length: int,
        compress: bool,
        level: int,
    ) -> None:
    sha_batch: list[str] = list(sha_batch)
    lab_batch: list[int] = list(lab_batch)
    # Preprocess metadata and write to a Parquet.
    outfile = outdir / "meta" / split / f"shard_{shardidx:08d}.parquet"
    buffers = read_buffers_from_disk(sha_batch)     # (Scott)
    # buffers = read_buffers_from_cloud(sha_batch)  # (Scott)
    preprocess_executables_to_parquet(buffers, outfile, compress, level)
    # Truncate and add to the SimpleDB.
    creator = CreateSimpleDB(outdir / "data" / split, shardsize=-1, samples_per_shard=samples_per_shard, exist_ok=True)
    for sha, lab, buf in zip(sha_batch, lab_batch, buffers):
        sample = CreateSimpleDBSample(name=sha, data=buf[0:max_length], malware=lab == 1, timestamp=-1, family=None)
        creator.add(sample)
    # Hacky solution to let multiple creators run in parallel.
    assert creator._cur_meta_df is not None
    assert creator._cur_size_df is not None
    assert creator._cur_data is not None
    creator._cur_meta_df["idx"] = shardidx * samples_per_shard + np.arange(len(sha_batch))
    creator._cur_size_df["idx"] = shardidx * samples_per_shard + np.arange(len(sha_batch))
    creator._cur_meta_df["shard"] = shardidx
    creator._cur_size_df["shard"] = shardidx
    creator._dump_shard_containers(
        creator._cur_data,
        creator._cur_size_df,
        creator._cur_meta_df,
        creator._cur_count,
        shardidx,
        compress=compress,
        level=level,
        threads=0,
        )

def process_one_shard_wrapper(args: tuple) -> None:  # type: ignore[type-arg]
    process_one_shard(*args)


def main() -> None:

    parser = ArgumentParser(description="Preprocess EMBER dataset.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for the shards.")
    parser.add_argument("--tr_index", type=Path, required=True, help="Path to an index file for the tr set.")
    parser.add_argument("--vl_index", type=Path, required=True, help="Path to an index file for the vl set.")
    parser.add_argument("--ts_index", type=Path, required=True, help="Path to an index file for the ts set.")
    parser.add_argument("--samples_per_shard", type=int, default=4096, help="Number of samples per shard.")
    parser.add_argument("--max_length", type=int, default=2**22, help="Truncate files larger than this size (in bytes).")
    parser.add_argument("--num_workers", type=int, default=0, help="Process shards in parallel using this many additional workers.")
    parser.add_argument("--compress", action="store_true", help="Compress data and metadata using Zstandard.")
    parser.add_argument("--level", type=int, default=22, help="Zstandard compression level (1-22).")
    parser.add_argument("--num_check", type=int, default=16, help="Number of samples to check after processing each split.")
    args = parser.parse_args()

    print("preprocess:")
    print(f"  outdir: {args.outdir}")
    print(f"  tr_index: {args.tr_index}")
    print(f"  vl_index: {args.vl_index}")
    print(f"  ts_index: {args.ts_index}")
    print(f"  samples_per_shard: {args.samples_per_shard}")
    print(f"  max_length: {args.max_length}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  compress: {args.compress}")
    print(f"  level: {args.level}")
    print(f"  num_check: {args.num_check}")

    if args.max_length > 2**32:
        raise ValueError("max_length cannot exceed 4 GiB (2^32 bytes).")

    outdir = Path(args.outdir)
    samples_per_shard = int(args.samples_per_shard)
    max_length =int(args.max_length)
    num_workers = int(args.num_workers)
    compress = bool(args.compress)
    level = int(args.level)

    # Create output directories.
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "meta").mkdir(exist_ok=True)
    (outdir / "data").mkdir(exist_ok=True)
    for split in ["ts", "vl", "tr"]:
        (outdir / "meta" / split).mkdir(exist_ok=True)

    # Sort shas and labels lexicographically for each split.
    shas: dict[str, list[str]] = {}
    labs: dict[str, list[int]] = {}
    for split, file in zip(["ts", "vl", "tr"], [args.ts_index, args.vl_index, args.tr_index]):
        _blob = np.loadtxt(file, dtype=str)  # type: ignore[no-untyped-call]
        _shas = _blob[:, 0]
        _labs = _blob[:, 1].astype(int)
        _idx  = np.argsort(_shas)
        shas[split] = _shas[_idx].tolist()
        labs[split] = _labs[_idx].tolist()
    print("Split information:")
    print(f"  ts: {np.unique(labs['ts'], return_counts=True)}")  # type: ignore[no-untyped-call]
    print(f"  vl: {np.unique(labs['vl'], return_counts=True)}")  # type: ignore[no-untyped-call]
    print(f"  tr: {np.unique(labs['tr'], return_counts=True)}")  # type: ignore[no-untyped-call]

    for split in ["ts", "vl", "tr"]:
        # Build the shards for each split.
        print(f"Preprocessing {split} ({num_workers} workers)...")
        total = math.ceil(len(shas[split]) / samples_per_shard)
        desc = f"Progress ({total} shards)..."
        sha_batches = batched(shas[split], n=samples_per_shard)
        lab_batches = batched(labs[split], n=samples_per_shard)
        iterable: list[tuple] = []  # type: ignore[type-arg]
        for shardidx, (sha_batch_, lab_batch_) in enumerate(zip(sha_batches, lab_batches)):
            sha_batch = list(sha_batch_)
            lab_batch = list(lab_batch_)  # type: ignore[call-overload]
            arg = (outdir, split, shardidx, sha_batch, lab_batch, samples_per_shard, max_length, compress, level)
            iterable.append(arg)
        t_start = time.time()
        if num_workers == 0:
            for args_ in tqdm(iterable, total=total, desc=desc):
                process_one_shard_wrapper(args_)
        else:
            with mp.Pool(num_workers) as pool:
                for _ in tqdm(pool.imap_unordered(process_one_shard_wrapper, iterable), total=total, desc=desc):
                    pass
        t_end = time.time()
        t_delta = t_end - t_start
        print(f"  Time: {t_delta:.4f} s/split")
        print(f"  Time: {t_delta / total:.4f} s/shard")
        print(f"  Time: {t_delta / len(shas[split]):.4f} s/sample")
        print(f"  Throughput: {total / t_delta:.4f} shards/s")
        print(f"  Throughput: {len(shas[split]) / t_delta:.4f} samples/s")
        shardfiles = list((outdir / "data" / split / "data").glob("data-*.bin*"))
        shardsizes = [os.path.getsize(f) for f in shardfiles]
        print(f"  Shard Size Total:  {sum(shardsizes) / (1024 ** 2):.0f} MB")
        print(f"  Shard Size Minimum: {min(shardsizes) / (1024 ** 2):.0f} MB")
        print(f"  Shard Size Maximum: {max(shardsizes) / (1024 ** 2):.0f} MB")
        print(f"  Shard Size Mean:    {mean(shardsizes) / (1024 ** 2):.0f} MB")
        print(f"  Shard Size Median:  {median(shardsizes) / (1024 ** 2):.0f} MB")

        # Validate the created shards.
        print(f"Validating {split}...")
        db = SimpleDB(outdir / "data" / split)
        df_size = db.get_size_df()
        df_meta = db.get_meta_df()
        if len(df_size.index) != len(shas[split]):
            raise RuntimeError(f"Mismatch in number of samples for split {split}: expected {len(shas[split])}, got {len(df_size.index)}.")
        print(f"  Total samples: {len(df_size.index)} (Passed)")
        if df_size["size"].max() > max_length:
            raise RuntimeError(f"Some samples in split {split} exceed max_length of {max_length} bytes.")
        print(f"  Max sample size: {df_size['size'].max()} bytes (Passed)")
        num_check = min(args.num_check, len(shas[split]))
        indices = np.random.choice(len(shas[split]), size=num_check, replace=False).tolist()
        sha_batch = [shas[split][i] for i in indices]
        lab_batch = [labs[split][i] for i in indices]
        buffers = read_buffers_from_disk(sha_batch)     # (Scott)
        # buffers = read_buffers_from_cloud(sha_batch)  # (Scott)
        for j in range(num_check):
            idx = indices[j]
            sha = sha_batch[j]
            lab = lab_batch[j]
            buf = buffers[j][0:max_length]
            row_size = df_size.iloc[idx]
            row_meta = df_meta.iloc[idx]
            if row_size["name"] != sha:
                raise RuntimeError(f"Mismatch in sample name for index {idx} in split {split}.")
            if row_meta["malware"] != (lab == 1):
                raise RuntimeError(f"Mismatch in sample label for index {idx} in split {split}.")
            shardfile: Path = db.files_data[row_size["shard"]]
            rawdata = shardfile.read_bytes()
            if compress:
                rawdata = zstd.decompress(rawdata)
            if rawdata[row_size["offset"]:row_size["offset"] + row_size["size"]] != buf:
                raise RuntimeError(f"Mismatch in sample data for index {idx} in split {split}.")
            print(f"  Checked sample {sha} in shard {shardfile.name} (Passed)")


if __name__ == "__main__":
    main()
