"""
Preprocess large binary dataset (parsing, splitting, sharding, etc.).
"""

from __future__ import annotations
from argparse import ArgumentParser
from collections.abc import Iterable
from collections import defaultdict
import hashlib
from itertools import batched
from itertools import islice
import math
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import sys
from typing import Any
from typing import Optional
import warnings

import lief
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import _parse_pe_and_get_size
from src.binanal import CharacteristicGuider
from src.binanal import StructureParser
from src.binanal import HierarchicalLevel
from src.binanal import LEVEL_STRUCTURE_MAP
from src.fileio import read_files_asynch
from src.simpledb import CreateSimpleDB
from src.simpledb import CreateSimpleDBSample


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

        meta = preprocess_portable_executable(buffer)
        df = build_dataframe(meta)
        dfs.append(df)
    return pl.concat(dfs)


def preprocess_executables_to_parquet(inputs: Iterable[bytes | Path], outfile: Path) -> None:
    """
    Preprocess a list of executables and write dataframe to a Parquet file.
    """
    df = preprocess_executables(inputs)
    df.write_parquet(outfile, compression="zstd", compression_level=22)


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


def main() -> None:

    parser = ArgumentParser(description="Preprocess EMBER dataset.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for the shards.")
    parser.add_argument("--tr_index", type=Path, required=True, help="Path to an index file for the tr set.")
    parser.add_argument("--vl_index", type=Path, required=True, help="Path to an index file for the vl set.")
    parser.add_argument("--ts_index", type=Path, required=True, help="Path to an index file for the ts set.")
    parser.add_argument("--samples_per_shard", type=int, default=1024, help="Number of samples per shard.")
    parser.add_argument("--max_length", type=int, default=2**22, help="Truncate files larger than this size (in bytes).")
    parser.add_argument("--num_workers", type=int, default=0, help="Process shards in parallel using this many additional workers.")
    args = parser.parse_args()

    # Create output directories.
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "meta").mkdir(exist_ok=True)
    (outdir / "data").mkdir(exist_ok=True)
    for split in ["ts", "vl", "tr"]:
        (outdir / "meta" / split).mkdir(exist_ok=True)
        # (outdir / "data" / split).mkdir(exist_ok=True)

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
    print(f"  ts: {np.unique(labs['ts'], return_counts=True)}")
    print(f"  vl: {np.unique(labs['vl'], return_counts=True)}")
    print(f"  tr: {np.unique(labs['tr'], return_counts=True)}")

    # Build the shards for each split.
    for split in ["ts", "vl", "tr"]:
        total = math.ceil(len(shas[split]) / args.samples_per_shard)
        desc = f"Preprocessing {split} ({total} shards)..."
        sha_batches = batched(shas[split], n=int(args.samples_per_shard))
        lab_batches = batched(labs[split], n=int(args.samples_per_shard))
        iterable = tqdm(zip(sha_batches, lab_batches), total=total, desc=desc)

        for shardidx, (sha_batch_, lab_batch_) in enumerate(iterable):
            creator = CreateSimpleDB(outdir / "data" / split, shardsize=-1, samples_per_shard=args.samples_per_shard, exist_ok=True)
            # Stupid type ignore to satisfy mypy.
            sha_batch: list[str] = list(sha_batch_)
            lab_batch: list[int] = list(lab_batch_)  # type: ignore
            # Preprocess metadata and write to a Parquet.
            outfile = outdir / "meta" / split / f"shard_{shardidx:08d}.parquet"
            buffers = read_buffers_from_disk(sha_batch)     # (Scott)
            # buffers = read_buffers_from_cloud(sha_batch)  # (Scott)
            preprocess_executables_to_parquet(buffers, outfile)
            # Truncate and add to the SimpleDB.
            for sha, lab, buf in zip(sha_batch, lab_batch, buffers):
                sample = CreateSimpleDBSample(name=sha, data=buf, malware=lab == 1, timestamp=-1, family=None)
                creator.add(sample)
            creator._cur_meta_df["idx"] = shardidx * args.samples_per_shard + np.arange(len(sha_batch))
            creator._cur_size_df["idx"] = shardidx * args.samples_per_shard + np.arange(len(sha_batch))
            creator._cur_meta_df["shard"] = shardidx
            creator._cur_size_df["shard"] = shardidx
            creator._dump_shard_containers()
            del creator


if __name__ == "__main__":
    main()
