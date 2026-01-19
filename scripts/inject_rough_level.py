"""
Injects the "ROUGH" hierarchical level into the metadata.
"""

from argparse import ArgumentParser
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Optional

import polars as pl
from tqdm import tqdm


def transform(df: pl.DataFrame) -> pl.DataFrame:

    dfs = []
    for sub in df.group_by("sha", maintain_order=True):
        sha, df_ = sub
        # Copy the MIDDLE entries and modify them to create ROUGH entries
        rough = df_.filter((pl.col("group") == "partitions") & (pl.col("level") == "MIDDLE"))
        rough = rough.with_columns(level=pl.col("level").replace("MIDDLE", "ROUGH"))
        rough = rough.with_columns(label=pl.col("label").replace("CODE", "OTHRSEC"))
        rough = rough.with_columns(label=pl.col("label").replace("DATA", "OTHRSEC"))
        rough = rough.with_columns(label=pl.col("label").replace("DNETCODE", "DNETSEC"))
        rough = rough.with_columns(label=pl.col("label").replace("DNETDATA", "DNETSEC"))
        # Get the other sections of the DataFrame
        chr = df_.filter((pl.col("group") == "characteristics"))
        coa = df_.filter((pl.col("group") == "partitions") & (pl.col("level") == "COARSE"))
        mid = df_.filter((pl.col("group") == "partitions") & (pl.col("level") == "MIDDLE"))
        fin = df_.filter((pl.col("group") == "partitions") & (pl.col("level") == "FINE"))
        # Concatenate all parts and add to the list
        new = pl.concat([chr, coa, rough, mid, fin], how="vertical")
        dfs.append(new)

    new = pl.concat(dfs, how="vertical")

    return new


def run(infile: Path, outfile: Optional[Path] = None) -> None:
    tqdm.write((f"[{os.getpid()}] Processing {infile} -> {outfile}"))

    df = pl.read_parquet(infile)

    df = transform(df)

    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(outfile.as_posix(), compression="zstd", compression_level=22)


def _run(args: tuple[Path, Path]) -> None:
    infile, outfile = args
    run(infile, outfile)


def main() -> None:

    parser = ArgumentParser()
    parser.add_argument("--indir", type=Path, default=Path("./data/meta/"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--num_files", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.outdir is None:
        root = Path("./tmp/inject/")
        if not args.quiet:
            print(f"No outdir was specified. Outfiles will be rooted at {root.as_posix()}.")
        args.outdir = root / args.indir.name

    args.outdir.mkdir(parents=True, exist_ok=True)

    infiles = sorted(args.indir.rglob("*.parquet"))[0:args.num_files]
    outfiles = [args.outdir / f.relative_to(args.indir) for f in infiles]

    if not args.quiet:
        print(f"Found {len(infiles)} input files in {args.indir.as_posix()}.")

    iterable = zip(infiles, outfiles)

    if args.num_workers < 2:
        for infile, outfile in zip(infiles, outfiles):
            run(infile, outfile)
    else:
        with mp.Pool(args.num_workers) as pool:
            list(tqdm(pool.imap(_run, iterable), total=len(infiles), desc="Processing...", disable=args.quiet))


if __name__ == "__main__":
    main()
