"""
Split SimpleDB datasets into train/validation/test subsets.
"""

from argparse import ArgumentParser
from hashlib import md5
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simpledb import SimpleDB
from src.simpledb import split_simple_db
from src.split import tr_vl_ts_split
from src.utils import seed_everything


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dir_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shardsize", type=int, default=2**30)
    parser.add_argument("--tr_size", type=float, default=0.8)
    parser.add_argument("--vl_size", type=float, default=0.1)
    parser.add_argument("--ts_size", type=float, default=0.1)
    parser.add_argument("--no_temporal", action="store_true")
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    db = SimpleDB(args.dir_root)
    print("Shardwise stats for the source DB:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(db.compute_shardwise_stats())

    idx = db.meta_df["idx"].to_numpy().astype(np.int64)
    labels = db.meta_df["malware"].to_numpy().astype(np.int64)
    timestamps = db.meta_df["timestamp"].to_numpy().astype(np.int64)

    if not args.no_temporal and np.any(timestamps < 0):
        raise ValueError("`timestamps` contain negative values; cannot perform a temporal split.")

    tr_idx, vl_idx, ts_idx = tr_vl_ts_split(
        idx,
        tr_size=args.tr_size,
        vl_size=args.vl_size,
        ts_size=args.ts_size,
        labels=labels,
        ratios=np.array([0.5, 0.5]),
        timestamps=None if args.no_temporal else timestamps,
        shuffle=not args.no_shuffle,
        random_state=args.seed,
        temporal_mode="balanced",
    )

    print(f"idx ({len(idx)}): distribution={np.unique(labels,               return_counts=True)} hash={md5(idx.tobytes()).hexdigest()}")     # type: ignore[no-untyped-call]
    print(f"tr_idx ({len(tr_idx)}): distribution={np.unique(labels[tr_idx], return_counts=True)} hash={md5(tr_idx.tobytes()).hexdigest()}")  # type: ignore[no-untyped-call]
    print(f"vl_idx ({len(vl_idx)}): distribution={np.unique(labels[vl_idx], return_counts=True)} hash={md5(vl_idx.tobytes()).hexdigest()}")  # type: ignore[no-untyped-call]
    print(f"ts_idx ({len(ts_idx)}): distribution={np.unique(labels[ts_idx], return_counts=True)} hash={md5(ts_idx.tobytes()).hexdigest()}")  # type: ignore[no-untyped-call]

    dirs_out = [
        args.dir_root.parent / f"{args.dir_root.name}_tr",
        args.dir_root.parent / f"{args.dir_root.name}_vl",
        args.dir_root.parent / f"{args.dir_root.name}_ts",
    ]
    if any(dir_out.exists() for dir_out in dirs_out):
        raise FileExistsError(f"One of the output directories already exists: {dirs_out}")

    split_simple_db(
        args.dir_root,
        dirs_out=dirs_out,
        indices=[tr_idx, vl_idx, ts_idx],
        shardsize=args.shardsize,
    )

    for dir_out in dirs_out:
        db_out = SimpleDB(dir_out)
        print(f"Shardwise stats for the output DB at {dir_out}:")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(db_out.compute_shardwise_stats())


if __name__ == "__main__":
    main()
