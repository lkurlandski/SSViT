"""
Compare the performance of SimpleDB's.

Experiment: args.subtype='chk'
Milli-Seconds/Sample: 2.845
Samples/Second: 351.5
MiB/Second: 333.4

Experiment: args.subtype='rnd' args.backend='mmap'
Milli-Seconds/Sample: 5.301
Samples/Second: 188.7
MiB/Second: 178.9

Experiment: args.subtype='rnd' args.backend='pread'
Milli-Seconds/Sample: 21.425
Samples/Second: 46.7
MiB/Second: 44.3

Experiment: args.subtype='itr' args.pread_block_bytes=1048576 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 2.884
Samples/Second: 346.8
MiB/Second: 328.9

Experiment: args.subtype='itr' args.pread_block_bytes=2097152 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 2.865
Samples/Second: 349.0
MiB/Second: 331.0

Experiment: args.subtype='itr' args.pread_block_bytes=4194304 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 3.217
Samples/Second: 310.9
MiB/Second: 294.8

Experiment: args.subtype='itr' args.pread_block_bytes=8388608 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 3.447
Samples/Second: 290.1
MiB/Second: 275.2

Experiment: args.subtype='itr' args.pread_block_bytes=16777216 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 3.822
Samples/Second: 261.6
MiB/Second: 248.2

Experiment: args.subtype='itr' args.pread_block_bytes=33554432 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 3.828
Samples/Second: 261.3
MiB/Second: 247.8

Experiment: args.subtype='itr' args.pread_block_bytes=67108864 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 5.519
Samples/Second: 181.2
MiB/Second: 171.8

Experiment: args.subtype='itr' args.pread_block_bytes=67108864 args.merge_slack_bytes=0 args.prefetch_next_window=0 args.use_readahead=0
Milli-Seconds/Sample: 4.620
Samples/Second: 216.5
MiB/Second: 205.3

Experiment: args.subtype='itr' args.pread_block_bytes=67108864 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 4.907
Samples/Second: 203.8
MiB/Second: 193.3

Experiment: args.subtype='itr' args.pread_block_bytes=67108864 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=0
Milli-Seconds/Sample: 5.403
Samples/Second: 185.1
MiB/Second: 175.5

Experiment: args.subtype='itr' args.pread_block_bytes=67108864 args.merge_slack_bytes=0 args.prefetch_next_window=1 args.use_readahead=1
Milli-Seconds/Sample: 5.043
Samples/Second: 198.3
MiB/Second: 188.1
"""

from argparse import ArgumentParser
from random import shuffle
from pathlib import Path
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simpledb import SimpleDB
from src.simpledb import RandomMappingSimpleDB
from src.simpledb import ChunkedMappingSimpleDB
from src.simpledb import IterableSimpleDB


parser = ArgumentParser()
parser.add_argument("--dir_root", type=Path, required=True)
parser.add_argument("--subtype", type=str, choices=["rnd", "chk", "itr"], required=True)
parser.add_argument("--backend", type=str, choices=["mmap", "pread"], default="mmap")
parser.add_argument("--pread_block_bytes", type=int, default=8 * 2**20)
parser.add_argument("--merge_slack_bytes", type=int, default=0)
parser.add_argument("--prefetch_next_window", type=int, default=1)
parser.add_argument("--use_readahead", type=int, default=0)
args = parser.parse_args()


PAGESIZE = 4096


def touch_pages_u8(t: torch.Tensor) -> None:
    _ = t[::PAGESIZE].sum().item()


db = SimpleDB(args.dir_root)
total = len(db.size_df.index)
indices = list(range(total))
nbytes = 0


if args.subtype == "itr":
    print(f"Experiment: {args.subtype=} {args.pread_block_bytes=} {args.merge_slack_bytes=} {args.prefetch_next_window=} {args.use_readahead=}")
    db = IterableSimpleDB(args.dir_root, pread_block_bytes=args.pread_block_bytes, merge_slack_bytes=args.merge_slack_bytes, prefetch_next_window=args.prefetch_next_window == 1, prefetch_max_bytes=1 * 2 ** 20, use_readahead=args.use_readahead == 1)
    with db as opened_db:
        t_i = time.time()
        for sample in tqdm(opened_db, total=total):
            t = sample.data
            nbytes += t.nbytes
            touch_pages_u8(t)
        t_f = time.time()
else:
    if args.subtype == "rnd":
        print(f"Experiment: {args.subtype=} {args.backend=}")
        db = RandomMappingSimpleDB(args.dir_root, num_open=len(db.files_data), backend=args.backend)
        shuffle(indices)
    elif args.subtype == "chk":
        print(f"Experiment: {args.subtype=}")
        db = ChunkedMappingSimpleDB(args.dir_root)
        indices.sort()
    with db.open() as opened_db:
        t_i = time.time()
        for idx in tqdm(indices):
            t = opened_db[idx].data
            nbytes += t.nbytes
            touch_pages_u8(t)
        t_f = time.time()


print(f"Milli-Seconds/Sample: {1000 * (t_f - t_i) / total:.3f}")
print(f"Samples/Second: {total / (t_f - t_i):.1f}")
print(f"MiB/Second: {nbytes / (t_f - t_i) / (2**20):.1f}")
