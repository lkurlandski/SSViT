"""
Benchmark.
"""

from argparse import ArgumentParser
from collections.abc import Callable
from collections.abc import Sequence
import json
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Optional
import warnings

import psutil
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import BinaryDataset
from src.data import IterableBinaryDataset
from src.data import IterableBinaryDatasetBatchedLoader
from src.data import MapBinaryDataset
from src.data import MapBinaryDatasetBatchedLoader
from src.data import MapBinaryDatasetMemoryMapped
from src.data import ContiguousSampler
from src.data import CollateFn


def benchmark_creation(constructor: Callable[..., BinaryDataset], files: Sequence[Path], labels: Sequence[int], max_length: Optional[int] = None) -> tuple[BinaryDataset, float, float, float]:
    start = time.time()

    dataset = constructor(files, labels, max_length)

    end = time.time()

    mem = psutil.virtual_memory().used

    return dataset, end - start, mem, mem


def benchmark_iteration(dataset: BinaryDataset, batch_size: int = 1, sampler: Optional[Sampler] = None, num_workers: int = 0) -> tuple[float, float, float]:

    dataloader = DataLoader(dataset, batch_size, False, sampler, num_workers=num_workers, collate_fn=CollateFn(), pin_memory=True)

    start = time.time()

    mems = []
    for i, _ in tqdm(enumerate(dataloader), leave=False, total=len(dataset) // batch_size):
        if i % (100 // batch_size) == 0:
            mems.append(psutil.virtual_memory().used)

    end = time.time()

    return end - start, mean(mems), max(mems)


def main() -> None:

    warnings.simplefilter("ignore")

    choices = [
        IterableBinaryDataset.__name__,
        IterableBinaryDatasetBatchedLoader.__name__,
        MapBinaryDataset.__name__,
        MapBinaryDatasetBatchedLoader.__name__,
        MapBinaryDatasetMemoryMapped.__name__,
    ]

    parser = ArgumentParser()
    parser.add_argument("--type", type=str, choices=choices, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input", type=Path, default=Path("./data/binaries"))
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--outfile", type=Path, required=False)
    args = parser.parse_args()

    d: dict[str, dict[str, float]] = {}

    print("Arguments:")
    print(f" Type:        {args.type}")
    print(f" Batch size:  {args.batch_size}")
    print(f" Num workers: {args.num_workers}")

    files: list[Path] = sorted(f for f in args.input.rglob("*") if f.is_file())[0:args.num_samples]
    labels = [-1] * len(files)
    mem = sum(f.stat().st_size for f in files)
    print("Datsets:")
    print(f" Num files:  {len(files)}")
    print(f" Total size: {mem / (1024 * 1024):.2f} MB")
    print(f" Avg size:   {mem / len(files) / (1024 * 1024):.2f} MB")

    d["config"] = {}
    d["config"]["type"] = args.type
    d["config"]["batch_size"] = args.batch_size
    d["config"]["num_workers"] = args.num_workers
    d["config"]["num_samples"] = len(files)
    d["config"]["max_length"] = args.max_length

    constructor = globals()[args.type]
    r = benchmark_creation(constructor, files, labels, args.max_length)
    dataset = r[0]
    stats = r[1:]
    print("Construction:")
    print(f" Time:   {stats[0]:.2f} seconds")
    print(f" Memory: {stats[1] / (1024 * 1024):.2f} MB")

    d["const"] = {}
    d["const"]["time"] = stats[0]
    d["const"]["mem-max"] = stats[2]

    sampler = None
    if isinstance(dataset, MapBinaryDatasetBatchedLoader):
        sampler = ContiguousSampler(len(dataset), dataset.chunk_size, shuffle=False)

    stats = benchmark_iteration(dataset, args.batch_size, sampler, args.num_workers)
    print("Iteration:")
    print(f" Time:         {stats[0]:.2f} seconds")
    print(f" Time/sample:  {stats[0] / len(dataset):.6f} seconds")
    print(f" Memory (avg): {stats[1] / (1024 * 1024):.2f} MB")
    print(f" Memory (max): {stats[2] / (1024 * 1024):.2f} MB")

    d["iter"] = {}
    d["iter"]["time"] = stats[0]
    d["iter"]["mem-avg"] = stats[2]
    d["iter"]["mem-max"] = stats[2]

    if args.outfile:
        with open(args.outfile, "a") as fp:
            fp.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
