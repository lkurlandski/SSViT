"""
Benchmark.
"""

from argparse import ArgumentParser
from collections.abc import Iterable
import json
from pathlib import Path
import random
from statistics import mean
import sys
import time
import warnings

import lief
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import HierarchicalLevel
from src.binanal import EntropyGuider
from src.data import Samples
from src.data import BinaryDataset
from src.data import CollateFn
from src.data import Preprocessor


def main() -> None:

    warnings.simplefilter("ignore")
    lief.logging.disable()

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--level", type=HierarchicalLevel, default=HierarchicalLevel.FINE)
    parser.add_argument("--no_parser", action="store_false", dest="do_parser")
    parser.add_argument("--no_entropy", action="store_false", dest="do_entropy")
    parser.add_argument("--no_characteristics", action="store_false", dest="do_characteristics")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input", type=Path, default=Path("./data/binaries"))
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--outfile", type=Path, required=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d: dict[str, dict[str, float]] = {}

    print("Arguments:")
    print(f" Batch size:      {args.batch_size}")
    print(f" Num workers:     {args.num_workers}")
    print(f" Device:          {device}")
    print(f" Level:           {args.level}")
    print(f" Parser:          {args.do_parser}")
    print(f" Entropy:         {args.do_entropy}")
    print(f" Characteristics: {args.do_characteristics}")

    files: list[Path] = sorted(f for f in args.input.rglob("*") if f.is_file())
    if args.shuffle:
        random.shuffle(files)
    files = files[0:args.num_samples]
    num_samples = len(files)
    labels = [-1] * len(files)
    mem = sum(f.stat().st_size for f in files)
    print("Datsets:")
    print(f" Num files:  {num_samples}")
    print(f" Total size: {mem / (1024 * 1024):.2f} MB")
    print(f" Avg size:   {mem / len(files) / (1024 * 1024):.2f} MB")

    d["config"] = {}
    d["config"]["batch_size"] = args.batch_size
    d["config"]["num_workers"] = args.num_workers
    d["config"]["num_samples"] = num_samples
    d["config"]["device"] = str(device)

    # Compile the numba
    EntropyGuider(np.zeros(1024, np.uint8))(dtype=np.float64)
    EntropyGuider(np.zeros(1024, np.uint8))(dtype=np.float32)
    # EntropyGuider(np.zeros(1, np.uint8))(dtype=np.float16)

    preprocessor = Preprocessor(do_parser=args.do_parser, do_entropy=args.do_entropy, do_characteristics=args.do_characteristics, level=args.level)
    dataset = BinaryDataset(files, labels, preprocessor=preprocessor)
    collate_fn = CollateFn(pin_memory=True)
    dataloader = DataLoader(dataset, args.batch_size, True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False)
    iterable: Iterable[tuple[int, Samples]] = tqdm(enumerate(dataloader), leave=False, total=len(dataset) // args.batch_size)

    start = time.time()

    mems = []
    for i, batch in iterable:
        files = batch.file
        names = batch.name
        labels = batch.label.to(device, non_blocking=True)
        inputs = batch.inputs.to(device, non_blocking=True)
        guides = batch.guides.to(device, non_blocking=True)
        structure = batch.structure.to(device, non_blocking=True)
        mems.append(psutil.virtual_memory().used)
        # del files, names, labels, inputs, guides, structure
        # if (i * args.batch_size) % 256 == 0:
        #     torch.cuda.empty_cache()

    end = time.time()

    stats = (end - start, (end - start) / num_samples, mean(mems), max(mems))

    print("Iteration:")
    print(f" Time (tot):   {stats[0]:.2f} seconds")
    print(f" Time (avg):   {stats[1]:.4f} seconds")
    print(f" Memory (avg): {stats[2] / (1024 * 1024):.2f} MB")
    print(f" Memory (max): {stats[3] / (1024 * 1024):.2f} MB")

    d["stats"] = {}
    d["stats"]["time-tot"] = stats[0]
    d["stats"]["time-avg"] = stats[1]
    d["stats"]["mem-avg"] = stats[2]
    d["stats"]["mem-max"] = stats[3]

    if args.outfile:
        with open(args.outfile, "a") as fp:
            fp.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
