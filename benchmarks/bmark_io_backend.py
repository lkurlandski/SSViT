"""
Benchmark different I/O backends.
"""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys
import time

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simpledb import SimpleDB
from src.simpledb import RandomMappingSimpleDB
from src.simpledb import ChunkedMappingSimpleDB
from src.simpledb import IterableSimpleDB


NBYTES = 1 * (2 ** 30)


parser = ArgumentParser()
parser.add_argument("--backend", type=str, choices=["db", "fp"], required=True)
parser.add_argument("--contiguous", action="store_true")
args = parser.parse_args()


if args.backend == "db":
    db = SimpleDB(Path("./datadb"), reader="pread", pread_cache=args.contiguous).open()
    indices = list(range(8527))
    if not args.contiguous:
        random.shuffle(indices)

    times = []
    n_bytes = 0
    for i in tqdm(indices):
        t_i = time.time()
        sample = db[i]
        t_f = time.time()
        n_bytes += sample.data.nbytes
        times.append(t_f - t_i)
        if n_bytes >= NBYTES:
            break
    db.close()


if args.backend == "fp":
    files  = []
    # files += list(filter(lambda f: f.is_file(), Path("./data/sor/").rglob("*")))
    files += list(filter(lambda f: f.is_file(), Path("./data/ass/").rglob("*")))[0:8527]
    random.shuffle(files)
    if args.contiguous:
        files.sort()

    times = []
    n_bytes = 0
    for f in tqdm(files):
        t_i = time.time()
        with open(f, "rb") as fp:
            data = fp.read()
        t_f = time.time()
        n_bytes += len(data)
        times.append(t_f - t_i)
        if n_bytes >= NBYTES:
            break


print(f"Total samples: {len(times)}")
print(f"Total bytes: {n_bytes}")
print(f"Total time: {sum(times):.6f}s")
print(f"Average time per sample: {sum(times) / len(times):.6f}s")
print(f"Average throughput: {n_bytes / sum(times) / 1e6:.2f} MB/s")
