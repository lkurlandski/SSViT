"""
Quickly analyze an experiment.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--resfile", type=Path)
parser.add_argument("--logfile", type=Path)
parser.add_argument("--jobid", type=str)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

ROOT = Path("/home/lk3591/Documents/code/SSViT")
LOGS = ROOT / "logs"
BULK = Path("/shared/rc/admalware/Documents/code/SSViT")

if args.verbose:
    print(f"{args.resfile=}")
    print(f"{args.logfile=}")
    print(f"{args.jobid=}")

# Get the logfile, if not provided.
if args.jobid is not None:
    files = list(LOGS.glob(f"*{args.jobid}*"))
    if len(files) == 0:
        raise FileNotFoundError(f"Could not find a log file in {LOGS.as_posix()} with JobId {args.jobid}")
    if len(files) > 1:
        raise RuntimeError()
    args.logfile = files[0]
    if args.verbose:
        print(f"{args.logfile=}")

# Get the resfile, if not provided.
if args.logfile is not None:
    with open(args.logfile) as fp:
        for line in fp:
            if "outdir" not in line:
                continue
            s = line.index("'")
            e = line.index("'", s + 1)
            outdir = Path(line[s + 1:e])
            break
        else:
            raise RuntimeError(f"outdir could not be extracted from {args.logfile}")
    if args.verbose:
        print(f"{outdir=}")

    if outdir.exists():
        resfile = outdir / "results.jsonl"
    elif (ROOT / outdir).exists():
        resfile = ROOT / outdir / "results.jsonl"
    elif (BULK / outdir).exists():
        resfile = BULK / outdir / "results.jsonl"
    else:
        raise FileNotFoundError(outdir)
    args.resfile = resfile
    if args.verbose:
        print(f"{args.resfile=}")

# Process the resfile.
log = []
with open(args.resfile) as fp:
    for line in fp:
        d = json.loads(line.strip())
        log.append(d)

if args.verbose:
    print(f"Found {len(log)} entries.")

# (lower-is-better, best-epoch, value)
best = {
    "tr_loss": (True, -1, float("inf")),
    "vl_loss": (True, -1, float("inf")),
    "vl_roc": (False, -1, -float("inf")),
    "vl_prc": (False, -1, -float("inf")),
}

for d in log:
    for k in best:
        if k not in d:
            continue

        l, e, v = best[k]

        if l and v < d[k]:
            continue
        if not l and v > d[k]:
            continue

        if args.verbose:
            print(f"Updating {k}: ({best[k]}) -> ", end="")

        best[k] = (l, d["epoch"], d[k])

        if args.verbose:
            print(f"({best[k]})")

d = {k: v[1] for k, v in best.items()}
print(d)

d = {k: v[2] for k, v in best.items()}
print("{", end="")
for i, (k, v) in enumerate(d.items()):
    if i != 0:
        print(", ", end="")
    print(f'"{k}": {v:.4f}', end="")
print("}")

