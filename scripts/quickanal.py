"""
Quickly analyze an experiment.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument("input", type=str, nargs="?", default=None)
parser.add_argument("--resfile", type=Path)
parser.add_argument("--logfile", type=Path)
parser.add_argument("--jobid", type=str)
parser.add_argument("--json", action="store_true")
parser.add_argument("--detailed", action="store_true")
parser.add_argument("--ddetailed", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

ROOT = Path("/home/lk3591/Documents/code/SSViT")
LOGS = ROOT / "logs"
BULK = Path("/shared/rc/admalware/Documents/code/SSViT")

if args.verbose:
    print(f"{args.input=}")
    print(f"{args.resfile=}")
    print(f"{args.logfile=}")
    print(f"{args.jobid=}")
    print(f"{args.detailed=}")
    print(f"{args.ddetailed=}")
    print(f"{args.quiet=}")
    print(f"{args.summary=}")

if args.input is not None:
    if (p := Path(args.input)).exists():
        with open(p) as fp:
            for line in fp:
                try:
                    json.loads(line.strip())
                    if args.verbose:
                        print(f"Input inferred to be resfile.")
                    args.resfile = p
                except:
                    if args.verbose:
                        print(f"Input inferred to be logfile.")
                    args.logfile = p
    elif args.input.isdigit():
        if args.verbose:
            print(f"Input inferred to be jobid.")
        args.jobid = args.input
    else:
        raise RuntimeError(f"Could not infer the type of generic input {args.input}.")

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
best: dict[str, tuple[bool, int, float]] = {
    "tr_loss": (True, -1, float("inf")),
    "aux_loss": (True, -1, float("inf")),
    "clf_loss": (True, -1, float("inf")),
    "vl_loss": (True, -1, float("inf")),
    "vl_aux_loss": (True, -1, float("inf")),
    "vl_clf_loss": (True, -1, float("inf")),
    "vl_roc": (False, -1, -float("inf")),
    "vl_prc": (False, -1, -float("inf")),
}

def float_print_json(d: dict[str, float]) -> None:
    print("{", end="")
    for i, (k, v) in enumerate(d.items()):
        if i != 0:
            print(", ", end="")
        print(f'"{k}": {v:.4f}', end="")
    print("}")


if args.detailed or args.ddetailed:
    if not args.quiet:
        print("Training Details:")
    if args.json:
        for d in log:
            if not args.ddetailed and not all(k in d for k in best):
                continue
            d = {k: d[k] for k in best if k in d}
            float_print_json(d)
    else:
        df = pd.DataFrame(log)
        if not args.ddetailed:
            df = df[df["vl_loss"].notnull()]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(df[[k for k in best if k in df.columns]])

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

if not args.quiet:
    print("Training Summary:")
if args.json:
    d = {k: e for k, (l, e, v) in best.items()}
    print(d)
    d = {k: v for k, (l, e, v) in best.items()}
    float_print_json(d)
else:
    for k, (l, e, v) in best.items():
        print(f"{k} @{e} {v:.4f}")

