"""
Quickly analyze an experiment.

Usage:
    python scripts/quickanal.py [input] [--resfile RESFILE] [--logfile LOGFILE] [--jobid JOBID]
        [--json] [--quiet] [--verbose]
        [--no_summary] [--detailed] [--ddetailed]
        [--include [INCLUDE INCLUDE INCLUDE ...]]

Example:
    python scripts/quickanal.py 123456 --detailed --include epoch vl_roc vl_prc
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
from typing import Callable
from typing import Literal

import numpy as np
import pandas as pd

def float_print_json(d: dict[str, float]) -> None:
    print("{", end="")
    for i, (k, v) in enumerate(d.items()):
        if i != 0:
            print(", ", end="")
        print(f'"{k}": {v:.4f}', end="")
    print("}")

DISPLAY = (
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None,
)


parser = ArgumentParser()
parser.add_argument("input", type=str, nargs="?", default=None, help="Generic input, i.e., the resfile/logfile/jobid.")
parser.add_argument("--resfile", type=Path, help="Path to the results.jsonl file.")
parser.add_argument("--logfile", type=Path, help="Path to the training log file, from which the resfile can be located.")
parser.add_argument("--jobid", type=str, help="JobId of the experiment, from which the logfile can be located.")
parser.add_argument("--system", type=str, default=None, required=False, choices=["mkwics", "rc", "empire"])
parser.add_argument("--json", action="store_true", help="Output in JSON format.")
parser.add_argument("--no_summary", action="store_true", help="Do not print the summary (the 'best' value for every input).")
parser.add_argument("--detailed", action="store_true", help="Print detailed information (entries where all inputs are non-null).")
parser.add_argument("--ddetailed", action="store_true", help="Print very detailed information (all entries).")
parser.add_argument("--include", type=str, nargs="+",
    default=["tr_loss", "vl_loss", "vl_roc", "vl_prc", "tr_gpu_mem", "vl_gpu_mem", "tr_throughput", "vl_throughput"],
    choices=["tr_loss", "vl_loss", "aux_loss", "clf_loss", "vl_aux_loss", "vl_clf_loss", "vl_roc", "vl_prc", "tr_gpu_mem", "vl_gpu_mem", "tr_throughput", "vl_throughput"],
    help="Metrics to include in the analysis (aside from 'epoch' and 'glbl_step'). If not provided, all metrics are included.")
parser.add_argument("--quiet", action="store_true", help="Suppress non-essential output.")
parser.add_argument("--verbose", action="store_true", help="Print verbose output for debugging.")
args = parser.parse_args()


if args.verbose:
    print(f"{args.input=}")
    print(f"{args.resfile=}")
    print(f"{args.logfile=}")
    print(f"{args.jobid=}")
    print(f"{args.system=}")
    print(f"{args.json=}")
    print(f"{args.no_summary=}")
    print(f"{args.detailed=}")
    print(f"{args.ddetailed=}")
    print(f"{args.quiet=}")
    print(f"{args.verbose=}")
    print(f"{args.include=}")


def detect_system() -> Literal["mkwics", "rc", "empire"]:
    if subprocess.run(["squeue"], capture_output=True, text=True).returncode != 0:
        return "mkwics"
    if subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip() == "lk3591":
        return "rc"
    return "empire"


SYSTEM = args.system if args.system is not None else detect_system()
if args.verbose:
    print(f"{SYSTEM=}")

if SYSTEM == "mkwics":
    ROOT = Path("/home/lk3591/Documents/code/SSViT")
    LOGS = ROOT / "logs"
    BULK = ROOT
elif SYSTEM == "rc":
    ROOT = Path("/home/lk3591/Documents/code/SSViT")
    LOGS = ROOT / "logs"
    BULK = Path("/shared/rc/admalware/Documents/code/SSViT")
    root = Path("/shared/rc/admalware")
elif SYSTEM == "empire":
    ROOT = Path("/mnt/home/lkurlandski/Documents/code/SSViT")
    LOGS = ROOT / "logs"
    BULK = ROOT
else:
    raise ValueError(SYSTEM)
if args.verbose:
    print(f"{ROOT=}")
    print(f"{LOGS=}")
    print(f"{BULK=}")


# Determine the type of the generic input.
if args.input is not None:
    if (p := Path(args.input)).exists():
        with open(p) as fp:
            for line in fp:
                try:
                    json.loads(line.strip())
                    if args.verbose:
                        print("Input inferred to be resfile.")
                    args.resfile = p
                    break
                except Exception:
                    if args.verbose:
                        print("Input inferred to be logfile.")
                    args.logfile = p
                    break
    elif args.input.isdigit():
        if args.verbose:
            print("Input inferred to be jobid.")
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

# Define the metrics to investigate and how to summarize them.
summary: dict[str, Callable[[np.ndarray], float]] = {  # type: ignore[type-arg]
    "epoch": np.max,
    "glbl_step": np.max,
    "tr_loss": np.min,
    "vl_loss": np.min,
    "aux_loss": np.min,
    "clf_loss": np.min,
    "vl_aux_loss": np.min,
    "vl_clf_loss": np.min,
    "vl_roc": np.max,
    "vl_prc": np.max,
    "tr_gpu_mem": np.max,
    "vl_gpu_mem": np.max,
    "tr_throughput": np.mean,
    "vl_throughput": np.mean,
}
if "epoch" in args.include:
    args.include.remove("epoch")
args.include.insert(0, "epoch")
if "glbl_step" in args.include:
    args.include.remove("glbl_step")
args.include.insert(1, "glbl_step")
if args.include == ["epoch", "glbl_step"]:
    args.include = list(summary.keys())
if len(set(args.include)) != len(args.include):
    raise RuntimeError("Duplicate entries found in --include.")
summary = {k: v for k, v in summary.items() if k in args.include}
if args.verbose:
    print(f"Tracking metrics: {list(summary.keys())}")

# Process the resfile.
log: list[dict[str, int | float]] = []
with open(args.resfile) as fp:
    for line in fp:
        d = json.loads(line.strip())
        if "tr_gpu_mem" in d:
            d["tr_gpu_mem"] = d["tr_gpu_mem"] / (1024 ** 3)  # Convert to GB.
        if "vl_gpu_mem" in d:
            d["vl_gpu_mem"] = d["vl_gpu_mem"] / (1024 ** 3)  # Convert to GB.
        log.append({k: v for k, v in d.items() if k in summary})
df = pd.DataFrame(log)
df = df[[c for c in summary.keys() if c in df.columns]]
if args.verbose:
    print(f"Found {len(log)} entries.")

# Remove missing things from the summary dict.
summary = {k: v for k, v in summary.items() if k in df.columns}

# Print very detailed information, if requested.
if args.ddetailed:
    if not args.quiet:
        print("Training Details:")
    if args.json:
        for d in log:
            float_print_json(d)
    else:
        with pd.option_context(*DISPLAY):
            print(df)

# Print detailed information, if requested.
if args.detailed and not args.ddetailed:
    if not args.quiet:
        print("Training Details:")
    if args.json:
        for d in log:
            if all(k in d for k in summary):
                float_print_json(d)
    else:
        with pd.option_context(*DISPLAY):
            print(df.dropna())

# Print summary information, if requested.
if not args.no_summary:
    if not args.quiet:
        print("Training Summary:")
    best = {}
    for key, func in summary.items():
        best[key] = func(df[key].dropna().to_numpy())
    locs = {}
    for key, val in best.items():
        locs[key] = next(iter(df["epoch"][df[key] == val].tolist()), -1)
    if args.json:
        float_print_json(locs)
        float_print_json(best)
    else:
        longest1 = max(len(key) for key in best.keys())
        longest2 = max(len(f"{bst:.4f}") for bst in best.values())
        for key in summary:
            loc = locs[key]
            bst = best[key]
            spaces1 = " " * (longest1 - len(key))
            spaces2 = " " * (longest2 - len(f"{bst:.4f}"))
            print(f"{key}:{spaces1} {bst:.4f}{spaces2} (epoch {loc})")
