"""
Create large batches of experiments.
"""

from argparse import ArgumentParser
from itertools import chain
from itertools import combinations
from itertools import product
import math
import os
from pathlib import Path
import sys
from typing import Any
from typing import Optional
import warnings

import lief
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import HierarchicalLevel
from src.binanal import CHARACTERISTICS
from src.helpers import Architecture
from src.helpers import ModelSize
from src.helpers import PatcherArchitecture
from src.helpers import PositionalEncodingArchitecture
from src.helpers import PatchPositionalEncodingArchitecture
from src.helpers import Scheduler


# ruff: noqa: F541


DEBUG = False


def fixed_width_string(string: Any, width: int, char: str = " ", left: bool = False) -> str:
    string = str(string)
    if left:
        return char * (width - len(string)) + string[0:width]
    return string[0:width] + char * (width - len(string))


class Configuration:

    def __init__(
        self,
        arch: Architecture,
        size: ModelSize,
        level: HierarchicalLevel,
        do_entropy: bool,
        which_characteristics: tuple[lief.PE.Section.CHARACTERISTICS, ...],
        max_length: int,
        seed: int,
    ) -> None:
        self.arch = arch
        self.size = size
        self.level = level
        self.do_entropy = do_entropy
        self.which_characteristics = which_characteristics
        self.max_length = max_length
        self.seed = seed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Configuration):
            return NotImplemented
        return self._attrs() == other._attrs()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Configuration):
            return NotImplemented
        return self._attrs() < other._attrs()

    def __repr__(self) -> str:
        return (
            f"Configuration("
            f"  arch={self.arch},\n"
            f"  size={self.size},\n"
            f"  level={self.level},\n"
            f"  do_entropy={self.do_entropy},\n"
            f"  which_characteristics={self.which_characteristics},\n"
            f"  max_length={self.max_length},\n"
            f"  seed={self.seed},\n"
            ")"
        )

    def __str__(self) -> str:
        return (
            f"{fixed_width_string(self.arch.value, 3, '_')}-"
            f"{fixed_width_string(self.size.value, 3, '_')}-"
            f"{fixed_width_string(self.level.value, 3, '_')}-"
            f"{'t' if self.do_entropy else 'f'}-"
            f"{fixed_width_string(sum(map(int, self.which_characteristics)), len(str(sum(map(int, CHARACTERISTICS)))), '0', left=True)}-"
            f"{fixed_width_string(self.max_length, 7, '8', left=True)}-"
            f"{fixed_width_string(self.seed, 1, '_')}"
        )

    def _attrs(self) -> tuple[Any, ...]:
        return (
            self.arch.value,
            self.size.value,
            self.level.value,
            self.do_entropy,
            tuple(map(int, self.which_characteristics)),
            self.max_length,
            self.seed,
        )

    def get_gradient_accumulation_steps(self, world_size: int = 1) -> int:
        batch_size, remainder = divmod(self.batch_size, world_size)
        if remainder != 0:
            print(f"WARNING ({str(self)}): {self.batch_size=} not divisible by {world_size=}.")
        steps, remainder = divmod(batch_size, self.per_device_batch_size)
        if remainder != 0:
            print(f"WARNING ({str(self)}): {batch_size=} not divisible by {self.per_device_batch_size=}.")
        return max(1, steps)

    @property
    def posenc(self) -> PositionalEncodingArchitecture:
        return PositionalEncodingArchitecture.LEARNED

    @property
    def patchposenc(self) -> PatchPositionalEncodingArchitecture:
        return PatchPositionalEncodingArchitecture.BTH

    @property
    def outdir(self) -> Path:
        root = Path("./output")
        if DEBUG:
            root = root / "tmp"
        parts = [
            f"arch--{self.arch.value}",
            f"size--{self.size.value}",
            f"level--{self.level.value}",
            f"entropy--{self.do_entropy}",
            f"characteristics--{'_'.join(sorted([str(c.name) for c in self.which_characteristics]))}",
            f"max_length--{self.max_length}",
            f"batch_size--{self.batch_size}",
            f"lr_max--{self.lr_max}",
            f"weight_decay--{self.weight_decay}",
            f"warmup_ratio--{self.warmup_ratio}",
            f"label_smoothing--{self.label_smoothing}",
            f"max_epochs--{self.max_epochs}",
            f"seed--{self.seed}",
        ]
        return root.joinpath(*parts)

    @property
    def batch_size(self) -> int:
        return 1024

    @property
    def per_device_batch_size(self) -> int:
        # I don't really know why, but throughput seems to plateau beyond 256 samples/device.
        # Unless overridden, cap the per-device batch size at 256.
        if os.environ.get("USE_MAX_PER_DEVICE_BATCH_SIZE", "0") == "1":
            return self.max_per_device_batch_size
        return min(256, self.max_per_device_batch_size)

    @property
    def max_per_device_batch_size(self) -> int:
        if self.arch == Architecture.MCV:
            return 64   # O(T)
        if self.arch == Architecture.MC2:
            return 1024 # O(1)
        if self.arch == Architecture.MCG:
            return 256  # O(1)

        # Assumes constant-memory encoder. # O(1)
        # TODO: there appears to be large memory fluctuations here.
        if self.arch == Architecture.VIT:
            if self.size == ModelSize.SM:  # O(N)
                return 2048
            if self.size == ModelSize.MD:  # O(N)
                return 512
            if self.size == ModelSize.LG:  # O(N)
                if self.do_entropy or self.which_characteristics:
                    return 64
                return 128

        raise NotImplementedError(f"ERROR ({str(self)}): batch size not defined for this configuration.")

    @property
    def num_workers(self) -> int:
        return 8

    @property
    def num_threads_per_worker(self) -> int:
        return 1

    @property
    def sched(self) -> Scheduler:
        return Scheduler.OCLR

    @property
    def lr_max(self) -> float:
        return 3.16e-4

    @property
    def lr_beg(self) -> float:
        return 0.050 * self.lr_max

    @property
    def lr_end(self) -> float:
        return 0.010 * self.lr_max

    @property
    def weight_decay(self) -> float:
        return 0.001

    @property
    def warmup_ratio(self) -> float:
        return 0.10

    @property
    def label_smoothing(self) -> float:
        return 0.00

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def mp16(self) -> bool:
        return False

    @property
    def tf32(self) -> bool:
        return True

    @property
    def stopper_patience(self) -> int:
        return -1

    @property
    def max_epochs(self) -> float:
        if DEBUG:
            return 2
        return 10

    @property
    def eval_epochs(self) -> float:
        if DEBUG:
            return 1
        return 0.50

    @property
    def chpt_epochs(self) -> float:
        if DEBUG:
            return 1
        return 0.50

    @property
    def logg_epochs(self) -> float:
        if DEBUG:
            return 1
        return 0.05


# Classifier throughput in samples/second on NVIDIA A100. Measurements taken using
# a sequence length of 1_000_000 and various logical batch sizes, e.g., 128 and 256.
# From experiments, throughput appears to scale linearly with batch size, but only
# up until a certain point, past which it plateaus; therefore, its less critical to
# consider the exact batch size used in these measurements. By constast, throughput
# appears to scale almost perfectly linearly (inversely, however) with sequence length.
# On the whole, these numbers vary wildly and should be taken with a grain of salt.
# (Architecture, do_entropy, do_characteristics) : (vl_throughput, tr_throughout)
THROUGHPUTS: dict[tuple[Architecture, ModelSize, bool, bool], Optional[tuple[float, float]]] = {
    (Architecture.MCV, ModelSize.MD, False, False) : None,
    (Architecture.MCV, ModelSize.MD, False, True)  : None,
    (Architecture.MCV, ModelSize.MD, True,  False) : None,
    (Architecture.MCV, ModelSize.MD, True,  True)  : None,

    (Architecture.MC2, ModelSize.MD, False, False) : (675, 450),
    (Architecture.MC2, ModelSize.MD, False, True)  : (425, 325),
    (Architecture.MC2, ModelSize.MD, True,  False) : (375, 300),
    (Architecture.MC2, ModelSize.MD, True,  True)  : (300, 270),

    (Architecture.MCG, ModelSize.MD, False, False) : (275, 120),
    (Architecture.MCG, ModelSize.MD, False, True)  : (150, 100),
    (Architecture.MCG, ModelSize.MD, True,  False) : (150, 100),
    (Architecture.MCG, ModelSize.MD, True,  True)  : (160,  90),

    (Architecture.VIT, ModelSize.MD, False, False) : (450, 350),
    (Architecture.VIT, ModelSize.MD, False, True)  : (275, 225),
    (Architecture.VIT, ModelSize.MD, True,  False) : (250, 225),
    (Architecture.VIT, ModelSize.MD, True,  True)  : (275, 225),

    (Architecture.VIT, ModelSize.LG, False, False) : (325, 150),
    (Architecture.VIT, ModelSize.LG, False, True)  : (150, 100),
    (Architecture.VIT, ModelSize.LG, True,  False) : (150, 100),
    (Architecture.VIT, ModelSize.LG, True,  True)  : (150, 100),
}


class Requirements:
    """
    Compute resource requirements for a job given its configuration.
    """

    MAX_CPU_PER_NODE = 36
    MAX_GPU_PER_NODE = 4
    MAX_MEM_PER_NODE = 360 * 10**9

    def __init__(self, config: Configuration) -> None:
        self.config = config

    @property
    def time(self) -> int:
        """Return the number of seconds required for the job (configure)."""
        key = (self.config.arch, self.config.size, self.config.do_entropy, bool(self.config.which_characteristics))

        if (vl_tr_throughputs := THROUGHPUTS.get(key)) is not None:
            tr_throughput = vl_tr_throughputs[1]
            vl_throughput = vl_tr_throughputs[0]
        else:
            print(f"WARNING ({str(self.config)}): throughput benchmark not found.")
            tr_throughput = 100
            vl_throughput = 100

        tr_throughput = tr_throughput / (self.config.max_length / 1e6) * self.gpus_per_node * self.nodes
        vl_throughput = vl_throughput / (self.config.max_length / 1e6) * self.gpus_per_node * self.nodes

        tr_samples = 2339771 * self.config.max_epochs
        vl_samples = 539882  * self.config.max_epochs / self.config.eval_epochs

        tr_seconds = tr_samples / tr_throughput
        vl_seconds = vl_samples / vl_throughput

        seconds = tr_seconds + vl_seconds
        seconds += 0.10 * seconds + 0.25 * 3600

        return int(seconds)

    @property
    def mem(self) -> int:
        """Return the number of bytes required for the job (configure)."""
        return 64 * self.gpus_per_node * 2**30

    @property
    def nodes(self) -> int:
        """Return the number of nodes required for the job (configure)."""
        return 1

    @property
    def gpus_per_node(self) -> int:
        """Return the number of GPUs per node required for the job (configure)."""
        return 1

    @property
    def ntasks_per_node(self) -> int:
        """Return the number of tasks per node required for the job (constant)."""
        return 1

    @property
    def world_size(self) -> int:
        """Return the total number of GPUs required for the job (constant)."""
        return self.gpus_per_node * self.nodes

    @property
    def cpus_per_task(self) -> int:
        """Return the number of CPUs per task/node required for the job (constant)."""
        return self.gpus_per_node * (self.config.num_workers * self.config.num_threads_per_worker + 1)

    @property
    def omp_num_threads(self) -> int:
        """Return the number of OMP threads required for the job (constant)."""
        return self.config.num_threads_per_worker


class ScriptBuilder:

    def __init__(self, config: Configuration, reqs: Requirements) -> None:
        self.config = config
        self.reqs = reqs

    def __call__(self) -> str:
        shebang = "\n".join([
            "#!/bin/bash",
        ])

        slurm = "\n".join([
            f"#SBATCH --account=admalware",
            f"#SBATCH --output=./logs/%x_%j.out",
            f"#SBATCH --partition=tier3",
            f"#SBATCH --nodes={self.reqs.nodes}",
            f"#SBATCH --gpus-per-node=a100:{self.reqs.gpus_per_node}",
            f"#SBATCH --ntasks-per-node={self.reqs.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.reqs.cpus_per_task}",
            f"#SBATCH --job-name={str(self.config)}",
            f"#SBATCH --time={self.fmt_time(self.reqs.time)}",
            f"#SBATCH --mem={self.fmt_mem(self.reqs.mem)}",
        ])

        environment = "\n".join([
            "source ./env/bin/activate",
        ])

        variables = "\n".join([
            f"export OMP_NUM_THREADS={self.reqs.omp_num_threads}",
            f"export PTW_NUM_THREADS={self.reqs.omp_num_threads}",
        ])

        torchrun = " \\\n".join([
            "torchrun",
            "--no-python",
            "--standalone",
            "--start-method 'forkserver'",
            f"--nnodes {self.reqs.nodes}",
            f"--nproc-per-node {self.reqs.gpus_per_node}",
        ])

        command = " \\\n".join([
            "python",
            "src/main.py",
            f"--outdir {self.config.outdir}",
            f"--arch {self.config.arch.value}",
            f"--posenc {self.config.posenc.value}",
            f"--patchposenc {self.config.patchposenc.value}",
            f"--size {self.config.size.value}",
            f"--level {self.config.level.value}",
            f"--do_entropy {self.config.do_entropy}",
            f"--which_characteristics {' '.join([str(c.name) for c in self.config.which_characteristics])}",
            f"--max_length {self.config.max_length}",
            f"--seed {self.config.seed}",
            f"--num_workers {self.config.num_workers}",
            f"--pin_memory {True if self.config.num_workers > 0 else False}",
            f"--gradient_accumulation_steps {self.config.get_gradient_accumulation_steps(self.reqs.world_size)}",
            f"--tr_batch_size {self.config.per_device_batch_size}",
            f"--vl_batch_size {self.config.per_device_batch_size}",
            f"--ts_batch_size {self.config.per_device_batch_size}",
            f"--sched {self.config.sched.value}",
            f"--lr_max {self.config.lr_max}",
            f"--lr_beg {self.config.lr_beg}",
            f"--lr_end {self.config.lr_end}",
            f"--weight_decay {self.config.weight_decay}",
            f"--warmup_ratio {self.config.warmup_ratio}",
            f"--label_smoothing {self.config.label_smoothing}",
            f"--stopper_patience {self.config.stopper_patience}",
            f"--stopper_metric {'vl_roc'}",
            f"--stopper_mode {'max'}",
            f"--device {self.config.device}",
            f"--ddp {self.reqs.gpus_per_node > 1}",
            f"--mp16 {self.config.mp16}",
            f"--tf32 {self.config.tf32}",
            f"--max_epochs {self.config.max_epochs}",
            f"--eval_epochs {self.config.eval_epochs}",
            f"--chpt_epochs {self.config.chpt_epochs}",
            f"--logg_epochs {self.config.logg_epochs}",
        ])

        if DEBUG:
            command += " \\\n" + " \\\n".join([
                f"--tr_num_samples 8192",
                f"--vl_num_samples 8192",
                f"--ts_num_samples 8192",
            ])

        command = torchrun + " \\" + "\n" + command if self.reqs.gpus_per_node > 1 else command

        script = ""
        script += shebang + "\n\n"
        script += slurm + "\n\n"
        script += environment + "\n\n"
        script += variables + "\n\n"
        script += command + "\n\n"

        return script

    @staticmethod
    def fmt_time(s: float, r: int = 60) -> str:
        s = int(math.ceil(s / r) * r)
        days = s // (24 * 3600)
        s %= (24 * 3600)
        hours = s // 3600
        s %= 3600
        minutes = s // 60
        s %= 60
        return f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(s):02d}"

    @staticmethod
    def fmt_mem(b: int) -> str:
        return f"{round(b / 2**30)}G"


def config_fiter(config: Configuration) -> bool:
    """Return True if the configuration should be run, False otherwise."""
    if not isinstance(config, Configuration):
        raise TypeError()

    # Debug
    if DEBUG:
        ...

    # Architecture
    if config.arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG) and config.size == ModelSize.LG:
        return False
    if config.arch == Architecture.MCV:
        return False
    if config.arch in (Architecture.VIT,) and config.size != ModelSize.LG:
        return False
    if config.size == ModelSize.SM:
        return False
    if config.arch != Architecture.VIT:
        return False

    # Structure
    if config.level not in (HierarchicalLevel.NONE, HierarchicalLevel.COARSE):
        return False

    # Semantics
    CHARS = (
        lief.PE.Section.CHARACTERISTICS.MEM_READ,
        lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE,
        lief.PE.Section.CHARACTERISTICS.CNT_CODE,
    )
    if len(config.which_characteristics) > 0 and any(c not in CHARS for c in config.which_characteristics):
        return False
    if len(config.which_characteristics) not in (0, 1, len(CHARS)):
        return False
    if config.do_entropy and len(config.which_characteristics) not in (0, len(CHARS)):
        return False
    if not config.do_entropy and len(config.which_characteristics) == len(CHARS):
        return False

    if config.do_entropy and len(config.which_characteristics) == len(CHARS):
        return False

    return True


def main() -> None:

    parser = ArgumentParser(description="Create large batches of experiments.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    outpath = Path("./run")
    outpath.mkdir(exist_ok=True)
    for f in outpath.iterdir():
        if f.is_file():
            f.unlink()

    configurations = product(
        [a for a in Architecture],
        [s for s in ModelSize],
        [l for l in HierarchicalLevel],
        [True, False],
        chain.from_iterable(combinations(CHARACTERISTICS, r) for r in range(len(CHARACTERISTICS) + 1)),
        [1000000],
        [0],
    )
    configurations = (Configuration(*config) for config in configurations)
    configurations = (config for config in configurations if config_fiter(config))
    configurations = sorted(configurations)

    alloutdirs: set[str] = set()
    allconfigs: set[str] = set()

    for config in tqdm(configurations, disable=True):
        if str(config) in allconfigs:
            raise RuntimeError(f"ERROR ({str(config)}): duplicate configuration detected.")
        allconfigs.add(str(config))

        reqs = Requirements(config)

        if str(config.outdir) in alloutdirs:
            raise RuntimeError(f"ERROR ({str(config.outdir)}): duplicate output directory detected.")
        alloutdirs.add(str(config.outdir))
        if config.outdir.exists() and any(config.outdir.iterdir()) and not DEBUG:
            print(f"WARNING ({str(config)}): output directory exists at {str(config.outdir)}.")

        builder = ScriptBuilder(config, reqs)
        outfile = (outpath / str(config)).with_suffix(".sh")
        script = builder()
        outfile.write_text(script)
        print(f"{outfile}")


if __name__ == "__main__":
    main()
