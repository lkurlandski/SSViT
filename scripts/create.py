"""
Create large batches of experiments.
"""

from argparse import ArgumentParser
from itertools import product
import math
from pathlib import Path
import sys
from typing import Any
import warnings

import lief
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import HierarchicalLevel
from src.binanal import CHARACTERISTICS
from src.helpers import Architecture
from src.helpers import ModelSize


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

    def get_outdir(self, world_size: int = 1) -> Path:
        parts = [
            f"arch--{self.arch.value}",
            f"size--{self.size.value}",
            f"level--{self.level.value}",
            f"entropy--{self.do_entropy}",
            f"characteristics--{'_'.join(sorted([str(c.name) for c in self.which_characteristics]))}",
            f"max_length--{self.max_length}",
            f"batch_size--{self.batch_size * self.gradient_accumulation_steps * world_size}",
            f"learning_rate--{self.learning_rate}",
            f"max_epochs--{self.max_epochs}",
            f"seed--{self.seed}",
        ]
        return Path("./output").joinpath(*parts)

    @property
    def gradient_accumulation_steps(self) -> int:
        return 1

    @property
    def batch_size(self) -> int:
        return 256

    @property
    def num_workers(self) -> int:
        return 8

    @property
    def num_threads_per_worker(self) -> int:
        return 1

    @property
    def learning_rate(self) -> float:
        return 1e-3

    @property
    def device(self) -> str:
        return "cuda"

    @property
    def mp16(self) -> bool:
        return True

    @property
    def tf32(self) -> bool:
        return True

    @property
    def max_epochs(self) -> float:
        return 10

    @property
    def eval_epochs(self) -> float:
        return 0.50

    @property
    def chpt_epochs(self) -> float:
        return 0.50

    @property
    def schd_epochs(self) -> float:
        return 1.0

    @property
    def logg_epochs(self) -> float:
        return 0.50


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
        tr_throughput = None
        vl_throughput = None

        if self.config.arch == Architecture.MCG:
            if self.config.max_length == 1 * 10**6:
                if self.config.batch_size == 256:
                    tr_throughput = 175
                    vl_throughput = 200

        if self.config.arch == Architecture.MC2:
            if self.config.max_length == 1 * 10**6:
                if self.config.batch_size == 256:
                    tr_throughput = 275
                    vl_throughput = 300

        if tr_throughput is None or vl_throughput is None:
            print(f"WARNING ({str(self.config)}): using fixed throughput of 100 samples/second.")
            tr_throughput = 100
            vl_throughput = 100

        tr_samples = 2339771 * self.config.max_epochs
        vl_samples = 539882  * self.config.max_epochs / self.config.eval_epochs

        tr_seconds = tr_samples / tr_throughput
        vl_seconds = vl_samples / tr_throughput

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
            f"--outdir {self.config.get_outdir(self.reqs.gpus_per_node * self.reqs.nodes)}",
            f"--arch {self.config.arch.value}",
            f"--size {self.config.size.value}",
            f"--level {self.config.level.value}",
            f"--do_entropy {self.config.do_entropy}",
            f"--which_characteristics {' '.join([str(c.name) for c in self.config.which_characteristics])}",
            f"--max_length {self.config.max_length}",
            f"--seed {self.config.seed}",
            f"--num_workers {self.config.num_workers}",
            f"--pin_memory {True if self.config.num_workers > 0 else False}",
            f"--gradient_accumulation_steps {self.config.gradient_accumulation_steps}",
            f"--tr_batch_size {self.config.batch_size}",
            f"--vl_batch_size {self.config.batch_size}",
            f"--ts_batch_size {self.config.batch_size}",
            f"--learning_rate {self.config.learning_rate}",
            f"--device {self.config.device}",
            f"--ddp {self.reqs.gpus_per_node > 1}",
            f"--mp16 {self.config.mp16}",
            f"--tf32 {self.config.tf32}",
            f"--max_epochs {self.config.max_epochs}",
            f"--eval_epochs {self.config.eval_epochs}",
            f"--chpt_epochs {self.config.chpt_epochs}",
            f"--schd_epochs {self.config.schd_epochs}",
            f"--logg_epochs {self.config.logg_epochs}",
        ])

        command = torchrun + "\n" + command if self.reqs.gpus_per_node > 1 else command

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
    chars = (
        lief.PE.Section.CHARACTERISTICS.MEM_READ,
        lief.PE.Section.CHARACTERISTICS.MEM_WRITE,
        lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE,
    )
    if not isinstance(config, Configuration):
        raise TypeError()
    if config.arch != Architecture.MC2:
        return False
    if config.which_characteristics and any(c not in chars for c in config.which_characteristics):
        return False
    if config.arch == Architecture.MCV:
        return False
    if config.arch == Architecture.VIT:
        return False
    if config.size != ModelSize.MD:
        return False
    if config.level != HierarchicalLevel.NONE:
        return False
    if config.do_entropy and config.which_characteristics:
        return False
    return True


def main() -> None:

    parser = ArgumentParser()
    args = parser.parse_args()

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
        [tuple()] + [(c,) for c in CHARACTERISTICS],
        [1000000],
        [0],
    )
    configurations = (Configuration(*config) for config in configurations)
    configurations = (config for config in configurations if config_fiter(config))
    configurations = sorted(configurations)

    alloutdirs: set[str] = set()
    allconfigs: set[str] = set()

    for config in tqdm(configurations):
        if str(config) in allconfigs:
            raise RuntimeError(f"ERROR ({str(config)}): duplicate configuration detected.")
        allconfigs.add(str(config))

        reqs = Requirements(config)

        outdir = config.get_outdir(reqs.gpus_per_node * reqs.nodes)
        if str(outdir) in alloutdirs:
            raise RuntimeError(f"ERROR ({str(outdir)}): duplicate output directory detected.")
        alloutdirs.add(str(outdir))
        if outdir.exists() and any(outdir.iterdir()):
            print(f"WARNING ({str(config)}): output directory exists at {str(outdir)}.")

        builder = ScriptBuilder(config, reqs)
        outfile = (outpath / str(config)).with_suffix(".sh")
        script = builder()
        outfile.write_text(script)


if __name__ == "__main__":
    main()
