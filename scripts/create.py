"""
Create large batches of experiments.
"""

from argparse import ArgumentParser
import inspect
from itertools import chain
from itertools import combinations
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from typing import Literal
from typing import Optional
from typing import Iterable
import warnings

import lief
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.binanal import HierarchicalLevel
from src.binanal import CHARACTERISTICS
from src.helpers import Design
from src.helpers import Architecture
from src.helpers import PatcherArchitecture
from src.helpers import PositionalEncodingArchitecture
from src.helpers import PatchPositionalEncodingArchitecture
from src.helpers import Scheduler
from src.helpers import MainArgs


# ruff: noqa: F541


DEBUG: bool
BENCH: bool
NGPUS: int
GPU: Literal["a100", "h100", "gh200", "nvidia_h200", "nvidia_h100_80gb_hbm3"]
SYSTEM: Literal["mkwics", "rc", "empire"]


TR_SAMPLES = 2339771
VL_SAMPLES =  539882


def fixed_width_string(string: Any, width: int, char: str = " ", left: bool = False) -> str:
    string = str(string)
    if left:
        return char * (width - len(string)) + string[0:width]
    return string[0:width] + char * (width - len(string))


class Configuration:

    def __init__(
        self,
        design: Design,
        arch: Architecture,
        parch: PatcherArchitecture,
        posenc: PositionalEncodingArchitecture,
        patchposenc: PatchPositionalEncodingArchitecture,
        level: HierarchicalLevel,
        do_entropy: bool,
        which_characteristics: tuple[lief.PE.Section.CHARACTERISTICS, ...],
        max_length: int,
        seed: int,
        model_config: dict[str, str | int | float | bool],
    ) -> None:
        self.design = design
        self.arch = arch
        self.parch = parch
        self.posenc = posenc
        self.patchposenc = patchposenc
        self.level = level
        self.do_entropy = do_entropy
        self.which_characteristics = which_characteristics
        self.max_length = max_length
        self.seed = seed
        self._model_config = model_config

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
            f"  design={self.design},\n"
            f"  arch={self.arch},\n"
            f"  parch={self.parch},\n"
            f"  posenc={self.posenc},\n"
            f"  patchposenc={self.patchposenc},\n"
            f"  level={self.level},\n"
            f"  do_entropy={self.do_entropy},\n"
            f"  which_characteristics={self.which_characteristics},\n"
            f"  max_length={self.max_length},\n"
            f"  seed={self.seed},\n"
            ")"
        )

    def __str__(self) -> str:
        parts = [
            fixed_width_string(self.design.value, 3, '_'),
            fixed_width_string(self.arch.value, 3, '_'),
            fixed_width_string(self.parch.value, 3, '_'),
            fixed_width_string(self.posenc.value, 3, '_'),
            fixed_width_string(self.patchposenc.value, 3, '_'),
            fixed_width_string(self.level.value, 3, '_'),
            't' if self.do_entropy else 'f',
            fixed_width_string(sum(map(int, self.which_characteristics)), len(str(sum(map(int, CHARACTERISTICS)))), '0', left=True),
            fixed_width_string(self.max_length, 7, '0', left=True),
            fixed_width_string(self.seed, 1, '_'),
        ]
        for key, value in sorted(self.model_config.items()):
            key = key.replace("moe_", "")
            key = key.replace("patcher_", "")
            parts.append(fixed_width_string(key, 2, "_") + fixed_width_string(value, 2, '_'))
        return "-".join(parts)

    def _attrs(self) -> tuple[Any, ...]:
        return (
            self.design.value,
            self.arch.value,
            self.parch.value,
            self.posenc.value,
            self.patchposenc.value,
            self.level.value,
            self.do_entropy,
            tuple(map(int, self.which_characteristics)),
            self.max_length,
            self.seed,
            tuple(sorted(self.model_config.items())),
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
    def outdir(self) -> Path:
        # TODO: it would be cleaner to separate the output root from the outdir.
        if SYSTEM == "mkwics":
            root = Path("/home/lk3591/")
        elif SYSTEM == "rc":
            root = Path("/shared/rc/admalware")
        elif SYSTEM == "empire":
            root = Path("/mnt/home/lkurlandski/")
        else:
            raise ValueError(SYSTEM)

        root /= "Documents/code/SSViT/output"
        if DEBUG:
            root = root / "debug"
        elif BENCH:
            root = root / "bench"
        else:
            root /= "04"

        parts = [
            f"design--{self.design.value}",
            f"arch--{self.arch.value}",
            f"parch--{self.parch.value}",
            f"posenc--{self.posenc.value}",
            f"patchposenc--{self.patchposenc.value}",
            f"level--{self.level.value}",
            f"entropy--{self.do_entropy}",
            f"characteristics--{'_'.join(sorted([str(c.name) for c in self.which_characteristics]))}",
            f"max_length--{self.truncate_length}",
            f"max_length_per_structure--{self.max_length_per_structure}",
            f"model_config--{'_'.join(sorted([f'{k}={v}' for k, v in self.model_config.items()]))}",
            f"batch_size--{self.batch_size}",
            f"lr_max--{self.lr_max}",
            f"lr_nclr_ncycles--{self.lr_nclr_ncycles}",
            f"lr_nclr_max_lr_gamma--{self.lr_nclr_max_lr_gamma}",
            f"weight_decay--{self.weight_decay}",
            f"warmup_ratio--{self.warmup_ratio}",
            f"label_smoothing--{self.label_smoothing}",
            f"max_epochs--{self.max_epochs}",
            f"freeze_embedding_epoch--{self.freeze_embedding_epoch}",
            f"seed--{self.seed}",
        ]

        return root.joinpath(*parts)

    @property
    def model_config(self) -> dict[str, str | int | float | bool]:
        config: dict[str, str | int | float | bool] = {}
        config.update(self._model_config)
        return config

    @property
    def resume(self) -> bool:
        return False

    @property
    def batch_size(self) -> int:
        return 1024

    @property
    def per_device_batch_size(self) -> int:
        # NOTE: the largest batch size that fits on a single GPU for each architecture
        # is not nessecarily the same as the batch size that yields the best throughput
        # and, in general, it seems that increasing the batch size to maximize GPU memory
        # tends to actually decrease the overall throughput slightly.

        if self.arch == Architecture.MCV:
            return 64   # O(T)
        if self.arch == Architecture.MC2:
            return 1024 # O(1)
        if self.arch == Architecture.MCG:
            return 256  # O(1)

        # Assumes constant-memory encoder. # O(1)
        if self.arch == Architecture.VIT and self.parch in (PatcherArchitecture.EXP, PatcherArchitecture.MEM, PatcherArchitecture.BAS):
            if self.design == Design.FLAT:
                if self.do_entropy or self.which_characteristics:
                    return 32
                return 64
            if self.design == Design.HIERARCHICAL:
                if self.do_entropy or self.which_characteristics:
                    return 16
                return 32
            if self.design == Design.STRUCTURAL:
                if self.do_entropy or self.which_characteristics:
                    return 32
                return 32

        if self.arch == Architecture.VIT and self.parch == PatcherArchitecture.DWC:
            return 128

        print(f"WARNING ({str(self)}): per_device_batch_size not found.")
        return 64

    def get_throughput(self) -> tuple[float, float]:

        # Throughputs for the FLAT model on 1MiB input length.
        match self.arch:
            case Architecture.MCV:
                tr_throughput = 100.00
                vl_throughput = 100.00
            case Architecture.MC2:
                tr_throughput = 450.00
                vl_throughput = 675.00
            case Architecture.MCG:
                tr_throughput = 120.00
                vl_throughput = 275.00
            case Architecture.VIT:
                tr_throughput = 150.00
                vl_throughput = 325.00
            case _:
                raise NotImplementedError(self.arch)

        # Adjust based on hierarchical level.
        match self.level:
            case HierarchicalLevel.NONE:
                tr_throughput *= 1.00
                vl_throughput *= 1.00
            case HierarchicalLevel.COARSE:
                tr_throughput *= 0.75
                vl_throughput *= 0.50
            case HierarchicalLevel.ROUGH:
                tr_throughput *= 0.65
                vl_throughput *= 0.45
            case HierarchicalLevel.MIDDLE:
                tr_throughput *= 0.60
                vl_throughput *= 0.40
            case HierarchicalLevel.FINE:
                tr_throughput *= 0.40
                vl_throughput *= 0.30
            case _:
                raise NotImplementedError(self.level)

        # Non-constant memory patcher architectures descrease throughput by around 4x.
        match self.parch:
            case PatcherArchitecture.MEM:
                tr_throughput *= 1.00
                vl_throughput *= 1.00
            case _:
                tr_throughput *= 0.25
                vl_throughput *= 0.25

        # Adjust based on input length (compute scales roughly linearly).
        tr_throughput = tr_throughput / (self.max_length / 2 ** 20)
        vl_throughput = vl_throughput / (self.max_length / 2 ** 20)

        return tr_throughput, vl_throughput

    @property
    def tr_throughput(self) -> float:
        return self.get_throughput()[0]

    @property
    def vl_throughput(self) -> float:
        return self.get_throughput()[1]

    @property
    def num_workers(self) -> int:
        if DEBUG:
            return 1
        return 8

    @property
    def num_threads_per_worker(self) -> int:
        return 1

    @property
    def sched(self) -> Scheduler:
        return Scheduler.NCLR

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
    def lr_nclr_ncycles(self) -> int:
        return 3

    @property
    def lr_nclr_max_lr_gamma(self) -> float:
        return 0.25

    @property
    def ema_enable(self) -> bool:
       return True

    @property
    def ema_decay(self) -> bool:
       return 0.9995

    @property
    def auxillary_loss_weight(self) -> float:
        if self.parch == PatcherArchitecture.EXP:
            return 0.005
        return 0.0

    @property
    def assert_auxillary_loss(self) -> bool:
        return self.parch == PatcherArchitecture.EXP

    @property
    def weight_decay(self) -> float:
        return 0.001

    @property
    def warmup_ratio(self) -> float:
        return 0.05

    @property
    def label_smoothing(self) -> float:
        return 0.00

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
    def max_structures(self) -> Optional[int]:
        if self.design == Design.STRUCTURAL:
            return 255  # [CLS] pooling inserts one additional token.
        return None

    @property
    def truncate_length(self) -> Optional[int]:
        if self.design == Design.STRUCTURAL:
            return None
        return self.max_length

    @property
    def max_length_per_structure(self) -> Optional[int]:
        if self.design == Design.STRUCTURAL:
            return self.max_length
        return None

    @property
    def stopper_patience(self) -> int:
        return -1

    @property
    def share_embeddings(self) -> bool:
        if self.design == Design.STRUCTURAL and self.parch == PatcherArchitecture.EXP:
            return True
        return False

    @property
    def share_patchers(self) -> bool:
        if self.design == Design.STRUCTURAL and self.parch == PatcherArchitecture.EXP:
            return True
        return False

    @property
    def max_epochs(self) -> float:
        if DEBUG:
            return 2
        if BENCH:
            return 1
        return 30

    @property
    def eval_epochs(self) -> float:
        if DEBUG:
            return 0.50
        if BENCH:
            return 1
        return 1.0

    @property
    def chpt_epochs(self) -> float:
        if DEBUG:
            return 1
        if BENCH:
            return 1
        return 0.50

    @property
    def logg_epochs(self) -> float:
        if DEBUG:
            return 0.25
        if BENCH:
            return 1
        return 0.05

    @property
    def enable_checkpoint(self) -> bool:
        return False

    @property
    def enable_compile(self) -> bool:
        return True

    @property
    def static_shapes_bin_patcher_seq_lengths(self) -> bool:
        return False

    @property
    def static_shapes_bin_patcher_batch_sizes(self) -> bool:
        return False

    @property
    def static_shapes_bin_backbone_seq_lengths(self) -> bool:
        return False

    @property
    def static_shapes_bin_backbone_batch_sizes(self) -> bool:
        return False

    @property
    def update_embedding_steps(self) -> int:
        return 1

    @property
    def freeze_embedding_epoch(self) -> int:
        return -1

    @property
    def find_unused_parameters(self) -> bool:
        if self.update_embedding_steps > 1:
            return True
        if self.freeze_embedding_epoch != -1:
            return True
        return False

    @property
    def param_grad_none(self) -> Literal["raise", "warn", "ignore"]:
        if self.find_unused_parameters:
            return "ignore"
        return "warn"

    def _num_samples(self, world_size: int) -> Optional[int]:
        minimum = 1024 * max(self.num_workers, 1) * world_size
        if DEBUG:
            return minimum
        if BENCH:
            return 4 * minimum
        return None

    def tr_num_samples(self, world_size: int) -> Optional[int]:
        return self._num_samples(world_size)

    def vl_num_samples(self, world_size: int) -> Optional[int]:
        return self._num_samples(world_size)

    def ts_num_samples(self, world_size: int) -> Optional[int]:
        return self._num_samples(world_size)


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
        return int(8 * 24 * 3600 / self.world_size)  # FIXME

        tr_throughput = self.config.tr_throughput * self.world_size
        vl_throughput = self.config.vl_throughput * self.world_size

        tr_seconds = 0
        vl_seconds = 0

        # Using a non-constant-memory encoder with frozen embeddings speeds up training by about 4x.
        # This increments the time by the slow period before boosting the tr_throughput value.
        if self.config.freeze_embedding_epoch != -1 and self.config.parch != PatcherArchitecture.MEM:
            tr_samples = TR_SAMPLES * self.config.freeze_embedding_epoch
            tr_seconds += tr_samples / tr_throughput
            tr_throughput *= 4

        tr_samples = TR_SAMPLES * (self.config.max_epochs - max(self.config.freeze_embedding_epoch, 0))
        vl_samples = VL_SAMPLES * self.config.max_epochs / self.config.eval_epochs

        tr_seconds += tr_samples / tr_throughput
        vl_seconds += vl_samples / vl_throughput

        if self.config.freeze_embedding_epoch != -1:
            tr_samples = TR_SAMPLES * self.config.freeze_embedding_epoch
            tr_seconds += tr_samples / (tr_throughput / 4)  # Assume 4x speedup when freezing embedding.

        add = 0.50 * 3600
        mul = 0.15

        seconds = tr_seconds + vl_seconds
        seconds += mul * seconds + add

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
        if NGPUS is not None:
            return NGPUS
        if self.config.arch == Architecture.VIT:
            return 4
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

        account: str   = "admalware"
        partition: str = "tier3"
        env: str       = "env"

        if SYSTEM == "empire":
            account   = "rit"
            partition = "rit"

        if SYSTEM == "rc" and GPU == "gh200":
            partition = "grace"
            env = "env-grace"

        slurm = "\n".join([
            f"#SBATCH --account={account}",
            f"#SBATCH --output=./logs/%x_%j.out",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={self.reqs.nodes}",
            f"#SBATCH --gpus-per-node={GPU}:{self.reqs.gpus_per_node}",
            f"#SBATCH --ntasks-per-node={self.reqs.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={self.reqs.cpus_per_task}",
            f"#SBATCH --job-name={str(self.config)}",
            f"#SBATCH --time={self.fmt_time(self.reqs.time)}",
            f"#SBATCH --mem={self.fmt_mem(self.reqs.mem)}",
        ])

        environment = "\n".join([
            f"source ./{env}/bin/activate",
        ])

        variables = "\n".join([
            f"export OMP_NUM_THREADS={self.reqs.omp_num_threads}",
            f"export PTW_NUM_THREADS={self.reqs.omp_num_threads}",
        ])

        locals = "\n".join([
            f"OUTDIR=\"{self.config.outdir}\"",
            f"CONFIG='{json.dumps(self.config.model_config)}'",
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
            f"--outdir \"$OUTDIR\"",
            f"--resume {self.config.resume} ",
            f"--design {self.config.design.value}",
            f"--arch {self.config.arch.value}",
            f"--parch {self.config.parch.value}",
            f"--posenc {self.config.posenc.value}",
            f"--patchposenc {self.config.patchposenc.value}",
            f"--model_config \"$CONFIG\"",
            f"--level {self.config.level.value}",
            f"--share_embeddings {self.config.share_embeddings}",
            f"--share_patchers {self.config.share_patchers}",
            f"--do_entropy {self.config.do_entropy}",
            f"--which_characteristics {' '.join([str(c.name) for c in self.config.which_characteristics])}",
            f"--max_length {self.config.truncate_length}",
            f"--max_length_per_structure {self.config.max_length_per_structure}",
            f"--max_structures {self.config.max_structures}",
            f"--seed {self.config.seed}",
            f"--num_workers {self.config.num_workers}",
            f"--pin_memory {True if self.config.num_workers > 0 else False}",
            f"--muddy_padded {True}",
            f"--gradient_accumulation_steps {self.config.get_gradient_accumulation_steps(self.reqs.world_size)}",
            f"--tr_batch_size {self.config.per_device_batch_size}",
            f"--vl_batch_size {self.config.per_device_batch_size}",
            f"--ts_batch_size {self.config.per_device_batch_size}",
            f"--sched {self.config.sched.value}",
            f"--lr_max {self.config.lr_max}",
            f"--lr_beg {self.config.lr_beg}",
            f"--lr_end {self.config.lr_end}",
            f"--lr_nclr_ncycles {self.config.lr_nclr_ncycles}",
            f"--lr_nclr_max_lr_gamma {self.config.lr_nclr_max_lr_gamma}",
            f"--ema_enable {self.config.ema_enable}",
            f"--ema_decay {self.config.ema_decay}",
            f"--auxillary_loss_weight {self.config.auxillary_loss_weight}",
            f"--assert_auxillary_loss {self.config.assert_auxillary_loss}",
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
            f"--enable_checkpoint {self.config.enable_checkpoint}",
            f"--enable_compile {self.config.enable_compile}",
            f"--static_shapes_bin_patcher_seq_lengths {self.config.static_shapes_bin_patcher_seq_lengths}",
            f"--static_shapes_bin_patcher_batch_sizes {self.config.static_shapes_bin_patcher_batch_sizes}",
            f"--static_shapes_bin_backbone_seq_lengths {self.config.static_shapes_bin_backbone_seq_lengths}",
            f"--static_shapes_bin_backbone_batch_sizes {self.config.static_shapes_bin_backbone_batch_sizes}",
            f"--update_embedding_steps {self.config.update_embedding_steps}",
            f"--freeze_embedding_epoch {self.config.freeze_embedding_epoch}",
            f"--param_grad_none {self.config.param_grad_none}",
            f"--find_unused_parameters {self.config.find_unused_parameters}",
            f"--max_epochs {self.config.max_epochs}",
            f"--eval_epochs {self.config.eval_epochs}",
            f"--chpt_epochs {self.config.chpt_epochs}",
            f"--logg_epochs {self.config.logg_epochs}",
            f"--tr_num_samples {self.config.tr_num_samples(self.reqs.world_size)}",
            f"--vl_num_samples {self.config.vl_num_samples(self.reqs.world_size)}",
            f"--ts_num_samples {self.config.ts_num_samples(self.reqs.world_size)}",
        ])

        command = torchrun + " \\" + "\n" + command if self.reqs.gpus_per_node > 1 else command

        script = ""
        script += shebang + "\n\n"
        script += slurm + "\n\n"
        script += environment + "\n\n"
        script += variables + "\n\n"
        script += locals + "\n\n"
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


def config_valid(config: Configuration) -> bool:
    """Return True if the configuration is conceptually valid, False otherwise."""

    # When the structure is FLAT, we exlude hierarchical levels, and vice versa.
    if config.design == Design.FLAT and config.level != HierarchicalLevel.NONE:
        return False
    if config.design != Design.FLAT and config.level == HierarchicalLevel.NONE:
        return False

    # For non-ViT architectures, these settings are irrelevant and we should just include one.
    if config.arch != Architecture.VIT:
        if config.parch != inspect.signature(MainArgs).parameters["parch"].default:
            return False
        if config.posenc != inspect.signature(MainArgs).parameters["posenc"].default:
            return False
        if config.patchposenc != inspect.signature(MainArgs).parameters["patchposenc"].default:
            return False

    return True


def config_fiter(config: Configuration) -> bool:
    """Return True if the configuration should be run, False otherwise."""

    if not config_valid(config):
        return False

    # Design
    ...

    # Architecture
    ...

    # Structure
    ...

    # Entropy
    ...

    # Characteristics
    ...

    # Model Config
    ...

    return True


def detect_system() -> Literal["mkwics", "rc", "empire"]:
    if subprocess.run(["squeue"], shell=True, capture_output=True, text=True).returncode != 0:
        return "mkwics"
    if subprocess.run(["whoami"], shell=True, capture_output=True, text=True).stdout.strip() == "lk3591":
        return "rc"
    return "empire"


def main() -> None:

    parser = ArgumentParser(description="Create large batches of experiments.")
    parser.add_argument("--debug", action="store_true", help="Configuration suitable for debugging.")
    parser.add_argument("--bench", action="store_true", help="Configuration suitable for benchmarking.")
    parser.add_argument("--system", type=str, default=None, choices=["mkwics", "rc", "empire"])
    parser.add_argument("--gpu", type=str, default="a100", choices=["a100", "h100", "gh200", "nvidia_h200", "nvidia_h100_80gb_hbm3"])
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use per job.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove existing outfiles.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing outfile.")
    args = parser.parse_args()

    global DEBUG
    global BENCH
    global NGPUS
    global SYSTEM
    global GPU
    DEBUG  = args.debug
    BENCH  = args.bench
    NGPUS  = args.ngpus
    GPU    = args.gpu
    SYSTEM = args.system if args.system is not None else detect_system()

    if SYSTEM == "rc":
        assert GPU in ("a100", "h100", "gh200"), GPU
    if SYSTEM == "empire":
        assert GPU in ("nvidia_h200", "nvidia_h100_80gb_hbm3"), GPU

    outpath = Path("./run")
    outpath.mkdir(exist_ok=True)
    for f in outpath.iterdir():
        if f.is_file() and not args.no_clean:
            f.unlink()

    DESIGNS      = [Design.STRUCTURAL]
    ARCHS        = [Architecture.VIT]
    POSENCS      = [PositionalEncodingArchitecture.FIXED]
    PATCHPOSENCS = [PatchPositionalEncodingArchitecture.NONE]
    LEVELS       = [HierarchicalLevel.FINE]
    DO_ENTROPYS  = [False]
    WHICH_CHARS  = [tuple()]
    MAX_LENGTHS  = [2**20]
    SEEDS        = [0]

    model_configs = [
        # {"patcher_pooling": "atn", "patcher_channels": 64, "patcher_depth": 2, "patcher_kernel_size": 64,  "patcher_stride": 64},
        {"patcher_pooling": "avg", "patcher_channels": 64, "patcher_depth": 2, "patcher_kernel_size": 64,  "patcher_stride": 64},
    ]
    stream = product(
        DESIGNS,
        ARCHS,
        [PatcherArchitecture.DWC],
        POSENCS,
        PATCHPOSENCS,
        LEVELS,
        DO_ENTROPYS,
        WHICH_CHARS,
        MAX_LENGTHS,
        SEEDS,
        model_configs,
    )

    configurations = (Configuration(*config) for config in stream)  # type: ignore[arg-type]
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
            print(f"WARNING ({str(config)}): output directory exists.")

        builder = ScriptBuilder(config, reqs)
        outfile = (outpath / str(config)).with_suffix(".sh")
        script = builder()
        if outfile.exists() and args.no_overwrite:
            pass
        else:
            outfile.write_text(script)
            print(f"{outfile}")


if __name__ == "__main__":
    main()
