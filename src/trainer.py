"""
Train and validation loops.
"""

from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import contextlib
from dataclasses import dataclass
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import threading
import time
from typing import Any
from typing import Callable
from typing import ContextManager
from typing import Optional
from typing import Protocol
from typing import Self
from typing import Sequence
import warnings

import numpy as np
import psutil
import pynvml
import torch
from torch import Tensor
from torch import distributed as dist
from torch.amp import GradScaler
from torch.distributed import checkpoint as dcp
from torch.distributed.algorithms.join import Join
from torch.distributed.algorithms.join import Joinable
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.checkpoint.state_dict import set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.tensor import DTensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel import DataParallel
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# Enable detailed timing statistics if the environment variable is set. This will muddy the logs
# with extra timing information, but can help diagnose bottlenecks in the data loading.
# Note that these statistics are very biased since they add synchronization points that impact
# performance significantly. Nonetheless, they give a rough idea of where time is being spent.
DETAILED_TIMING_STATISTICS = os.environ.get("DETAILED_TIMING_STATISTICS", "0") == "1"
if DETAILED_TIMING_STATISTICS:
    warnings.warn("Detailed timing statistics enabled; this may impact performance.", RuntimeWarning)


def _syncronize_if_detailed_timing() -> None:
    if DETAILED_TIMING_STATISTICS:
        torch.cuda.synchronize()


def debug_autocast_dtypes(model: nn.Module) -> None:
    def hook(mod, inp, out):  # type: ignore[no-untyped-def]
        if isinstance(out, torch.Tensor):
            print(f"{mod.__class__.__name__}: in={inp[0].dtype}, out={out.dtype}")
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            m.register_forward_hook(hook)


class Batch(Protocol):
    """
    Structural type for a batch of samples.

    CUDA, PyTorch, and Python are a little weird and picky about when, where, and how
    data is moved to/from the GPU and doing things incorrectly can hurt memory utilization.

    For example, if the Iterable[Batch] yields a Batch that is already on the GPU, then
    Python cannot properly garbage collect the previous Batch until the new Batch is fully
    materialized on the GPU. Using `del batch` or `torch.cuda.empty_cache()` inside the training
    loop does nothing to properly decrement the reference count of the previous Batch and
    thus does not free GPU memory as expected. Therefore, we wind up with extra batches on
    the GPU. CUDA, being asynchronous, does not free the memory from the previous Batch until
    the next-next batch. So we wind up with three batches on the GPU at once instead of just one.
    """

    def __len__(self) -> int:
        """
        Return the number of samples in the batch.
        """
        ...

    def to(self, device: torch.device, non_blocking: bool) -> Self:
        """
        Move the batch to the specified device.
        """
        ...

    def finalize(self, ftype: torch.dtype, itype: torch.dtype, ltype: torch.dtype) -> Self:
        """
        Finalize the batch for use within the network.
        """
        ...

    def clone(self) -> Self:
        """
        Return a deep copy of the batch.
        """
        ...

    @property
    def label(self) -> Tensor:
        ...

    @property
    def inputs(self) -> Any:
        ...

    @property
    def allguides(self) -> Any:
        ...


def is_dist() -> bool:
    """Return True if in a distributed training environment else False."""
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)


def local_rank() -> int:
    """Return the node-local rank of the current process."""
    return int(os.environ.get("LOCAL_RANK", 0))


def rank() -> int:
    """Return the global rank of the current process."""
    if is_dist():
        return int(dist.get_rank())
    return 0


def local_world_size() -> int:
    """Return the node-local world size."""
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if is_dist():
        return int(torch.cuda.device_count())
    return 1


def world_size() -> int:
    """Return the global world size."""
    if dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def maybe_join(model: Module, join: bool = True) -> ContextManager[None] | Join:
    """Return a Join context if in distributed the model is Joinable, else a no-op context."""
    if join and is_dist() and isinstance(model, Joinable):
        return Join([model])
    return contextlib.nullcontext()


def maybe_no_sync(model: Module, sync_now: bool) -> ContextManager[None]:
    """Return a no_sync context if in distributed and the model has a no_sync method, else a no-op context."""
    if not is_dist() or sync_now or not hasattr(model, "no_sync"):
        return contextlib.nullcontext()
    return model.no_sync()  # type: ignore[no-any-return,operator]


def barrier(tag: str = "", device: Optional[torch.device] = None) -> None:
    if not is_dist():
        return
    try:
        if "nccl" in dist.get_backend():
            torch.cuda.synchronize(device)
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception as e:
        print(f"[rank {rank()}] barrier({tag}) failed: {e}", flush=True)
        raise


@dataclass
class TrainerArgs:
    outdir: Path = Path("./output/tmp")
    disable_tqdm: bool = False
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mp16: bool = False
    max_epochs: Optional[float] = 1.0
    max_steps: Optional[int] = None
    eval_epochs: Optional[float] = 1.0
    eval_steps: Optional[int] = None
    chpt_epochs: Optional[float] = 1.0
    chpt_steps: Optional[int] = None
    schd_epochs: Optional[float] = 1.0
    schd_steps: Optional[int] = None
    logg_epochs: Optional[float] = 1.0
    logg_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.max_epochs is not None) == (self.max_steps is not None):
            raise ValueError("Exactly one of `max_epochs` or `max_steps` must be specified.")
        if (self.eval_epochs is not None) and (self.eval_steps is not None):
            raise ValueError("At most one of `eval_epochs` or `eval_steps` may be specified.")
        if (self.chpt_epochs is not None) and (self.chpt_steps is not None):
            raise ValueError("At most one of `chpt_epochs` or `chpt_steps` may be specified.")
        if (self.schd_epochs is not None) and (self.schd_steps is not None):
            raise ValueError("At most one of `schd_epochs` or `schd_steps` may be specified.")
        if (self.logg_epochs is not None) and (self.logg_steps is not None):
            raise ValueError("At most one of `logg_epochs` or `logg_steps` may be specified.")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls.from_dict(vars(namespace))


class EarlyStopper:

    def __init__(self, patience: int | float = 0, threshold: float = 0.0001, lower_is_worse: bool = False) -> None:
        self.patience = patience
        self.threshold = threshold
        self.lower_is_worse = lower_is_worse
        self.best = -float("inf") if lower_is_worse else float("inf")
        self.current = -1.0
        self.count = 0

    def step(self, val: float) -> Self:
        self.current = val
        if self.lower_is_worse and (self.current > self.best + self.threshold):
            self.best = self.current
            self.count = 0
        elif not self.lower_is_worse and (self.current < self.best - self.threshold):
            self.best = self.current
            self.count = 0
        return self

    @property
    def stop(self) -> bool:
        if self.current == self.best:
            return False
        self.count += 1
        return self.count >= self.patience


def mp_dtype(mp16: bool, device: torch.device) -> torch.dtype:
    if not mp16:
        return torch.float32
    if device.type == "cpu":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported(False):
            return torch.bfloat16
        if torch.cuda.is_bf16_supported(True):
            return torch.float16
        if not torch.cuda.is_available():
            return torch.float16
    raise RuntimeError(f"Unsupported device for mixed precision: {device}.")


def unwrapddp(model: Module) -> Module:
    if isinstance(model, DistributedDataParallel):
        return model.module  # type: ignore[no-any-return]
    return model


def is_fsdp2(model: Module) -> bool:
    return any(isinstance(p, DTensor) for p in model.parameters())


# Source: torcheval.metrics.functional.classification.auroc
@torch.jit.script
def _binary_auroc_compute_jit(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=-1) != 0, [0, 1], value=1.0)
    sorted_target = torch.gather(target, -1, indices)
    sorted_weight = (
        torch.tensor(1.0, device=target.device)
        if weight is None
        else torch.gather(weight, -1, indices)
    )
    cum_tp_before_pad = (sorted_weight * sorted_target).cumsum(-1)
    cum_fp_before_pad = (sorted_weight * (1 - sorted_target)).cumsum(-1)

    shifted_mask = mask.sum(-1, keepdim=True) >= torch.arange(
        mask.size(-1), 0, -1, device=target.device
    )

    cum_tp = torch.zeros_like(cum_tp_before_pad)
    cum_fp = torch.zeros_like(cum_fp_before_pad)

    cum_tp.masked_scatter_(shifted_mask, cum_tp_before_pad[mask])
    cum_fp.masked_scatter_(shifted_mask, cum_fp_before_pad[mask])

    if len(mask.shape) > 1:
        factor = cum_tp[:, -1] * cum_fp[:, -1]
    else:
        factor = cum_tp[-1] * cum_fp[-1]
    # Set AUROC to 0.5 when the target contains all ones or all zeros.
    auroc = torch.where(
        factor == 0,
        0.5,
        torch.trapz(cum_tp, cum_fp).double() / factor,
    )
    return auroc


class Trainer:
    """
    Trainer class for training models with PyTorch.

    Usage
    -----
    It is not recommended to use trainer.train() or trainer.evaluate() directly, as these
    have complex side effects on the Trainer's internal state. Instead, use Trainer.__call__().
    >>> trainer = Trainer(...)
    >>> trainer = trainer()
    >>> print(trainer.best_epoch, trainer.best_metric)

    NOTE
    -----
    - If the GradScaler causes all optimization steps to be skipped, the LR scheduler will trigger a
        warning that it was called before optimizer.step().

    TODO
    ----
    - Adjust the progress bars to display the longest running rank automatically.
    """

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_loader: Collection[Batch] | DataLoader[Batch],
        vl_loader: Collection[Batch] | DataLoader[Batch],
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
        device: Optional[torch.device] = None,
        padbatch: Optional[Batch] = None,
    ) -> None:
        self.args = args
        self.model = model
        self.tr_loader = tr_loader
        self.vl_loader = vl_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer if optimizer is not None else AdamW(model.parameters())
        self.scheduler = scheduler if scheduler is not None else LambdaLR(self.optimizer, lambda _: 1.0)
        self.stopper = stopper if stopper is not None else EarlyStopper(patience=float("inf"))
        self.device = device if device is not None else next(self.model.parameters()).device
        self.padbatch = padbatch.to(self.device, non_blocking=True).finalize(self.mp_dtype, torch.int32, torch.int64) if padbatch is not None else None
        self.monitor = Monitor(device=self.device)
        self.log: list[Mapping[str, int | float]] = []
        self.glbl_step = 0
        self._next_eval_step: Optional[int] = None
        self._next_chpt_step: Optional[int] = None
        self._next_schd_step: Optional[int] = None
        self._next_logg_step: Optional[int] = None
        self._done: bool = False

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  args={self.args.__class__.__name__}(...),\n"
            f"  model={self.model.__class__.__name__}(...),\n"
            f"  tr_loader={self.tr_loader.__class__.__name__}(...),\n"
            f"  vl_loader={self.vl_loader.__class__.__name__}(...),\n"
            f"  loss_fn={self.loss_fn.__class__.__name__}(...),\n"
            f"  optimizer={self.optimizer.__class__.__name__}(...),\n"
            f"  scheduler={self.scheduler.__class__.__name__}(...),\n"
            f"  stopper={self.stopper.__class__.__name__}(...),\n"
            f"  device={self.device},\n"
            f")"
        )

    def __call__(self) -> Self:
        if self.glbl_step < 0:
            raise RuntimeError("glbl_step must be non-negative.")

        self.args.outdir.mkdir(parents=True, exist_ok=True)
        self.monitor.start()

        # Initialize hook schedules.
        if self.eval_steps is not None:
            self._next_eval_step = self.glbl_step + self.eval_steps
        if self.chpt_steps is not None:
            self._next_chpt_step = self.glbl_step + self.chpt_steps
        if self.schd_steps is not None:
            self._next_schd_step = self.glbl_step + self.schd_steps
        if self.logg_steps is not None:
            self._next_logg_step = self.glbl_step + self.logg_steps

        # Conduct an initial validation and checkpointing on the naked model.
        if self.glbl_step == 0:
            self.run_due_hooks(None, do_eval=True, do_chpt=True, do_schd=False, do_logg=True)

        # Continuously train the model on the training set until finished.
        while not self._done:
            self.train()

        self.monitor.stop()

        return self

    def train(self) -> None:
        """
        Train the model on the training set.
        """
        barrier("Trainer::train:before", self.device)

        self.optimizer.zero_grad()
        self.model.train()
        scaler = GradScaler(self.device.type, enabled=self.mp_dtype == torch.float16)

        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, Batch]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Training...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)
        iterable = iter(iterable)

        # If distributed, pad the dataloader so all ranks have the same number of batches.
        if is_dist():
            assert self.padbatch is not None
            length = torch.tensor(len(dataloader), device=self.device)
            longest = length.clone()
            dist.all_reduce(longest, op=dist.ReduceOp.MAX, group=None)
            padding = itertools.repeat(self.padbatch, int(longest.item() - length.item()))
            iterable = itertools.chain(iterable, enumerate(padding, start=len(dataloader)))

        def step() -> None:
            """Update the model parameters and increment the global step counter."""
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            self.glbl_step += 1

        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
        t_0: float = time.time()
        def report() -> dict[str, float]:
            """Assemble a training report (excluding GPU statistics) and reset the `results` container."""
            nonlocal results
            nonlocal t_0
            # Aggregate results across workers and move to CPU.
            if is_dist():
                results = self.reduce_results(results)
            results = {k: results[k].detach().to("cpu") for k in results}
            num_samples = int(results.pop("num_samples").item())
            # Average statistics over total number of samples.
            report = {}
            for k in results:
                report[k] = results[k].item() / num_samples
            report["tr_time"] = time.time() - t_0
            report["tr_samples"] = num_samples
            report["tr_throughput"] = num_samples / report["tr_time"]
            # Clear results container for next cycle before returning and reset the timer.
            results = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
            t_0 = time.time()
            return report


        #### DETAILED TIMING STATISTICS ####
        t_detailed: float = time.time()
        def t_detailed_log(t_detailed_list: list[float]) -> None:
            if not DETAILED_TIMING_STATISTICS:
                return
            nonlocal t_detailed
            torch.cuda.synchronize()
            t_detailed_list.append(time.time() - t_detailed)
            t_detailed = time.time()
        t_detailed_preps: list[float] = []  # time to prepare
        t_detailed_trans: list[float] = []  # time to transfer
        t_detailed_final: list[float] = []  # time to finalize
        t_detailed_comps: list[float] = []  # time to compute
        t_detailed_steps: list[float] = []  # time to step
        ####################################


        mini_step = -1  # mini-step within this epoch
        grda_modl = 0   # gradient accumulation modulo local steps
        for mini_step, batch in iterable:
            t_detailed_log(t_detailed_preps)

            batch = batch.to(self.device, non_blocking=True)
            t_detailed_log(t_detailed_trans)

            batch = batch.finalize(self.mp_dtype, torch.int32, torch.int64)
            t_detailed_log(t_detailed_final)

            # Determine if this is a real or padded batch of data.
            real = mini_step < len(dataloader)
            is_last_real = (mini_step + 1 == len(dataloader))
            results["num_samples"] += len(batch) * int(real)

            # Sync on gradient accumulation boundary or final real mini-step
            grda_modl = (grda_modl + 1) % self.args.gradient_accumulation_steps
            sync_gradients = grda_modl == 0 or is_last_real

            # Compute normalized loss
            with maybe_no_sync(self.model, sync_gradients):
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                loss = loss * int(real) / self.args.gradient_accumulation_steps
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                results["tr_loss"] += loss.detach() * len(batch) * self.args.gradient_accumulation_steps
            t_detailed_log(t_detailed_comps)

            # Update model weights and possibly run hooks (validation, checkpointing, etc.)
            if sync_gradients:
                step()
                t_detailed_log(t_detailed_steps)

                if is_last_real and grda_modl != 0:
                    grda_modl = 0
                do_log = any(self._due_hooks())
                do_eval = self._due_hooks()[0]
                # Freeing up memory before running the validation cycle seems to keep GPU memory usage lower
                # across all subsequent training and validation phases (yes, both of them). Why? Don't know!
                if do_eval:
                    del batch, outputs, loss
                    gc.collect()
                self.run_due_hooks(report() if do_log else None)
                self.model.train()
                if do_log:
                    self.monitor.clear()

            # Stop if we've reached the maximum number of global steps
            if self.glbl_step >= self.max_steps:
                self._done = True
                break

            t_detailed = time.time()

        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")

        #### DETAILED TIMING STATISTICS ####
        def times_to_stats(times: list[float], skip: int) -> tuple[float, float, float, float, float]:
            """Convert a list of times to (throughput, min, max, median, average).
            Skip the first `gradient_accumulation_steps` mini-steps to allow for warmup.
            This will not work correctly for edge cases, e.g., distributed training, etc.
            """
            num_samples = results["num_samples"].item() - (len(batch) * self.args.gradient_accumulation_steps)
            times = times[skip:]
            return (
                float(num_samples / np.sum(times)),
                float(np.min(times)),
                float(np.max(times)),
                float(np.median(times)),
                float(np.mean(times)),
            )
        if DETAILED_TIMING_STATISTICS:
            num_samples = results["num_samples"].item() - (len(batch) * self.args.gradient_accumulation_steps)
            t_detailed_preps = list(times_to_stats(t_detailed_preps, self.args.gradient_accumulation_steps))
            t_detailed_trans = list(times_to_stats(t_detailed_trans, self.args.gradient_accumulation_steps))
            t_detailed_final = list(times_to_stats(t_detailed_final, self.args.gradient_accumulation_steps))
            t_detailed_comps = list(times_to_stats(t_detailed_comps, self.args.gradient_accumulation_steps))
            t_detailed_steps = list(times_to_stats(t_detailed_steps, 1))
            print(
                f"[rank {rank()}] Detailed timing statistics for training run ({num_samples=} {mini_step=} glbl_step={self.glbl_step}):\n"
                f"  phase    throughput latency\n"
                f"  prepare  {t_detailed_preps[0]:.2f}     {t_detailed_preps[4]:.2f}\n"
                f"  transfer {t_detailed_trans[0]:.2f}     {t_detailed_trans[4]:.2f}\n"
                f"  finalize {t_detailed_final[0]:.2f}     {t_detailed_final[4]:.2f}\n"
                f"  compute  {t_detailed_comps[0]:.2f}     {t_detailed_comps[4]:.2f}\n"
                f"  optimize {t_detailed_steps[0]:.0f}     {t_detailed_steps[4]:.2f}\n"
            )

        barrier("Trainer::train:after", self.device)

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation set.
        """
        barrier("Trainer::evaluate:before", self.device)
        t_0 = time.time()

        was_training = self.model.training
        self.model.eval()

        dataloader = self.vl_loader
        iterable: Iterable[tuple[int, Batch]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Validating...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)
        iterable = iter(iterable)

        # If distributed, pad the dataloader so all ranks have the same number of batches.
        if is_dist():
            assert self.padbatch is not None
            length = torch.tensor(len(dataloader), device=self.device)
            longest = length.clone()
            dist.all_reduce(longest, op=dist.ReduceOp.MAX, group=None)
            padding = itertools.repeat(self.padbatch, int(longest.item() - length.item()))
            iterable = itertools.chain(iterable, enumerate(padding, start=len(dataloader)))

        mini_step = -1
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
        alllabels: list[Tensor] = []
        alllogits: list[Tensor] = []

        with torch.no_grad():
            for mini_step, batch in iterable:
                batch = batch.to(self.device, non_blocking=True).finalize(self.mp_dtype, torch.int32, torch.int64)
                real = mini_step < len(dataloader)
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                if real:
                    results["num_samples"] += len(batch)
                    results["vl_loss"] += loss * len(batch)
                    alllabels.append(batch.label)
                    alllogits.append(outputs)

        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")

        labels = torch.cat(alllabels, dim=0)
        logits = torch.cat(alllogits, dim=0)

        # Weight-average the metrics in this rank by the number of valid samples
        metrics = self.compute_metrics(labels, logits)
        for k, v in metrics.items():
            results[f"vl_{k}"] = v.detach() * labels.shape[0]

        # Aggregate results across workers and/or move to CPU
        if is_dist():
            results = self.reduce_results(results)
        else:
            results = {k: results[k].detach().to("cpu") for k in results}
        num_samples = int(results.pop("num_samples").item())

        # Average statistics over total number of samples
        report = {}
        for k in results:
            report[k] = results[k].item() / num_samples
        report["vl_time"] = time.time() - t_0
        report["vl_samples"] = num_samples
        report["vl_throughput"] = num_samples / report["vl_time"]

        if was_training:
            self.model.train()

        barrier("Trainer::evaluate:after", self.device)
        return report

    def _due_hooks(self) -> tuple[bool, bool, bool, bool]:
        do_eval = self._next_eval_step is not None and self.glbl_step >= self._next_eval_step
        do_chpt = self._next_chpt_step is not None and self.glbl_step >= self._next_chpt_step
        do_schd = self._next_schd_step is not None and self.glbl_step >= self._next_schd_step
        do_logg = self._next_logg_step is not None and self.glbl_step >= self._next_logg_step
        return do_eval, do_chpt, do_schd, do_logg

    def run_due_hooks(
        self,
        tr_report: Optional[dict[str, float]],
        *,
        do_eval: Optional[bool] = None,
        do_chpt: Optional[bool] = None,
        do_schd: Optional[bool] = None,
        do_logg: Optional[bool] = None,
    ) -> None:
        """
        Conduct periodic activities, such as running validation and checkpointing.

        Args:
            tr_report: Optional dictionary of training statistics to include in the report.
            do_eval: If True, run validation even if not scheduled.
            do_chpt: If True, save a checkpoint even if not scheduled.
            do_schd: If True, step the LR scheduler even if not scheduled.
            do_logg: If True, update the console and log files even if not scheduled.

        NOTE: the schedules are only advanced if fired organically, not via the `do_*` arguments.
        NOTE: logging will occur if any hook fires, but will not advance the logging schedule.
        """
        due_eval, due_chpt, due_schd, due_logg = self._due_hooks()

        do_eval = do_eval if do_eval is not None else due_eval
        do_chpt = do_chpt if do_chpt is not None else due_chpt
        do_schd = do_schd if do_schd is not None else due_schd
        do_logg = do_logg if do_logg is not None else due_logg

        # If no hooks are scheduled, exit prematurely.
        if not (do_eval or do_chpt or do_schd or do_logg):
            return

        # Otherwise, assemble report to log.
        report = {"epoch": self.glbl_step / self.steps_per_epoch, "glbl_step": self.glbl_step, "lr": self.scheduler.get_last_lr()[0]}
        if tr_report is not None:
            report |= tr_report
            report |= {f"tr_{k}": v for k, v in self.get_monitor_report().items()}

        # If scheduled, conduct validation and add to the report.
        if do_eval:
            self.monitor.clear()
            report |= self.evaluate()
            report |= {f"vl_{k}": v for k, v in self.get_monitor_report().items()}

        # If scheduled, save a checkpoint.
        if do_chpt:
            self.to_checkpoint(self.args.outdir / f"checkpoint-{self.glbl_step}")

        # If scheduled, step the LR scheduler.
        if do_schd:
            self.scheduler.step()

        # Update console and log files.
        self._update_cons(report)
        self._update_logs(report)

        # Advance schedules that fired.
        if due_eval:
            self._next_eval_step += self.eval_steps  # type: ignore[operator]
        if due_chpt:
            self._next_chpt_step += self.chpt_steps  # type: ignore[operator]
        if due_schd:
            self._next_schd_step += self.schd_steps  # type: ignore[operator]
        if due_logg:
            self._next_logg_step += self.logg_steps  # type: ignore[operator]

        # Handle early stopping (only if validation is orgnanically scheduled).
        if self.stopper is not None:
            if due_eval:
                self.stopper.step(report[f"{self.args.metric}"])
            if self.stopper.stop:
                self._done = True

    def forward(self, batch: Batch) -> Tensor:
        """
        Send a batch of inputs forward through the model.
        """
        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            return self.model(batch.inputs, batch.allguides)  # type: ignore[no-any-return]

    def compute_loss(self, batch: Batch, outputs: Tensor) -> Tensor:
        """
        Compute the loss over a batch of examples.
        """
        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            return self.loss_fn(outputs, batch.label)  # type: ignore[no-any-return]

    def reduce_results(self, results: dict[str, Tensor], check: bool = True) -> dict[str, Tensor]:
        """
        Reduce a results dictionary across all distributed ranks.
        """
        keys = sorted(results.keys())

        # Verify all ranks have the same keys using a hash signature.
        if check:
            sig_bytes = hashlib.sha1(",".join(keys).encode()).digest()[:8]
            sig = int.from_bytes(sig_bytes, "big", signed=True)
            sig_t = torch.tensor([sig], dtype=torch.int64, device=self.device)
            sig_min = sig_t.clone()
            sig_max = sig_t.clone()
            dist.all_reduce(sig_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(sig_max, op=dist.ReduceOp.MAX)
            if sig_min.item() != sig_max.item():
                raise RuntimeError("Metrics keys do not match across ranks.")

        # Sum-reduce values across workers
        vals = torch.stack([results[k].to(self.device, dtype=torch.float64) for k in keys], dim=0)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        return {k: vals[i] for i, k in enumerate(keys)}

    def compute_metrics(self, labels: Tensor, logits: Tensor) -> dict[str, Tensor]:
        """
        Compute binary classification metrics.
        """
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = (probs >= 0.5).to(torch.int32)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        acc = (tp + tn) / (tp + tn + fp + fn).clamp_min_(1)
        pre = tp / (tp + fp).clamp_min_(1)
        rec = tp / (tp + fn).clamp_min_(1)
        f_1 = 2 * pre * rec / (pre + rec).clamp_min_(1)
        roc = _binary_auroc_compute_jit(probs, labels)

        return {
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f-1": f_1,
            "roc": roc,
        }

    def _dataloader_lengths(self, dataloader: Collection[Batch] | DataLoader[Batch]) -> list[int]:
        """Return the lengths of the dataloader on each distributed rank."""
        if not is_dist():
            return [len(dataloader)]
        length = torch.tensor(len(dataloader), device=self.device)
        lengths = [torch.zeros(1, dtype=length.dtype, device=self.device) for _ in range(world_size())]
        dist.all_gather(lengths, length, group=None)
        return [int(l.item()) for l in lengths]

    @property
    def mp_dtype(self) -> torch.dtype:
        return mp_dtype(self.args.mp16, self.device)

    @property
    def tr_dataloader_lengths(self) -> list[int]:
        return self._dataloader_lengths(self.tr_loader)

    @property
    def vl_dataloader_lengths(self) -> list[int]:
        return self._dataloader_lengths(self.vl_loader)

    @property
    def steps_per_epoch(self) -> int:
        length = max(self.tr_dataloader_lengths)
        return math.ceil(length / self.args.gradient_accumulation_steps)

    def _epochs_to_steps(self, f: Optional[float]) -> Optional[int]:
        if f is None:
            return None
        return max(1, math.ceil(self.steps_per_epoch * f))

    @property
    def max_steps(self) -> int:
        if self.args.max_steps is not None:
            return self.args.max_steps
        assert self.args.max_epochs is not None
        max_steps = self._epochs_to_steps(self.args.max_epochs)
        assert max_steps is not None
        return max_steps

    @property
    def eval_steps(self) -> Optional[int]:
        if self.args.eval_steps is not None:
            return max(1, self.args.eval_steps)
        return self._epochs_to_steps(self.args.eval_epochs)

    @property
    def chpt_steps(self) -> Optional[int]:
        if self.args.chpt_steps is not None:
            return max(1, self.args.chpt_steps)
        return self._epochs_to_steps(self.args.chpt_epochs)

    @property
    def schd_steps(self) -> Optional[int]:
        if self.args.schd_steps is not None:
            return max(1, self.args.schd_steps)
        return self._epochs_to_steps(self.args.schd_epochs)

    @property
    def logg_steps(self) -> Optional[int]:
        if self.args.logg_steps is not None:
            return max(1, self.args.logg_steps)
        return self._epochs_to_steps(self.args.logg_epochs)

    def get_monitor_report(self) -> dict[str, float]:
        mn_report = self.monitor.get_report(clear=True)
        mn_report = {k: torch.tensor(v, device=self.device, dtype=torch.float64) for k, v in mn_report.items()}
        mn_report = self.reduce_results(mn_report) if is_dist() else mn_report
        mn_report = {k: v.detach().to("cpu").item() for k, v in mn_report.items()}
        return mn_report

    def _update_logs(self, results: Mapping[str, int | float]) -> None:
        if rank() != 0:
            return
        self.log.append(results)
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")

    def _update_cons(self, results: Mapping[str, int | float]) -> None:
        if rank() != 0:
            return
        d = {}
        d["epoch"] = round(results["epoch"], 2)
        d["glbl_step"] = int(results["glbl_step"])
        d["lr"] = round(results["lr"], 6)
        if any(k.startswith("tr_") for k in results):
            d["tr_loss"] = round(results["tr_loss"], 3)
            d["tr_gpu_utl"] = round(results["tr_gpu_utl"], 2)
            d["tr_gpu_mem"] = round(results["tr_gpu_mem"] / (1024 ** 3), 2)
            d["tr_time"] = round(results["tr_time"], 0)
            d["tr_samples"] = int(results["tr_samples"])
            d["tr_throughput"] = round(results["tr_throughput"], 2)
        if any(k.startswith("vl_") for k in results):
            d["vl_loss"] = round(results["vl_loss"], 3)
            d["vl_acc"] = round(results["vl_acc"], 3)
            d["vl_gpu_utl"] = round(results["vl_gpu_utl"], 2)
            d["vl_gpu_mem"] = round(results["vl_gpu_mem"] / (1024 ** 3), 2)
            d["vl_time"] = round(results["vl_time"], 0)
            d["vl_samples"] = int(results["vl_samples"])
            d["vl_throughput"] = round(results["vl_throughput"], 2)
        print(d)

    def to_checkpoint(self, path: str | Path) -> None:
        """
        Save a checkpoint of the trainer state to the specified path.

        Structure:
            path
            ├── meta.json
            ├── args.pickle
            ├── log.pickle
            ├── stopper.pickle
            ├── rng-cpu.pt
            ├── rng-gpu.pt
            ├── padbatch.pt
            ├── loss_fn.pt
            ├── scheduler.pt
            ├── model.pt         # non-FSDP only
            ├── optimizer.pt     # non-FSDP only
            ├── model/           # FSDP only
            └── optimizer/       # FSDP only
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Rank 0 saves everything except model and optimizer state dicts.
        if rank() == 0:
            # JSON-serializable metadata
            meta = {
                "glbl_step": self.glbl_step,
                "_next_eval_step": self._next_eval_step,
                "_next_chpt_step": self._next_chpt_step,
                "_next_schd_step": self._next_schd_step,
                "_next_logg_step": self._next_logg_step,
            }
            Path(path / "meta.json").write_text(json.dumps(meta, indent=2))
            # Pickled objects
            Path(path / "args.pickle").write_bytes(pickle.dumps(self.args))
            Path(path / "log.pickle").write_bytes(pickle.dumps(self.log))
            Path(path / "stopper.pickle").write_bytes(pickle.dumps(self.stopper))
            # Torch objects
            torch.save(torch.random.get_rng_state(), path / "rng-cpu.pt")
            torch.save(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, path / "rng-gpu.pt")
            torch.save(self.padbatch, path / "padbatch.pt")
            # Torch state dicts
            torch.save(self.loss_fn.state_dict(), path / "loss_fn.pt")
            torch.save(self.scheduler.state_dict(), path / "scheduler.pt")

        # All ranks save model and optimizer state dicts for FSDP.
        if is_fsdp2(self.model):
            mstate, ostate = get_state_dict(self.model, self.optimizer)
            dcp.save(mstate, checkpoint_id=path / "model")
            dcp.save(ostate, checkpoint_id=path / "optimizer")
        # Only rank 0 saves model and optimizer state dicts for non-FSDP.
        elif rank() == 0:
            mstate, ostate = unwrapddp(self.model).state_dict(), self.optimizer.state_dict()
            torch.save(mstate, path / "model.pt")
            torch.save(ostate, path / "optimizer.pt")

        if is_dist():
            dist.barrier(device_ids=[torch.cuda.current_device()])

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        model: Module,
        wrap_model: Callable[[Module], Module],
        tr_loader: Collection[Batch] | DataLoader[Batch],
        vl_loader: Collection[Batch] | DataLoader[Batch],
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        optimizer_init: Optional[Callable[[Iterable[torch.nn.Parameter]], Optimizer]] = None,
        scheduler: Optional[LRScheduler] = None,
        scheduler_init: Optional[Callable[[Optimizer], LRScheduler]] = None,
        device: Optional[torch.device] = None,
    ) -> Trainer:
        """
        Load a Trainer instance from a checkpoint.

        NOTE: `model`, `loss_fn`, `optimizer`, and `scheduler` must all be the same type
        as those used to create the checkpoint; their prior state dicts will be loaded from disk.
        The initializers `optimizer_init` and `scheduler_init` must return the same type as well.
        `device` must correspond to the device of which `wrap_model` will move the model to.
        """
        if isinstance(model, (FullyShardedDataParallel, DistributedDataParallel, DataParallel)) or is_fsdp2(model) or next(model.parameters()).is_cuda:
            raise RuntimeError("Trainer.from_checkpoint requires a non-distributed model instance on the cpu.")

        path = Path(path)

        # Determine if we're loading from a DCP/FSDP checkpoint or a regular checkpoint.
        if (path / "model").is_dir() and (path / "optimizer").is_dir():
            is_dcp = True
        elif (path / "model.pt").is_file() and (path / "optimizer.pt").is_file():
            is_dcp = False
        else:
            raise RuntimeError(f"Invalid checkpoint path: {path}.")

        # Warn if custom optimizer or scheduler will be ignored.
        if is_dcp and optimizer is not None and optimizer_init is None:
            warnings.warn("A custom optimizer was provided but will be ignored since the checkpoint was saved with FSDP.")
        if is_dcp and scheduler is not None and scheduler_init is None:
            warnings.warn("A custom scheduler was provided but will be ignored since the checkpoint was saved with FSDP.")

        # Establish default optimizer and scheduler if not provided (defaults should match __init__).
        if optimizer_init is None:
            optimizer_init = lambda params: AdamW(params)
        if optimizer is None and not is_dcp:
            optimizer = optimizer_init(model.parameters())
        if scheduler_init is None:
            scheduler_init = lambda optim: LambdaLR(optim, lambda _: 1.0)
        if scheduler is None and not is_dcp:
            assert optimizer is not None
            scheduler = scheduler_init(optimizer)

        # Load the simple objects first that don't require complex state dict handling.
        meta = json.loads((path / "meta.json").read_text())
        glbl_step = meta["glbl_step"]
        _next_eval_step = meta["_next_eval_step"]
        _next_chpt_step = meta["_next_chpt_step"]
        _next_schd_step = meta["_next_schd_step"]
        _next_logg_step = meta["_next_logg_step"]
        args = pickle.loads((path / "args.pickle").read_bytes())
        log = pickle.loads((path / "log.pickle").read_bytes())
        stopper = pickle.loads((path / "stopper.pickle").read_bytes())
        if (rng_cpu := torch.load(path / "rng-cpu.pt", map_location="cpu")) is not None:
            torch.random.set_rng_state(rng_cpu)
        if (rng_gpu := torch.load(path / "rng-gpu.pt", map_location="cpu")) is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_gpu)
        padbatch = torch.load(path / "padbatch.pt", map_location="cpu", weights_only=False)  # Load custom types (unsafe)
        loss_fn.load_state_dict(torch.load(path / "loss_fn.pt", map_location="cpu"))

        if is_dcp:
            # DCP/FSDP checkpoints require special handling to load the model and optimizer state dicts.
            # Concretely, the model must first be wrapped appropriately, before the optimizer is initialized.
            # Then both their state dicts can be loaded and from the checkpoint.
            model = wrap_model(model)
            optimizer = optimizer_init(model.parameters())
            mstate, ostate = get_state_dict(model, optimizer)
            dcp.load(mstate, checkpoint_id=path / "model")
            dcp.load(ostate, checkpoint_id=path / "optimizer")
            set_state_dict(model, optimizer, model_state_dict=mstate, optim_state_dict=ostate)
        else:
            # Regular/DDP checkpoints can just be loaded in standard fashion.
            assert optimizer is not None
            mstate = torch.load(path / "model.pt", map_location="cpu")
            set_model_state_dict(model, mstate)
            model = wrap_model(model)
            ostate = torch.load(path / "optimizer.pt", map_location="cpu")
            optimizer.load_state_dict(ostate)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device if device is not None else "cpu")

        scheduler = scheduler_init(optimizer)
        if scheduler is not None:
            sstate = torch.load(path / "scheduler.pt", map_location="cpu")
            scheduler.load_state_dict(sstate)

        trainer = cls(
            args,
            model,
            tr_loader,
            vl_loader,
            loss_fn,
            optimizer,
            scheduler,
            stopper,
            device,
            padbatch,
        )
        trainer.glbl_step = glbl_step
        trainer._next_eval_step = _next_eval_step
        trainer._next_chpt_step = _next_chpt_step
        trainer._next_schd_step = _next_schd_step
        trainer._next_logg_step = _next_logg_step
        trainer.log = log

        return trainer


class Monitor:
    """
    Basic CPU/GPU monitoring.

    Collects:
        - cpu_utl: CPU utilization as a proportion, e.g., 0.50 = 50% of one CPU
        - cpu_mem: CPU memory used in bytes
        - gpu_utl: GPU utilization as a proportion, e.g., 0.50 = 50% of one GPU
        - gpu_mem: GPU memory used in bytes
        - h_to_d: Host to Device PCIe throughput in bytes/second
        - d_to_h: Device to Host PCIe throughput in bytes/second

    See: https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
    """

    COLLECT = (
        "cpu_utl",
        "cpu_mem",
        "gpu_utl",
        "gpu_mem",
        "h_to_d",
        "d_to_h",
    )

    SUMMARY: dict[str, Callable[[np.ndarray | list[float]], np.float32]] = {  # type: ignore[type-arg]
        "cpu_utl": np.mean,
        "cpu_mem": np.max,
        "gpu_utl": np.mean,
        "gpu_mem": np.max,
        "h_to_d": np.mean,
        "d_to_h": np.mean,
    }

    def __init__(self, *, device: Optional[torch.device] = None, dindex: Optional[int] = None, interval_s: float = 0.1) -> None:
        if bool(device is None) == bool(dindex is None):
            raise ValueError("Exactly one of `device` or `dindex` must be specified.")
        self.device = device
        self.dindex = dindex
        self.interval = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._samples: dict[str, list[float]] = {k: [] for k in self.COLLECT}
        self._cpu_only = device.type == "cpu" if device is not None else False

    def start(self) -> Monitor:
        if not self._cpu_only:
            pynvml.nvmlInit()
            if self.dindex is not None:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.dindex)
            elif self.device is not None:
                self.handle = self._get_handle_from_device(self.device)
            else:
                raise RuntimeError("Internal error: neither device nor dindex is set.")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Monitor:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            if not self._cpu_only:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        return self

    def clear(self) -> Monitor:
        self._samples = {k: [] for k in self.COLLECT}
        return self

    def get_report(self, clear: bool = False) -> dict[str, float]:
        """
        Return a summary statistic for each collected metric.
        """
        d = {k: float(self.SUMMARY[k](v)) if len(v) > 0 else float("nan") for k, v in self._samples.items()}
        if clear:
            self.clear()
        return d

    def _run(self) -> None:
        while self._running:
            d = self._collect()
            for k, v in d.items():
                self._samples[k].append(v)
            time.sleep(self.interval)

    def _collect(self) -> dict[str, float]:
        cpu_utl = psutil.cpu_percent(interval=0.0) / 100.0
        cpu_mem = psutil.virtual_memory().used
        if self._cpu_only:
            return {
                "cpu_utl": cpu_utl,
                "cpu_mem": cpu_mem,
                "gpu_utl": float("nan"),
                "gpu_mem": float("nan"),
                "h_to_d": float("nan"),
                "d_to_h": float("nan"),
            }
        gpu_utl = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu / 100.0
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        h_to_d  = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) * 1024.0
        d_to_h  = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) * 1024.0
        return {
            "cpu_utl": cpu_utl,
            "cpu_mem": cpu_mem,
            "gpu_utl": gpu_utl,
            "gpu_mem": gpu_mem,
            "h_to_d": h_to_d,
            "d_to_h": d_to_h,
        }

    @staticmethod
    def _get_handle_from_device(device: torch.device) -> Any:
        # NOTE: pynvml must be initialized before calling this method with pynvml.nvmlInit()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got {device!r}")

        # Resolve CUDA runtime index (visible index)
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()

        # Query CUDA device properties
        props = torch.cuda.get_device_properties(cuda_idx)

        # Prefer UUID mapping (robust across reordering)
        dev_uuid = getattr(props, "uuid", None)
        if dev_uuid is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(str(dev_uuid))
            except pynvml.NVMLError:
                pass

        # Fallback: PCI bus ID mapping (also robust)
        pci_bus_id = getattr(props, "pci_bus_id", None)
        if pci_bus_id is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByPciBusId(str(pci_bus_id))
            except pynvml.NVMLError:
                pass

        # Last resort: map via CUDA_VISIBLE_DEVICES if present
        cvd = os.getenv("CUDA_VISIBLE_DEVICES")
        if cvd:
            physical_idx = int(cvd.split(",")[cuda_idx].strip())
            try:
                return pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            except pynvml.NVMLError:
                pass

        # Absolute last fallback: assume CUDA and NVML indices align
        try:
            return pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)
        except pynvml.NVMLError:
            pass

        raise RuntimeError(f"Failed to find NVML handle for device {device}.")
