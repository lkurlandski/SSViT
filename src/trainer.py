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
from functools import partial
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
from typing import Literal
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

from src.utils import Timer


ALLOW_PARAM_GRAD_NONE = os.environ.get("ALLOW_PARAM_GRAD_NONE", "0") == "1"
if ALLOW_PARAM_GRAD_NONE:
    warnings.warn("Allowing parameters with no gradients; some weights may not be updated.")


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

    The proper pattern for effectively using this batch interface is as follows:
        - Initialize low-precision tensors on the CPU
        - Transfer them over the the GPU with Batch.to(device, non_blocking=True)
        - Cast them to the desired dtypes with Batch.finalize(ftype, itype, ltype)
        - Access the tensors with Batch.get_label(), Batch.get_inputs(), and Batch.get_guides()
           This stage should NOT involve any copies or computations (to keep profiling sane).
    """

    def __len__(self) -> int:
        """
        Return the number of samples in the batch.
        """
        ...

    def clone(self) -> Self:
        """
        Return a deep copy of the batch.
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

    def get_label(self) -> Tensor:
        """
        Access the labels tensor, which should have been finalized in `finalize` (no copy/compute).
        """
        ...

    def get_inputs(self) -> Tensor | Sequence[Tensor]:
        """
        Access the inputs tensor(s), which should have been finalized in `finalize` (no copy/compute).
        """
        ...

    def get_guides(self) -> Optional[Tensor] | Sequence[Optional[Tensor]]:
        """
        Access the guides tensor(s), which should have been finalized in `finalize` (no copy/compute).
        """
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
    stopper_metric: str = "vl_loss"
    stopper_mode: str = "min"
    stopper_threshold: float = 0.0
    stopper_patience: float = float("inf")
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mp16: bool = False
    max_epochs: Optional[float] = None
    max_steps: Optional[int] = None
    eval_epochs: Optional[float] = None
    eval_steps: Optional[int] = None
    chpt_epochs: Optional[float] = None
    chpt_steps: Optional[int] = None
    logg_epochs: Optional[float] = None
    logg_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.max_epochs is not None) and (self.max_steps is not None):
            raise ValueError("At most one of `max_epochs` or `max_steps` must be specified.")
        if self.max_epochs is None and self.max_steps is None:
            self.max_epochs = 1.0
        if (self.eval_epochs is not None) and (self.eval_steps is not None):
            raise ValueError("At most one of `eval_epochs` or `eval_steps` may be specified.")
        if self.eval_epochs is None and self.eval_steps is None:
            self.eval_epochs = 1.0
        if (self.chpt_epochs is not None) and (self.chpt_steps is not None):
            raise ValueError("At most one of `chpt_epochs` or `chpt_steps` may be specified.")
        if self.chpt_epochs is None and self.chpt_steps is None:
            self.chpt_epochs = 1.0
        if (self.logg_epochs is not None) and (self.logg_steps is not None):
            raise ValueError("At most one of `logg_epochs` or `logg_steps` may be specified.")
        if self.logg_epochs is None and self.logg_steps is None:
            self.logg_epochs = 1.0
        if self.stopper_mode not in ("min", "max"):
            raise ValueError("`stopper_mode` must be 'min' or 'max'.")
        if self.stopper_patience < 0:
            self.stopper_patience = float("inf")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls.from_dict(vars(namespace))


class EarlyStopper:
    """
    Stateful class to support early stopping conditions.

    Args:
        patience: Number of steps with no improvement after which stopping will be triggered.
            If `patience` is 0, the stopper will signal to stop if `step()` is called without improvement.
            If patience is N, the stopper won't signal to stop until `step()` has been called N times
            without improvement; if the N+1-th call to `step()` also shows no improvement, then the stopper
            will signal to stop. If patience is `float('inf')`, the stopper will never signal to stop.
        threshold: Minimum change in the monitored metric to qualify as an improvement.
            If `threshold` is 0, any improvement in the monitored metric will qualify.
            If `threshold` is positive, the monitored metric must improve by at least `threshold`.
            If `threshold` is negative, the monitored metric must "improve" by at least `-threshold`,
            i.e., it may actually worsen by up to `-threshold` and still count as an improvement.
        mode: Determines whether an increase or decrease is considered an improvement.
            If `mode` is "max", an increase in the monitored metric is considered an improvement.
            If `mode` is "min", a decrease in the monitored metric is considered an improvement.

    Usage:
        >>> stopper = EarlyStopper(patience=2, threshold=0.001, mode="max")
        >>> stopper = stopper.step(0.5)
        >>> stopper = stopper.step(0.6)
        >>> stopper = stopper.step(0.6005)  # No improvement; count := 1  <=  patience
        >>> stopper = stopper.step(0.6010)  # No improvement; count := 2  <=  patience
        >>> stopper = stopper.step(0.6011)  # Improvement;    count := 0  <=  patience
        >>> stopper = stopper.step(0.6)     # No improvement; count := 1  <=  patience
        >>> stopper = stopper.step(0.6)     # No improvement; count := 2  <=  patience
        >>> stopper = stopper.step(0.6)     # No improvement; count := 3  >   patience
        >>> print(stopper.stop)             # True
        >>> stopper = stopper.step(0.7)     # Raises RuntimeError
    """

    def __init__(self, patience: int | float = float("inf"), threshold: float = 0.0, mode: Literal["min", "max"] = "min") -> None:
        if patience < 0:
            raise ValueError("`patience` must be >= 0")
        if mode not in ("min", "max"):
            raise ValueError("`mode` must be 'min' or 'max'")

        self.patience = patience
        self.threshold = threshold
        self.mode = mode

        self.best: Optional[float] = None
        self.num_bad_steps = 0
        self._stop = False

    def step(self, value: float) -> Self:
        if self._stop:
            raise RuntimeError("Early stopping already triggered.")

        # Set the baseline the first time step() is called.
        if self.best is None:
            self.best = value
            self.num_bad_steps = 0
            return self

        improved: Optional[bool] = None
        if self.mode == "min":  # Improvement means "sufficiently smaller".
            improved = value < self.best - self.threshold
        if self.mode == "max":  # Improvement means "sufficiently larger".
            improved = value > self.best + self.threshold
        if improved is None:
            raise RuntimeError("This should never happen.")

        if improved:
            self.best = value
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1
            if self.num_bad_steps > self.patience:
                self._stop = True

        return self

    @property
    def stop(self) -> bool:
        return self._stop


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


# Source: torcheval.metrics.functional.classification.precision_recall_curve
@torch.jit.script
def _compute_for_each_class(
    input: torch.Tensor,
    target: torch.Tensor,
    pos_label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=0) != 0, [0, 1], value=1.0)
    num_tp = (target[indices] == pos_label).cumsum(0)[mask]
    num_fp = (1 - (target[indices] == pos_label).long()).cumsum(0)[mask]
    precision = (num_tp / (num_tp + num_fp)).flip(0)
    recall = (num_tp / num_tp[-1]).flip(0)
    threshold = threshold[mask].flip(0)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)])
    recall = torch.cat([recall, recall.new_zeros(1)])

    # If recalls are NaNs, set NaNs to 1.0s.
    if torch.isnan(recall[0]):
        recall = torch.nan_to_num(recall, 1.0)

    return precision, recall, threshold


# Source: torcheval.metrics.functional.tensor_utils
def _riemann_integral(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -torch.sum((x[1:] - x[:-1]) * y[:-1])


# Source: torcheval.metrics.functional.classification.auprc
@torch.jit.script
def _binary_auprc_compute_jit(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:

    p, r, t = _compute_for_each_class(input, target, 1)
    return _riemann_integral(r, p)


def flush() -> None:
    print("", flush=True, end="")


def print_parameter_summary(model: nn.Module, spaces: int = 0) -> None:
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            print(f"{' ' * spaces}param {name} grad: None")
        else:
            print(f"{' ' * spaces}param {name} grad: {grad.dtype} {tuple(grad.shape)} {grad.device} {grad.sum().item():.4f}")


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
        self.scaler = GradScaler(self.device.type, enabled=self.mp_dtype == torch.float16)
        self.log: list[Mapping[str, int | float]] = []
        self.glbl_step = 0
        self._next_eval_step: Optional[int] = None
        self._next_chpt_step: Optional[int] = None
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
        if self.logg_steps is not None:
            self._next_logg_step = self.glbl_step + self.logg_steps

        pbar = tqdm(total=self.max_epochs, disable=self.args.disable_tqdm, leave=False, unit="step")
        pbar.set_description(f"Epoch {self.glbl_step // self.steps_per_epoch} of {self.max_steps / self.steps_per_epoch}")

        # Conduct an initial validation and checkpointing on the naked model.
        if self.glbl_step == 0:
            self.run_due_hooks(None, do_eval=True, do_chpt=True, do_logg=True)

        # Continuously train the model on the training set until finished.
        while not self._done:
            self.train()
            pbar.update(1)
            pbar.set_description(f"Epoch {self.glbl_step // self.steps_per_epoch} of {self.max_steps / self.steps_per_epoch}")

        self.monitor.stop()

        return self

    def train(self) -> None:
        """
        Train the model on the training set.
        """
        barrier("Trainer::train:before", self.device)
        timer = Timer()

        self.optimizer.zero_grad()
        self.model.train()

        # Get a wrapped dataloader with padding to the longest rank.
        dataloader = self.tr_loader
        iterable = self._wrap_and_pad_loader(dataloader, "Training...", leave=False)

        def get_report() -> dict[str, float]:
            """Assemble a training report (excluding GPU statistics)."""
            # Aggregate results across workers and move to CPU.
            allresults = self.reduce_results(results) if is_dist() else results
            allresults = {k: allresults[k].detach().to("cpu") for k in allresults}
            num_samples = int(allresults.pop("num_samples").item())
            grad_steps = math.ceil(int(allresults.pop("grad_steps").item()) / world_size())
            # Average statistics over some period for logging.
            report = {}
            for k in allresults:
                # Average these over number of samples.
                if k in ("tr_loss",):
                    report[k] = allresults[k].item() / num_samples
                # Average these over number of steps.
                elif k in ("grad_norm", "param_norm", "param_delta",):
                    report[k] = allresults[k].item() / grad_steps
                # Do not average these.
                elif k in ("num_samples",):
                    report[k] = allresults[k].item()
                # Otherwise raise an error.
                else:
                    raise KeyError(f"Unknown training metric '{k}' encountered.")
            report["tr_time"] = timer.get_elapsed()
            report["tr_samples"] = num_samples
            report["tr_throughput"] = num_samples / report["tr_time"]
            return report

        # Parmameter delta.
        flat_params_bef: Tensor
        flat_params_aft: Tensor
        with torch.no_grad():
            flat_params_bef = torch.cat([p.view(-1) for p in self.model.parameters()])

        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
        timer.start()
        mini_step = -1  # mini-step within this epoch
        for mini_step, batch in iterable:
            batch = batch.to(self.device, non_blocking=True)
            batch = batch.finalize(self.mp_dtype, torch.int32, torch.int64)

            # Determine if this is a real or padded batch of data.
            real = mini_step < len(dataloader)
            results["num_samples"] += len(batch) * int(real)

            # Sync on gradient accumulation boundary or final real mini-step
            step_in_accum = (mini_step + 1) % self.args.gradient_accumulation_steps
            is_accum_boundary = (step_in_accum == 0)
            is_last_step = (mini_step + 1 == len(iterable))
            sync_gradients = is_accum_boundary or is_last_step

            # Compute normalized loss
            with maybe_no_sync(self.model, sync_gradients):
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                loss = loss * int(real) / self.args.gradient_accumulation_steps
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                results["tr_loss"] += loss.detach() * len(batch) * self.args.gradient_accumulation_steps

            # Check for parameters with no gradients
            if not ALLOW_PARAM_GRAD_NONE and any(param.grad is None for param in self.model.parameters()):
                flush()
                print(f"{'-' * 20} Parameter Summary After Step {self.glbl_step:09} {'-' * 20}")
                print_parameter_summary(self.model, spaces=2)
                print(f"{'-' * 80}")
                raise RuntimeError("Some of the parameters have no gradients.")

            # Update model weights and possibly run hooks (validation, checkpointing, etc.)
            if sync_gradients:
                results["grad_steps"] += 1
                # Take an optimization step
                grad_norm = self.step()
                results["grad_norm"] += grad_norm.detach()
                # Compute parameter delta
                with torch.no_grad():
                    flat_params_aft = torch.cat([p.view(-1) for p in self.model.parameters()])
                results["param_norm"] += flat_params_aft.norm().detach()
                results["param_delta"] += (flat_params_aft - flat_params_bef).norm().detach()
                flat_params_bef = flat_params_aft
                # Determine what hooks are to be executed.
                do_logg = any(self._due_hooks())
                do_eval = self._due_hooks()[0]
                # Free up memory before validation to keep GPU memory usage lower
                if do_eval:
                    del batch, outputs, loss
                    gc.collect()
                # Close the progress bar if evaluating on the last step
                if do_eval and is_last_step and isinstance(iterable, tqdm):
                    iterable.close()
                # Prepare a training report and reset the tracking objects
                if do_logg:
                    report = get_report()
                    timer.start()
                    results = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
                else:
                    report = None
                # Run the due hooks, but pause the timer during their execution
                timer.pause()
                self.run_due_hooks(report)
                self.model.train()
                timer.resume()
                # Clear the monitor if logging was performed
                if do_logg:
                    self.monitor.clear()

            # Stop if we've reached the maximum number of global steps
            if self.glbl_step >= self.max_steps:
                self._done = True
                break

        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")

        timer.stop()
        barrier("Trainer::train:after", self.device)

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation set.
        """
        barrier("Trainer::evaluate:before", self.device)
        t_0 = time.time()

        was_training = self.model.training
        self.model.eval()

        # Get a wrapped dataloader with padding to the longest rank.
        dataloader = self.vl_loader
        iterable = self._wrap_and_pad_loader(dataloader, "Validating...", leave=False)

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
                    alllabels.append(batch.get_label())
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

    def _wrap_and_pad_loader(self, dataloader: Collection[Batch] | DataLoader[Batch], desc: str = "", leave: bool = True) -> tqdm[tuple[int, Batch]]:
        if is_dist() and self.padbatch is None:
            raise RuntimeError("padbatch must be specified for distributed training.")

        # Get the length of this rank's dataloader and the longest dataloader across all ranks.
        length = torch.tensor(len(dataloader), device=self.device)
        longest = length.clone()
        if is_dist():
            dist.all_reduce(longest, op=dist.ReduceOp.MAX, group=None)
        length = int(length.to(torch.int64).item())
        longest = int(longest.to(torch.int64).item())

        # Get a (possibly padded) iterable over the samples in the loader.
        num_padding = longest - length
        padding = itertools.repeat(self.padbatch, num_padding)
        iterable = itertools.chain(dataloader, padding)

        # Wrap the iterable in enumerate and a tqdm progress bar.
        iterable = enumerate(iterable)
        iterable = tqdm(iterable, desc, longest, leave, disable=self.args.disable_tqdm, ascii=True)

        return iterable

    def _due_hooks(self) -> tuple[bool, bool, bool]:
        do_eval = self._next_eval_step is not None and self.glbl_step >= self._next_eval_step
        do_chpt = self._next_chpt_step is not None and self.glbl_step >= self._next_chpt_step
        do_logg = self._next_logg_step is not None and self.glbl_step >= self._next_logg_step
        return do_eval, do_chpt, do_logg

    def run_due_hooks(
        self,
        tr_report: Optional[dict[str, float]],
        *,
        do_eval: Optional[bool] = None,
        do_chpt: Optional[bool] = None,
        do_logg: Optional[bool] = None,
    ) -> None:
        """
        Conduct periodic activities, such as running validation and checkpointing.

        Args:
            tr_report: Optional dictionary of training statistics to include in the report.
            do_eval: If True, run validation even if not scheduled.
            do_chpt: If True, save a checkpoint even if not scheduled.
            do_logg: If True, update the console and log files even if not scheduled.

        NOTE: the schedules are only advanced if fired organically, not via the `do_*` arguments.
        NOTE: logging will occur if any hook fires, but will not advance the logging schedule.
        """
        due_eval, due_chpt, due_logg = self._due_hooks()

        do_eval = do_eval if do_eval is not None else due_eval
        do_chpt = do_chpt if do_chpt is not None else due_chpt
        do_logg = do_logg if do_logg is not None else due_logg

        # If no hooks are scheduled, exit prematurely.
        if not (do_eval or do_chpt or do_logg):
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

        # Always update the log and console.
        self._update_cons(report)
        self._update_logs(report)

        # Advance schedules that fired.
        if due_eval:
            self._next_eval_step += self.eval_steps  # type: ignore[operator]
        if due_chpt:
            self._next_chpt_step += self.chpt_steps  # type: ignore[operator]
        if due_logg:
            self._next_logg_step += self.logg_steps  # type: ignore[operator]

        # Handle early stopping (only if validation is orgnanically scheduled).
        if due_eval:
            self.stopper.step(report[f"{self.args.stopper_metric}"])
        if self.stopper.stop:
            self.print("Early stopping triggered.")
            self._done = True

    def forward(self, batch: Batch) -> Tensor:
        """
        Send a batch of inputs forward through the model.
        """
        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            return self.model(batch.get_inputs(), batch.get_guides())  # type: ignore[no-any-return]

    def compute_loss(self, batch: Batch, outputs: Tensor) -> Tensor:
        """
        Compute the loss over a batch of examples.
        """
        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            return self.loss_fn(outputs, batch.get_label())  # type: ignore[no-any-return]

    def step(self) -> Tensor:
        """
        Take an optimization step and return the norm of gradients.
        """
        self.scaler.unscale_(self.optimizer)
        norm: Tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        warnings.filterwarnings("ignore")
        self.scheduler.step()
        warnings.resetwarnings()
        self.optimizer.zero_grad()
        self.glbl_step += 1
        return norm

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
        prc = _binary_auprc_compute_jit(probs, labels)

        return {
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f-1": f_1,
            "roc": roc,
            "prc": prc,
        }

    def print(self, *args, sep: str = " ", end: str = "\n", file: str | None = None) -> None:
        if self.args.disable_tqdm:
            print(*args, sep=sep, end=end, file=file)
        else:
            tqdm.write(sep.join(map(str, args)), file=file, end=end)

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
    def max_epochs(self) -> float:
        if self.args.max_epochs is not None:
            return self.args.max_epochs
        return self.args.max_steps / self.steps_per_epoch

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
            d["grad_norm"] = round(results["grad_norm"], 3)
            d["param_norm"] = round(results["param_norm"], 3)
            d["param_delta"] = round(results["param_delta"], 3)
            d["tr_loss"] = round(results["tr_loss"], 3)
            d["tr_gpu_utl"] = round(results["tr_gpu_utl"], 2)
            d["tr_gpu_mem"] = round(results["tr_gpu_mem"] / (1024 ** 3), 2)
            d["tr_time"] = round(results["tr_time"], 0)
            d["tr_samples"] = int(results["tr_samples"])
            d["tr_throughput"] = round(results["tr_throughput"], 2)
        if any(k.startswith("vl_") for k in results):
            d["vl_loss"] = round(results["vl_loss"], 3)
            d["vl_roc"] = round(results["vl_roc"], 3)
            d["vl_prc"] = round(results["vl_prc"], 3)
            d["vl_gpu_utl"] = round(results["vl_gpu_utl"], 2)
            d["vl_gpu_mem"] = round(results["vl_gpu_mem"] / (1024 ** 3), 2)
            d["vl_time"] = round(results["vl_time"], 0)
            d["vl_samples"] = int(results["vl_samples"])
            d["vl_throughput"] = round(results["vl_throughput"], 2)
        self.print(d)

    def to_checkpoint(self, path: str | Path) -> None:
        """
        Save a checkpoint of the trainer state to the specified path.

        Structure:
            path
            ├── meta.json
            ├── args.pickle
            ├── log.pickle
            ├── log.jsonl
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
                "_next_logg_step": self._next_logg_step,
            }
            Path(path / "meta.json").write_text(json.dumps(meta, indent=2))
            # Pickled objects
            Path(path / "args.pickle").write_bytes(pickle.dumps(self.args))
            Path(path / "log.pickle").write_bytes(pickle.dumps(self.log))
            with open(path / "log.jsonl", "w") as fp:
                for entry in self.log:
                    try:
                        line = json.dumps(entry)
                    except Exception as err:
                        line = json.dumps({"error": f"Could not serialize log entry: {err}"})
                    fp.write(line + "\n")
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
            optimizer_init = AdamW
        if optimizer is None and not is_dcp:
            optimizer = optimizer_init(model.parameters())
        if scheduler_init is None:
            scheduler_init = partial(LambdaLR, lr_lambda=lambda _: 1.0)
        if scheduler is None and not is_dcp:
            assert optimizer is not None
            scheduler = scheduler_init(optimizer)

        # Load the simple objects first that don't require complex state dict handling.
        meta = json.loads((path / "meta.json").read_text())
        glbl_step = meta["glbl_step"]
        _next_eval_step = meta["_next_eval_step"]
        _next_chpt_step = meta["_next_chpt_step"]
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
