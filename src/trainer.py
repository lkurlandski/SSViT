"""
Train and validation loops.
"""

from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Generator
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
import sys
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
from torch.optim.optimizer import ParamsT
from torch.profiler import record_function
from torch.profiler import profile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src.utils import Timer
from src.utils import check_tensor


CHECK_PARAM_GRAD_NONE = os.environ.get("CHECK_PARAM_GRAD_NONE", "1") == "1"
if not CHECK_PARAM_GRAD_NONE:
    warnings.warn("Parameters will not be checked for having gradients.")

ALLOW_PARAM_GRAD_NONE = not CHECK_PARAM_GRAD_NONE or os.environ.get("ALLOW_PARAM_GRAD_NONE", "0") == "1"
if ALLOW_PARAM_GRAD_NONE:
    warnings.warn("Parameters are allowed to have no gradients.")

TRAINER_SAVE_PREDICTIONS = os.environ.get("TRAINER_SAVE_PREDICTIONS", "0") == "1"
if TRAINER_SAVE_PREDICTIONS:
    warnings.warn("Trainer will save predictions during evaluation.")


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

    def get_names(self) -> list[str]:
        """
        Return a list of fixed-width names, e.g., hexadecimal hashes, of the samples in the batch.
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

    def get_otherkwds(self) -> dict[str, Any]:
        """
        Access any other keyword arguments needed for the model, which should have been finalized in `finalize` (no copy/compute).
        """
        ...


def largest_possible_dataloader_length(loader: DataLoader[Any]) -> int:
    # Validate the type and properties of the dataloader's dataset.
    if not isinstance(loader.dataset, (Dataset, IterableDataset)):
        raise TypeError("`loader.dataset` must be a `Dataset` or `IterableDataset`.")
    if not hasattr(loader.dataset, "__len__"):
        raise ValueError("`loader.dataset` has no `__len__` method.")
    if len(loader.dataset) is None:
        raise ValueError("`len(loader.dataset)` returned None.")
    # The length for underlying map-style datasets is exact.
    if isinstance(loader.dataset, Dataset) and not isinstance(loader.dataset, IterableDataset):
        return len(loader)
    # The length for underlying iterable-style datasets is an estimate.
    length = len(loader.dataset)  # type: ignore[arg-type]
    length = length / (loader.batch_size if loader.batch_size is not None else 1)
    length = math.floor(length) if loader.drop_last else math.ceil(length)
    length = max(0, int(length) + loader.num_workers - 1)
    return length


def round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


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


def _check_logits_and_labels(logits: Tensor, labels: Tensor, num_classes: Optional[int] = None) -> None:
    # Check the shape and dtype of logits and labels.
    check_tensor(logits, (None, None), torch.float64)
    check_tensor(labels, (None,), torch.int64)
    # Infer num_classes if not provided.
    num_classes = logits.shape[1] if num_classes is None else num_classes
    check_tensor(logits, (None, num_classes), torch.float64)
    # Check consistency between logits and labels.
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Logits and labels have different number of samples: {logits.shape[0]} vs {labels.shape[0]}.")
    # Check that labels are in bounds.
    unq: list[int] = torch.unique(labels).tolist()
    if any((u < 0 or u >= num_classes) for u in unq):
        raise ValueError(f"Labels contain out-of-bounds values: {unq} not in [0, {num_classes - 1}].")


def check_logits_and_labels(logits: Tensor, labels: Tensor, num_classes: Optional[int] = None) -> None:
    """
    Check the logits and labels for shape, dtype, and consistency prior to metric computation.
    """
    try:
        _check_logits_and_labels(logits, labels, num_classes)
    except Exception:
        torch.save(logits, f"tmp/debug_logits_rank-{rank()}.pt")
        torch.save(labels, f"tmp/debug_labels_rank-{rank()}.pt")
        print(f"[rank {rank()}] saved logits to tmp/debug_logits_rank-{rank()}.pt for debugging.")
        print(f"[rank {rank()}] saved labels to tmp/debug_labels_rank-{rank()}.pt for debugging.")
        print(f"[rank {rank()}] labels {tuple(labels.shape)} {labels.dtype} {labels.device} {labels.isnan().any().item()} {labels.min().item()} {labels.max().item()}")
        print(f"[rank {rank()}] logits {tuple(logits.shape)} {logits.dtype} {logits.device} {logits.isnan().any().item()} {logits.min().item()} {logits.max().item()}")
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
    assert_auxillary_loss: bool = False
    auxillary_loss_weight: float = 0.0

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


def get_last_aux_loss(model: nn.Module) -> Optional[Tensor]:
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    last_aux_loss = getattr(model, "last_aux_loss", None)
    if last_aux_loss is None:
        return None
    if not isinstance(last_aux_loss, Tensor):
        raise TypeError("Model's `last_aux_loss` attribute must be a Tensor.")
    if last_aux_loss.ndim > 0:
        warnings.warn("Model's `last_aux_loss` attribute has more than 0 dimensions; summing to scalar.")
    return last_aux_loss.sum()


class Trainer:
    """
    Trainer class for training models with PyTorch.
    """

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_loader: DataLoader[Batch],
        vl_loader: DataLoader[Batch],
        padbatch: Batch,
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.args = args
        self.model = model
        self.tr_loader = tr_loader
        self.vl_loader = vl_loader
        self.padbatch = padbatch
        self.loss_fn = loss_fn
        self.optimizer = optimizer if optimizer is not None else AdamW(model.parameters())
        self.scheduler = scheduler if scheduler is not None else LambdaLR(self.optimizer, lambda _: 1.0)
        self.stopper = stopper if stopper is not None else EarlyStopper(patience=float("inf"))
        self.device = device if device is not None else next(self.model.parameters()).device
        self.monitor = Monitor(device=self.device)
        self.scaler = GradScaler(self.device.type, enabled=self.mp_dtype == torch.float16)
        self.log: list[Mapping[str, int | float]] = []
        self.glbl_step = 0
        self.epoch_idx = 0
        self._next_eval_step: Optional[int] = None
        self._next_chpt_step: Optional[int] = None
        self._next_logg_step: Optional[int] = None
        self._done: bool = False
        self._vl_dataloader_lengths = self.get_dataloader_lengths(self.vl_loader)
        self._tr_dataloader_lengths = self.get_dataloader_lengths(self.tr_loader)

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
            f"  glbl_step={self.glbl_step},\n"
            f"  epoch_idx={self.epoch_idx},\n"
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

        pbar = tqdm(total=self.max_epochs - self.epoch_idx, disable=self.args.disable_tqdm, leave=False, unit="step")
        pbar.set_description(f"Epoch {self.epoch_idx} of {self.max_steps / self.steps_per_epoch}")

        # Conduct an initial validation and checkpointing on the naked model.
        if self.glbl_step == 0:
            self.run_due_hooks(None, do_eval=True, do_chpt=True, do_logg=True)

        # Determine the first mini-batch to train upon.
        start_mini_step = 0
        if self.glbl_step > 0:
            steps_into_epoch = self.glbl_step - (self.epoch_idx * self.steps_per_epoch)
            mini_steps_into_epoch = steps_into_epoch * self.args.gradient_accumulation_steps
            start_mini_step = 1 + mini_steps_into_epoch
            self.print(f"[INFO] [rank {rank()}] [Trainer::train] {steps_into_epoch=} {mini_steps_into_epoch=} {start_mini_step=}")

        # Continuously train the model on the training set until finished.
        while not self._done:
            self.train(start_mini_step=start_mini_step)
            start_mini_step = 0
            self.epoch_idx += 1
            pbar.update(1)
            pbar.set_description(f"Epoch {self.epoch_idx} of {self.max_steps / self.steps_per_epoch}")
            if self.epoch_idx >= int(os.environ.get("TRAINER_EARLY_TERMINATE", f"{sys.maxsize}")):
                self.print("Early termination triggered.")
                break

        self.monitor.stop()

        return self

    def train(self, prof: Optional[profile] = None, start_mini_step: int = 0) -> None:
        """
        Train the model on the training set.

        Args:
            prof: If provided, will profile using this profiler.
            start_mini_step: If provided, will skip mini_batches until this index.
        """
        barrier("Trainer::train:before", self.device)
        timer = Timer()

        self.optimizer.zero_grad()
        self.model.train()

        # Get a wrapped dataloader with padding to the longest rank and gradient accumulation.
        iterable = self.get_iterable(self.tr_loader, self.tr_dataloader_length, "Training...", leave=False)
        # self.print(f"[INFO] [rank {rank()}] [Trainer::train] {len(self.tr_loader)=} {largest_possible_dataloader_length(self.tr_loader)=} {self.tr_dataloader_length=} {len(iterable)=}")

        def get_report() -> dict[str, float]:
            """Assemble a training report (excluding GPU statistics)."""
            # Aggregate results across workers and move to CPU.
            allresults = self.reduce_results(results) if is_dist() else results
            allresults = {k: allresults[k].detach().to("cpu") for k in allresults}
            # If zero real samples were seen across all ranks then `num_samples` will be zero and `grad_steps` will be missing.
            num_samples = int(allresults.pop("num_samples").item())
            grad_steps = math.ceil(int(allresults.pop("grad_steps", torch.tensor(0)).item()) / world_size())
            # Average statistics over some period for logging. If real data was seen, we average the
            # losses over number of samples and the norms over the number of gradient steps.
            # If no real data was seen, don't average anything because `num_samples` and `grad_steps`
            # will both be 0, so we'll get a ZeroDivisionError. We still want the key in the report, though.
            # Note that we must consider whether or not real data was seen since the last time we called
            # `get_report`, not just in this cycle, i.e., we cannot use `anyone_had_real_window` here.
            report = {}
            for k in allresults:
                report[k] = allresults[k].item()
                if k in ("tr_loss", "aux_loss", "clf_loss"):
                    if num_samples > 0:
                        report[k] /= num_samples
                elif k in ("grad_norm", "param_norm", "param_delta",):
                    if num_samples > 0:
                        report[k] /= grad_steps
                elif k not in ("num_samples",):
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

        # NOTE: the dynamic nature of this dictionary is making it challenging to account for all the required keys.
        # TODO: We may want to refactor this at some point with a list of static, required keys.
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))

        had_real_window = False  # whether any real data was seen in window of mini-steps
        mini_step = -1           # mini-step within this epoch
        real: bool               # whether the current mini-batch has real data
        batch: Batch             # current mini-batch

        iterator = iter(iterable)
        timer.start()
        while True:
            with record_function("stage::prepare"):
                try:
                    mini_step, real, batch = next(iterator)
                except StopIteration:
                    break
            if mini_step < start_mini_step:
                continue
            with record_function("stage::transfer"):
                batch = batch.to(self.device, non_blocking=True)
            with record_function("stage::finalize"):
                batch = batch.finalize(self.mp_dtype, torch.int32, torch.int64)

            # Determine if this is a real or padded batch of data.
            had_real_window |= bool(real)
            results["num_samples"] += len(batch) * int(real)

            # Sync on gradient accumulation boundary.
            step_in_accum = (mini_step + 1) % self.args.gradient_accumulation_steps
            sync_gradients = (step_in_accum == 0)

            # Compute normalized loss
            with maybe_no_sync(self.model, sync_gradients):
                with record_function("stage::forward"):
                    outputs = self.forward(batch)
                loss, losses = self.compute_loss(batch, outputs)
                loss = loss * int(real) / self.args.gradient_accumulation_steps
                with record_function("stage::backward"):
                    self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                results["tr_loss"] += loss.detach() * len(batch) * self.args.gradient_accumulation_steps
                for lossname in losses:
                    results[lossname] += losses[lossname].detach() * len(batch) * int(real)

            # Check for parameters with no gradients
            # Skip for fake batches, as they take anomalous paths for MoE models and naturally have missing gradients
            if real and CHECK_PARAM_GRAD_NONE and any(param.grad is None for param in self.model.parameters()):
                flush()
                print(f"{'-' * 20} Parameter Summary After Step {self.glbl_step:09} {'-' * 20}")
                print_parameter_summary(self.model, spaces=2)
                print(f"{'-' * 80}")
                if not ALLOW_PARAM_GRAD_NONE:
                    raise RuntimeError("Some of the parameters have no gradients.")

            # Update model weights and possibly run hooks (validation, checkpointing, etc.)
            if sync_gradients:
                # Determine how many ranks had a real window of data
                had = torch.tensor(int(had_real_window), device=self.device, dtype=torch.int32)
                if is_dist():
                    dist.all_reduce(had, op=dist.ReduceOp.SUM)
                real_ranks = int(had.item())
                anyone_had_real_window = (real_ranks > 0)
                grad_scale = (world_size() / real_ranks) if (is_dist() and anyone_had_real_window) else 1.0
                # If anyone had real data, take an optimization step and log stats
                if anyone_had_real_window:
                    results["grad_steps"] += 1
                    # Take an optimization step
                    with record_function("stage::step"):
                        grad_norm = self.step(grad_scale=grad_scale)
                    results["grad_norm"] += grad_norm.detach()
                    # Compute parameter delta
                    with torch.no_grad():
                        flat_params_aft = torch.cat([p.view(-1) for p in self.model.parameters()])
                    results["param_norm"] += flat_params_aft.norm().detach()
                    results["param_delta"] += (flat_params_aft - flat_params_bef).norm().detach()
                    flat_params_bef = flat_params_aft
                else:
                    # Do this dumb shit to ensure we don't get a KeyError in other places.
                    results["grad_steps"] += 0
                    results["grad_norm"] += 0.0
                    results["param_norm"] += 0.0
                    results["param_delta"] += 0.0
                # Determine what hooks are to be executed
                do_eval, do_chpt, do_logg = self._due_hooks()
                ad_eval, ad_chpt, ad_logg = None, None, None
                # Since the step-based schedules are likely overestimating the number of steps/epoch,
                # we might want to manually fire them if we're using epoch-based schedules at the end of an epoch.
                if not anyone_had_real_window:
                    if self._should_fire_hook_at_end_of_epoch(self.args.eval_epochs):
                        do_eval, ad_eval = True, True
                    if self._should_fire_hook_at_end_of_epoch(self.args.chpt_epochs):
                        do_chpt, ad_chpt = True, True
                    if self._should_fire_hook_at_end_of_epoch(self.args.logg_epochs):
                        do_logg, ad_logg = True, True
                should_prepare_report = any((do_eval, do_chpt, do_logg))
                # Free up memory before validation to keep GPU memory usage lower
                if do_eval:
                    del batch, outputs, loss
                    gc.collect()
                # Close the progress bar if we're going to stop training right after this
                if do_eval and isinstance(iterable, tqdm) and (mini_step + 1 == len(iterable) or not anyone_had_real_window):
                    iterable.close()
                # Prepare a training report and reset the tracking objects
                if should_prepare_report:
                    report = get_report()
                    timer.start()
                    results = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
                else:
                    report = None
                # Run the due hooks, but pause the timer during their execution
                timer.pause()
                self.run_due_hooks(report, do_eval=do_eval, do_chpt=do_chpt, do_logg=do_logg, ad_eval=ad_eval, ad_chpt=ad_chpt, ad_logg=ad_logg)
                self.model.train()
                timer.resume()
                # Clear the monitor if logging was performed
                if should_prepare_report:
                    self.monitor.clear()
                # Break from the loop if every rank is on padding
                if not anyone_had_real_window:
                    self.optimizer.zero_grad(set_to_none=True)
                    break
                had_real_window = False

            if prof is not None:
                if not isinstance(prof, profile):
                    raise TypeError("prof must be a torch.profiler.profile instance.")
                prof.step()  # type: ignore[no-untyped-call]

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
        iterable = self.get_iterable(self.vl_loader, self.vl_dataloader_length, "Validating...", leave=False)
        # self.print(f"[INFO] [rank {rank()}] [Trainer::evaluate] {len(self.vl_loader)=} {largest_possible_dataloader_length(self.vl_loader)=} {self.vl_dataloader_length=} {len(iterable)=}")

        mini_step = -1
        totalloss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        totallosses: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))
        alllabels: list[Tensor] = []  # [(B,)]    list of one-dimensional tensors, num-samples
        alllogits: list[Tensor] = []  # [(B, C)]  list of two-dimensional tensors, num-samples x num-classes
        allnames: list[Tensor] = []   # [(B, F)]  list of two-dimensional tensors, num-samples x name-length

        with torch.no_grad():
            for mini_step, real, batch in iterable:
                batch = batch.to(self.device, non_blocking=True).finalize(self.mp_dtype, torch.int32, torch.int64)
                outputs = self.forward(batch)
                loss, losses = self.compute_loss(batch, outputs)
                if real:
                    totalloss += loss.to(torch.float64) * len(batch)
                    for lossname in losses:
                        totallosses[lossname] += losses[lossname].to(torch.float64) * len(batch)
                    alllabels.append(batch.get_label())
                    alllogits.append(outputs)
                    allnames.append(torch.tensor([name.encode("latin1") for name in batch.get_names()], dtype=torch.uint8))

        # self.print(f"[INFO] [rank {rank()}] [Trainer::evaluate] {mini_step=} num_real={len(alllabels)} num_fake={mini_step + 1 - len(alllabels)}")
        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")
        if len(alllabels) == 0 or len(alllogits) == 0:
            raise RuntimeError(f"[rank {rank()}] no real samples were evaluated.")

        totallosses = dict(totallosses)
        labels = torch.cat(alllabels, dim=0).to(device=self.device, dtype=torch.int64)
        logits = torch.cat(alllogits, dim=0).to(device=self.device, dtype=torch.float64)
        names  = torch.cat(allnames, dim=0).to(device=self.device, dtype=torch.uint8)
        check_logits_and_labels(logits, labels)

        if totalloss.grad is not None or totalloss.dtype != torch.float64:
            raise RuntimeError("totalloss should be a non-gradient float64 tensor.")
        for lossname in totallosses:
            if totallosses[lossname].grad is not None or totallosses[lossname].dtype != torch.float64:
                raise RuntimeError(f"totallosses['{lossname}'] should be a non-gradient float64 tensor.")
        if labels.grad is not None or labels.dtype != torch.int64:
            raise RuntimeError("labels should be a non-gradient int64 tensor.")
        if logits.grad is not None or logits.dtype != torch.float64:
            raise RuntimeError("logits should be a non-gradient float64 tensor.")

        if is_dist():
            # Determine the lengths of collection of labels/logits across all ranks.
            length  = torch.tensor([labels.shape[0]], device=self.device, dtype=torch.int64)
            lengths = [torch.zeros_like(length) for _ in range(world_size())]
            dist.all_gather(lengths, length)
            longest = int(torch.max(torch.stack(lengths)).item())
            # Pad labels/logits to common length so gather works.
            if (pad := longest - labels.shape[0]) > 0:
                labels = torch.nn.functional.pad(labels, (0, pad),       value=0)
                logits = torch.nn.functional.pad(logits, (0, 0, 0, pad), value=torch.nan)
                names  = torch.nn.functional.pad(names,  (0, 0, 0, pad), value=0)
            # Gather all labels and logits.
            glabels: Optional[list[Tensor]] = None
            glogits: Optional[list[Tensor]] = None
            gnames: Optional[list[Tensor]]  = None
            if rank() == 0:
                glabels = [torch.empty((longest,),                 device=labels.device, dtype=labels.dtype) for _ in range(world_size())]
                glogits = [torch.empty((longest, logits.shape[1]), device=logits.device, dtype=logits.dtype) for _ in range(world_size())]
                gnames  = [torch.empty((longest, names.shape[1]),  device=names.device,  dtype=names.dtype)  for _ in range(world_size())]
            dist.gather(labels.contiguous(), gather_list=glabels, dst=0)
            dist.gather(logits.contiguous(), gather_list=glogits, dst=0)
            dist.gather(names.contiguous(),  gather_list=gnames,  dst=0)
            # Concatenate and remove padding.
            if rank() == 0:
                assert glabels is not None
                assert glogits is not None
                assert gnames is not None
                labels = torch.cat([glabels[r][: int(lengths[r].item())] for r in range(world_size())], dim=0)
                logits = torch.cat([glogits[r][: int(lengths[r].item())] for r in range(world_size())], dim=0)
                names  = torch.cat([gnames[r][: int(lengths[r].item())]  for r in range(world_size())], dim=0)
                check_logits_and_labels(logits, labels)
            # Sum the loss across all ranks.
            dist.all_reduce(totalloss, op=dist.ReduceOp.SUM)
            for lossname in totallosses:
                dist.all_reduce(totallosses[lossname], op=dist.ReduceOp.SUM)
            # Cleanup some references to prevent bugs.
            del length, lengths, longest, pad, glabels, glogits, gnames

        # Compute the metrics.
        if rank() == 0:
            metrics = self.compute_metrics(labels.to("cpu"), logits.to("cpu"))
            metrics["loss"] = totalloss.to("cpu") / labels.shape[0]
            for lossname in totallosses:
                metrics[lossname] = totallosses[lossname].to("cpu") / labels.shape[0]

        # Prepare a validation report.
        report: Optional[dict[str, float]] = None
        if rank() == 0:
            report = {}
            for k in metrics:
                report[f"vl_{k}"] = metrics[k].item()
            report["vl_time"] = time.time() - t_0
            report["vl_samples"] = labels.shape[0]
            report["vl_throughput"] = labels.shape[0] / report["vl_time"]

        # Broadcast the report to all ranks.
        if is_dist():
            obj = [report]
            dist.broadcast_object_list(obj, src=0)
            report = obj[0]

        if report is None:
            raise RuntimeError("Unreachable.")

        if was_training:
            self.model.train()

        # Save the labels, logits, and names if needed.
        if os.environ.get("TRAINER_SAVE_PREDICTIONS", "1") == "1" and rank() == 0:
            output = self.args.outdir / f"predictions-{self.glbl_step}"
            output.mkdir(parents=True, exist_ok=True)
            torch.save(labels, (output / "labels.pt").as_posix())
            torch.save(logits, (output / "logits.pt").as_posix())
            torch.save(names,  (output / "names.pt").as_posix())

        barrier("Trainer::evaluate:after", self.device)
        return report

    def get_iterable(self, dataloader: DataLoader[Batch], length: int, desc: str = "", leave: bool = True) -> tqdm[tuple[int, bool, Batch]]:
        """
        Get a padded and wrapped stream of batches from a dataloader.

        Args:
            dataloader: The dataloader to wrap and pad.
            length: The length to pad the dataloader to.
            desc: Description for the tqdm progress bar.
            leave: Whether to leave the tqdm progress bar after iteration.

        Returns:
            A tqdm iterable yielding tuples of (mini_step, is_real, batch), where `is_real` is
            True if the batch is from the original dataloader, and False if it is a padded batch.
        """

        def stream() -> Generator[tuple[int, bool, Batch]]:
            # Yield all real batches first.
            i = -1
            for i, batch in enumerate(dataloader):
                yield i, True, batch
            # Then yield padded batches.
            while i < length - 1:
                i += 1
                yield i, False, self.padbatch.clone()
            return

        return tqdm(stream(), desc, length, leave, disable=self.args.disable_tqdm or rank() != 0, ascii=True)

    def _due_hooks(self) -> tuple[bool, bool, bool]:
        do_eval = self._next_eval_step is not None and self.glbl_step >= self._next_eval_step
        do_chpt = self._next_chpt_step is not None and self.glbl_step >= self._next_chpt_step
        do_logg = self._next_logg_step is not None and self.glbl_step >= self._next_logg_step
        return do_eval, do_chpt, do_logg

    def _should_fire_hook_at_end_of_epoch(self, x_epochs: Optional[float]) -> bool:
        """
        Decide whether a hook with an epoch period should fire at the *end* of this epoch.

        Args:
            x_epochs: The number of epochs after which the hook should fire.
        """
        if x_epochs is None:
            return False
        if x_epochs == 1.0:
            return True
        if x_epochs < 1.0 and (1 / x_epochs).is_integer():
            return True
        if x_epochs > 1.0 and ((self.epoch_idx + 1) % x_epochs == 0):
            return True
        return False

    def run_due_hooks(
        self,
        tr_report: Optional[dict[str, float]],
        *,
        do_eval: Optional[bool] = None,
        do_chpt: Optional[bool] = None,
        do_logg: Optional[bool] = None,
        ad_eval: Optional[bool] = None,
        ad_chpt: Optional[bool] = None,
        ad_logg: Optional[bool] = None,
    ) -> None:
        """
        Conduct periodic activities, such as running validation, checkpointing, and logging.

        Args:
            tr_report: Optional dictionary of training statistics to include in the report.
            do_eval: If True, run validation even if not scheduled.
            do_chpt: If True, save a checkpoint even if not scheduled.
            do_logg: If True, update the logs even if not scheduled.
            ad_eval: Whether or not to advance the eval schedule. If None, will advance only if fired organically.
            ad_chpt: Whether or not to advance the checkpoint schedule. If None, will advance only if fired organically.
            ad_logg: Whether or not to advance the log schedule. If None, will advance only if fired organically.

        NOTE: logging will occur if any hook fires, but will not advance the logging schedule by default.
        NOTE: the early stoppper will be updated if validation was run and the validation schedule advanced.
        """
        due_eval, due_chpt, due_logg = self._due_hooks()

        do_eval = do_eval if do_eval is not None else due_eval
        do_chpt = do_chpt if do_chpt is not None else due_chpt
        do_logg = do_logg if do_logg is not None else due_logg

        # If no hooks are scheduled, exit prematurely.
        if not (do_eval or do_chpt or do_logg):
            return

        # Otherwise, assemble report to log.
        report = {
            "epoch": self.epoch_idx,
            "glbl_step": self.glbl_step,
            "lr": self.scheduler.get_last_lr()[0],
        }
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

        # Determine which schedules to advance.
        adv_eval = due_eval if ad_eval is None else ad_eval
        adv_chpt = due_chpt if ad_chpt is None else ad_chpt
        adv_logg = due_logg if ad_logg is None else ad_logg

        # Advance schedules that fired.
        if adv_eval:
            self._next_eval_step += self.eval_steps  # type: ignore[operator]
        if adv_chpt:
            self._next_chpt_step += self.chpt_steps  # type: ignore[operator]
        if adv_logg:
            self._next_logg_step += self.logg_steps  # type: ignore[operator]

        # Handle early stopping.
        if do_eval and adv_eval:
            self.stopper.step(report[f"{self.args.stopper_metric}"])
        if self.stopper.stop:
            self.print("Early stopping triggered.")
            self._done = True

    def forward(self, batch: Batch) -> Tensor:
        """
        Send a batch of inputs forward through the model.
        """
        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            return self.model(batch.get_inputs(), batch.get_guides(), **batch.get_otherkwds())  # type: ignore[no-any-return]

    def compute_loss(self, batch: Batch, outputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute the loss over a batch of examples.

        Args:
            batch: The batch of data.
            outputs: The model outputs for the batch.

        Returns:
            loss - The computed loss tensor.
            losses - A dictionary of detached individual loss components.
        """
        losses: dict[str, Tensor] = {}
        dtype = outputs.dtype
        device = outputs.device

        with torch.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.args.mp16):
            clf_loss: Tensor = self.loss_fn(outputs, batch.get_label())
            losses["clf_loss"] = clf_loss.to(device=device, dtype=dtype)

        if (last_aux_loss := get_last_aux_loss(self.model)) is not None:
            losses["aux_loss"] = last_aux_loss.to(device=device, dtype=dtype) * self.args.auxillary_loss_weight
        elif self.args.assert_auxillary_loss:
            raise RuntimeError("An auxillary loss was expected but not found.")

        loss = torch.tensor(0.0, dtype=dtype, device=device)
        for k in losses:
            loss += losses[k]

        losses = {k: losses[k].detach() for k in losses}

        return loss, losses

    def step(self, *, grad_scale: float = 1.0) -> Tensor:
        """
        Take an optimization step and return the norm of gradients.

        NOTE: If the GradScaler causes all optimization steps to be skipped, the LR scheduler
            will trigger a misleading warning that it was called before optimizer.step().
        """
        self.scaler.unscale_(self.optimizer)
        if grad_scale != 1.0:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_scale)
        norm: Tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        total_steps = getattr(self.scheduler, "total_steps", None)
        if total_steps is not None and self.glbl_step >= total_steps:
            self.print(
                f"[WARN] [rank {rank()}] [Trainer::step] scheduler {type(self.scheduler).__name__} reached {total_steps} steps. "
                f"Skipping `scheduler.step()` on global step {self.glbl_step}. "
                f"Learning rate will be held at {self.scheduler.get_last_lr()[0]:.6e}."
            )
        else:
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

    def print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        if self.args.disable_tqdm:
            print(*args, sep=sep, end=end)
        else:
            tqdm.write(sep.join(map(str, args)), sys.stderr, end=end)

    def get_dataloader_lengths(self, dataloader: DataLoader[Batch]) -> list[int]:
        """Return the maximum possible lengths of the dataloader on each distributed rank."""
        length = torch.tensor(largest_possible_dataloader_length(dataloader), dtype=torch.int64, device=self.device)
        if not is_dist():
            return [int(length.item())]
        lengths = [torch.zeros(1, dtype=length.dtype, device=self.device) for _ in range(world_size())]
        dist.all_gather(lengths, length, group=None)
        return [int(l.item()) for l in lengths]

    @property
    def tr_dataloader_lengths(self) -> list[int]:
        """Maximum possible lengths of the training dataloader on each distributed rank."""
        return self._tr_dataloader_lengths

    @property
    def vl_dataloader_lengths(self) -> list[int]:
        """Maximum possible lengths of the validation dataloader on each distributed rank."""
        return self._vl_dataloader_lengths

    @property
    def tr_dataloader_length(self) -> int:
        """Maximum possible length of the training dataloader across all distributed ranks, padded to multiple of `gradient_accumulation_steps`."""
        return round_up_to_multiple(max(self.tr_dataloader_lengths), self.args.gradient_accumulation_steps)

    @property
    def vl_dataloader_length(self) -> int:
        """Maximum possible length of the validation dataloader across all distributed ranks."""
        return max(self.vl_dataloader_lengths)

    @property
    def mp_dtype(self) -> torch.dtype:
        return mp_dtype(self.args.mp16, self.device)

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.tr_dataloader_length / self.args.gradient_accumulation_steps)

    def _epochs_to_steps(self, f: Optional[float]) -> Optional[int]:
        if f is None:
            return None
        return max(1, math.ceil(self.steps_per_epoch * f))

    @property
    def max_epochs(self) -> float:
        if self.args.max_epochs is not None:
            return self.args.max_epochs
        if self.args.max_steps is not None:
            return self.args.max_steps / self.steps_per_epoch
        raise RuntimeError("Unreachable.")

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
            if "tr_aux_loss" in results:
                d["tr_clf_loss"] = round(results["tr_clf_loss"], 3)
                d["tr_aux_loss"] = round(results["tr_aux_loss"], 3)
            d["tr_gpu_utl"] = round(results["tr_gpu_utl"], 2)
            d["tr_gpu_mem"] = round(results["tr_gpu_mem"] / (1024 ** 3), 2)
            d["tr_time"] = round(results["tr_time"], 0)
            d["tr_samples"] = int(results["tr_samples"])
            d["tr_throughput"] = round(results["tr_throughput"], 2)
        if any(k.startswith("vl_") for k in results):
            d["vl_loss"] = round(results["vl_loss"], 3)
            if "vl_aux_loss" in results:
                d["vl_clf_loss"] = round(results["vl_clf_loss"], 3)
                d["vl_aux_loss"] = round(results["vl_aux_loss"], 3)
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
        tr_loader: DataLoader[Batch],
        vl_loader: DataLoader[Batch],
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        optimizer_init: Optional[Callable[[ParamsT], Optimizer]] = None,
        get_param_groups: Callable[[Module], ParamsT] = lambda m: m.parameters(),
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
            optimizer = optimizer_init(get_param_groups(model))
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
            if torch.cuda.device_count() != len(rng_gpu):
                warnings.warn(f"Number of CUDA devices from checkpoint was {len(rng_gpu)} but number available now is {torch.cuda.device_count()}. CUDA RNG states may not be restored correctly.")
            else:
                torch.cuda.set_rng_state_all(rng_gpu)
        padbatch = torch.load(path / "padbatch.pt", map_location="cpu", weights_only=False)  # Load custom types (unsafe)
        loss_fn.load_state_dict(torch.load(path / "loss_fn.pt", map_location="cpu"))

        if is_dcp:
            # DCP/FSDP checkpoints require special handling to load the model and optimizer state dicts.
            # Concretely, the model must first be wrapped appropriately, before the optimizer is initialized.
            # Then both their state dicts can be loaded and from the checkpoint.
            model = wrap_model(model)
            optimizer = optimizer_init(get_param_groups(model))
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
            padbatch,
            loss_fn,
            optimizer,
            scheduler,
            stopper,
            device,
        )
        trainer.glbl_step = glbl_step
        trainer.epoch_idx = glbl_step // trainer.steps_per_epoch
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
