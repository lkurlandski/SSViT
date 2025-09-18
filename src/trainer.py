"""
Train and validation loops.

TODO: update the Trainer class to maintain responsibility for wrapping models.
 - Pass in model_init and optimizer_init Callables to the Trainer constructor.
 - At each validation iteration, create a new model, set its state_dict, then wrap it.
 - This should circumvent the issues with FSDP in the validation loop.
"""

from __future__ import annotations
from argparse import Namespace
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
import contextlib
from dataclasses import dataclass
import datetime
from functools import partial
import gc
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
from typing import Any
from typing import Callable
from typing import ContextManager
from typing import Literal
from typing import Optional
from typing import Self
from typing import Union
import warnings

import numpy as np
import psutil
import pynvml
import torch
from torch import Tensor
from torch import BFloat16Tensor
from torch import HalfTensor
from torch import FloatTensor
from torch import DoubleTensor
from torch import CharTensor
from torch import ByteTensor
from torch import ShortTensor
from torch import IntTensor
from torch import LongTensor
from torch import distributed as dist
from torch.amp import GradScaler
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed.algorithms.join import Join
from torch.distributed.algorithms.join import Joinable
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel import DataParallel
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FOrHSamples

# TODO: Write a `Batch` Protocol instead of using FOrHSamples directly.
# Currently, it must define __len__ (number of samples), .clone() (a deep copy),
# .inputs (Tensor), .characteristics (Tensor|None), and .label (Tensor).


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
    return model.no_sync()  # type: ignore[no-any-return]


_METRICS_PG: Optional[dist.ProcessGroup] = None
def init_metrics_group() -> None:
    """Create a small, CPU-only comms group for scalars/metrics."""
    global _METRICS_PG
    if _METRICS_PG is not None:
        raise RuntimeError("Metrics process group is already initialized.")
    if not is_dist():
        return
    _METRICS_PG = dist.new_group(ranks=list(range(world_size())), backend="gloo")
    dist.barrier(group=_METRICS_PG)


@dataclass
class TrainerArgs:
    outdir: Path = Path("./output/tmp")
    epochs: int = 1
    disable_tqdm: bool = False
    logging_steps: int = -1
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mp16: bool = False

    def __post_init__(self) -> None:
        if self.logging_steps > 0:
            warnings.warn(f"Logging every {self.logging_steps} `logging_steps` is enabled, which may slow down training.")

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
        if not torch.cuda.is_available(True):
            return torch.float16
    raise RuntimeError(f"Unsupported device for mixed precision: {device}.")


def unwrapddp(model: Module) -> Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def is_fsdp2(model: Module) -> bool:
    return any(isinstance(p, DTensor) for p in model.parameters())


class Trainer:
    """
    Trainer class for training models with PyTorch.

    Parameters
    ----------
    epoch (int): this is the current number of completed epochs, i.e., it is incremented after train().
        If starting from scratch, epoch is 0 at initialization. If you want to start training ON the i-th
        epoch, set trainer.epoch = i - 1 before calling trainer().

    Notes
    -----
    - If the GradScaler causes all optimization steps to be skipped, the LR scheduler will trigger a
        warning that it was called before optimizer.step().
    """

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        vl_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
        device: Optional[torch.device] = None,
        padbatch: Optional[FOrHSamples] = None,
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
        self.padbatch = padbatch.to(self.device) if padbatch is not None else None
        self.monitor = Monitor(device=self.device)
        self.log: list[Mapping[str, int | float]] = []
        self.best_epoch = -1
        self.best_metric = -float("inf") if args.lower_is_worse else float("inf")
        self.glbl_step = -1
        self.epoch = -1

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
        self.args.outdir.mkdir(parents=True, exist_ok=True)
        self.monitor.start()

        # NOTE: epoch and glbl_step are -1 at initialization, but may have been modified before __call__,
        # especially by Trainer.from_checkpoint, which operates in presicely this manner.
        self.epoch     = max(self.epoch, 0)
        self.glbl_step = max(self.glbl_step, 0)

        # Conduct an initial validation on the naked model before any training, if starting from epoch 0.
        if self.epoch == 0:
            tr_report  = {"tr_loss": float("nan"), "tr_time": float("nan")}
            tr_report |= {f"tr_{k}": float("nan") for k, _ in self.monitor.get_report(clear=True).items()}
            vl_report  = self.evaluate()
            vl_report |= {f"vl_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
            report  = {"epoch": 0, "glbl_step": self.glbl_step, "lr": float("nan")}
            report |= tr_report | vl_report
            self._update_cons(report)
            self._update_logs(report)
            self._update_best(report)
            self.to_checkpoint(self.args.outdir / f"checkpoint-{self.epoch}.pt")

        # Conduct training/validation for the remaining epochs (`self.args.epochs`` in total).
        for _ in range(self.args.epochs - self.epoch):
            self.epoch += 1
            self.monitor.clear()
            tr_report  = self.train()
            tr_report |= {f"tr_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
            vl_report  = self.evaluate()
            vl_report |= {f"vl_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
            report  = {"epoch": self.epoch, "glbl_step": self.glbl_step, "lr": self.scheduler.get_last_lr()[0]}
            report |= tr_report | vl_report
            self._update_cons(report)
            self._update_logs(report)
            self._update_best(report)
            self.to_checkpoint(self.args.outdir / f"checkpoint-{self.epoch}.pt")
            self.scheduler.step()
            if self.stopper is not None:
                self.stopper.step(vl_report[self.args.metric])
                if self.stopper.stop:
                    break

        self.monitor.stop()

        return self

    def train(self) -> dict[str, float]:
        """
        Train the model for one epoch.
        """
        t_0 = time.time()

        self.optimizer.zero_grad()
        self.model.train()
        scaler = GradScaler(self.device.type, enabled=mp_dtype(self.args.mp16, self.device) == torch.float16)

        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, FOrHSamples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Training...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))

        locl_step = 0   # local step within this epoch
        mini_step = -1  # mini-step within this epoch
        def step() -> None:
            """Update the model parameters."""
            nonlocal locl_step
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            locl_step += 1
            self.glbl_step += 1

        with maybe_join(self.model):
            for mini_step, batch in iterable:
                results["num_samples"] += len(batch)
                sync_gradients = (mini_step + 1) % self.args.gradient_accumulation_steps == 0

                # Compute normalized loss
                with maybe_no_sync(self.model, sync_gradients):
                    outputs = self.forward(batch)
                    loss = self.compute_loss(batch, outputs) / self.args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    results["tr_loss"] += loss.detach() * len(batch)

                # Update model weights
                if sync_gradients:
                    step()

        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")

        # Update weights if there are remaining gradients
        if mini_step > 0 and (mini_step + 1) % self.args.gradient_accumulation_steps != 0:
            step()

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
        report["tr_time"] = time.time() - t_0

        return report

    def evaluate(self) -> dict[str, float]:
        """
        Evaluate the model on the validation set.

        NOTE: We use a padding-batch strategy so each worker performs the same number of forward passes.
        This helps prevent deadlocks and hanging. Unfortunately, this still does not work with FSDP models.
        """
        t_0 = time.time()

        self.model.eval()

        dataloader = self.vl_loader
        iterable: Iterable[tuple[int, FOrHSamples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Validating...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)
        iterable = iter(iterable)

        mini_step = -1
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))

        batch: FOrHSamples
        padbatch = self.padbatch

        procdev = torch.device("cpu")
        procgrp = _METRICS_PG

        with torch.no_grad():
            while True:
                # Get the next batch of data or padbatch if this rank is out of data.
                try:
                    mini_step, batch = next(iterable)
                    has_local = torch.tensor([1], dtype=torch.int32, device=procdev)
                    padbatch = batch.clone() if mini_step == 0 and padbatch is None else padbatch
                    real = 1.0
                except StopIteration:
                    if world_size() == 1:
                        break
                    if padbatch is None:
                        raise RuntimeError("This rank has no data and we could not create a dummy batch.")
                    mini_step, batch = mini_step + 1, padbatch
                    has_local = torch.tensor([0], dtype=torch.int32, device=procdev)
                    real = 0.0
                # If all ranks are out of data, all ranks should stop.
                if world_size() > 1:
                    has_any = has_local.clone()
                    dist.all_reduce(has_any, op=dist.ReduceOp.SUM, group=procgrp)
                    if int(has_any[0].item()) == 0:
                        break
                # Run the inputs through the model, but only count the metrics if this rank has real data.
                results["num_samples"] += len(batch) * int(real)
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)
                results["vl_loss"] += loss * len(batch) * real
                for k in metrics:
                    results[f"vl_{k}"] += metrics[k] * len(batch) * real

        if mini_step < 0:
            raise RuntimeError(f"[rank {rank()}] empty dataloader.")

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

        return report

    def forward(self, batch: FOrHSamples) -> Tensor:
        """
        Send a batch of inputs forward through the model.
        """
        with torch.autocast(self.device.type, dtype=mp_dtype(self.args.mp16, self.device), enabled=self.args.mp16):
            return self.model(batch.inputs, batch.characteristics)

    def compute_loss(self, batch: FOrHSamples, outputs: Tensor) -> Tensor:
        """
        Compute the loss over a batch of examples.
        """
        with torch.autocast(self.device.type, dtype=mp_dtype(self.args.mp16, self.device), enabled=self.args.mp16):
            return self.loss_fn(outputs, batch.label)

    def reduce_results(self, results: dict[str, Tensor], check: bool = True) -> dict[str, Tensor]:
        """
        Reduce the results dictionary across all distributed ranks using a separate CPU process group.
        """
        # Verify all ranks have the same keys
        keys = sorted(results.keys())
        if check:
            sizes = [None] * world_size()
            dist.all_gather_object(sizes, len(keys), group=_METRICS_PG)
            if len(set(sizes)) != 1:
                raise RuntimeError(f"Metrics keys do not match across ranks. Detected key sets with sizes: {sizes}.")
        # Sum-reduce values across workers
        vals = torch.empty((len(results)), dtype=torch.float64, device="cpu")
        for i, k in enumerate(keys):
            vals[i] = results[k].detach()
        dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=_METRICS_PG)
        return {k: vals[i] for i, k in enumerate(keys)}

    def compute_metrics(self, batch: FOrHSamples, outputs: Tensor) -> dict[str, Tensor]:
        """
        Compute the validation metrics over a set of examples.

        FIXME: we need to compute the global tp, tn, fp, fn across all samples, then compute the metrics.
        """
        outputs = outputs.to(torch.float32)
        labels = batch.label
        preds = torch.argmax(outputs, dim=1)

        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        acc = (tp + tn) / (tp + tn + fp + fn).clamp_min_(1)
        pre = tp / (tp + fp).clamp_min_(1)
        rec = tp / (tp + fn).clamp_min_(1)
        f_1 = 2 * pre * rec / (pre + rec).clamp_min_(1)

        return {
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f-1": f_1,
        }

    def _update_logs(self, results: Mapping[str, int | float]) -> None:
        self.log.append(results)
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")

    def _update_cons(self, results: Mapping[str, int | float]) -> None:
        d = {}
        d["epoch"] = results["epoch"]
        d["tr_loss"] = round(results["tr_loss"], 3)
        d["vl_loss"] = round(results["vl_loss"], 3)
        d["vl_acc"] = round(results["vl_acc"], 3)
        d["tr_gpu_utl"] = round(results["tr_gpu_utl"], 2)
        d["vl_gpu_utl"] = round(results["vl_gpu_utl"], 2)
        d["tr_time"] = int(round(results["tr_time"], 0)) if not math.isnan(results["tr_time"]) else float("nan")
        d["vl_time"] = int(round(results["vl_time"], 0)) if not math.isnan(results["vl_time"]) else float("nan")
        print(d)

    def _update_best(self, results: Mapping[str, int | float]) -> None:
        if self.args.lower_is_worse and results[self.args.metric] > self.best_metric:
            self.best_epoch = results["epoch"]  # type: ignore[assignment]
            self.best_metric = results[self.args.metric]
        elif not self.args.lower_is_worse and results[self.args.metric] < self.best_metric:
            self.best_epoch = results["epoch"]  # type: ignore[assignment]
            self.best_metric = results[self.args.metric]

    def to_checkpoint(self, file: Path) -> None:
        if rank() != 0:
            return

        if is_fsdp2(self.model):
            warnings.warn("Trainer::to_checkpoint does not save FSDP models' optimizer state across all ranks.")

        payload = {
            "args": pickle.dumps(self.args),
            "log": pickle.dumps(self.log),
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "epoch": self.epoch,
            "glbl_step": self.glbl_step,
            "model": get_model_state_dict(unwrapddp(self.model), options=StateDictOptions(full_state_dict=True, cpu_offload=True)),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "stopper": pickle.dumps(self.stopper),
            "padbatch": self.padbatch,
            "rng-cpu": torch.random.get_rng_state(),
            "rng-gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        torch.save(payload, file)

    @classmethod
    def from_checkpoint(
        cls,
        file: str | Path,
        model: Module,
        tr_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        vl_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        loss_fn: Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
        device: Optional[torch.device] = None,
        padbatch: Optional[FOrHSamples] = None,
    ) -> Trainer:
        # NOTE: this will not create the optimizer for FSDP models properly.

        if isinstance(model, (FullyShardedDataParallel, DistributedDataParallel, DataParallel)) or is_fsdp2(model):
            raise RuntimeError("Trainer.from_checkpoint requires a non-distributed model instance.")

        payload: dict[str, Any] = torch.load(file, map_location="cpu")

        # These must match the default in the __init__ method
        optimizer = optimizer if optimizer is not None else AdamW(model.parameters())
        scheduler = scheduler if scheduler is not None else LambdaLR(optimizer, lambda _: 1.0)

        # log, epoch, glbl_step, best_epoch, best_metric
        log = pickle.loads(payload.pop("log"))
        epoch = payload.pop("epoch")
        glbl_step = payload.pop("glbl_step")
        best_epoch = payload.pop("best_epoch")
        best_metric = payload.pop("best_metric")

        # model, optimzier
        set_model_state_dict(model, payload.pop("model"))
        optimizer.load_state_dict(payload.pop("optimizer"))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device if device is not None else next(model.parameters()).device)

        # loss_fn, scheduler, stopper, padbatch
        loss_fn.load_state_dict(payload.pop("loss_fn"))
        scheduler.load_state_dict(payload.pop("scheduler"))
        stopper = pickle.loads(payload.pop("stopper"))
        padbatch = payload.pop("padbatch")

        # rng-cpu, rng-gpu
        rng_cpu = payload.pop("rng-cpu")
        if rng_cpu is not None:
            torch.random.set_rng_state(rng_cpu)
        rng_gpu = payload.pop("rng-gpu")
        if rng_gpu is not None:
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_gpu)
            else:
                warnings.warn("Saved GPU state detected but CUDA is not available.")

        # args
        args: TrainerArgs = pickle.loads(payload.pop("args"))

        # trainer
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
        trainer.epoch = epoch
        trainer.glbl_step = glbl_step
        trainer.best_epoch = best_epoch
        trainer.best_metric = best_metric
        trainer.log = log

        if payload:
            raise RuntimeError(f"Unexpected keys in checkpoint payload: {list(payload.keys())}.")

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
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.dindex) if self.dindex is not None else self._get_handle_from_device(self.device)
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
        d = {k: np.mean(v) if len(v) > 0 else float("nan") for k, v in self._samples.items()}
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
