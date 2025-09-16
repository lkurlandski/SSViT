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
import datetime
from functools import partial
import gc
import inspect
import json
import math
import os
from pathlib import Path
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
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.tensor import DTensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FOrHSamples


def is_dist() -> bool:
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def rank() -> int:
    if is_dist():
        return int(dist.get_rank())
    return 0


def local_world_size() -> int:
    if "LOCAL_WORLD_SIZE" in os.environ:
        # Works when checking within subprocesses.
        return int(os.environ["LOCAL_WORLD_SIZE"])
    if is_dist():
        return int(torch.cuda.device_count())
    return 1


def world_size() -> int:
    if dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def maybe_join(model: Module, join: bool = True) -> ContextManager[None] | Join:
    """
    Returns a Join context if in distributed the model is Joinable, else a no-op context.

    This is particularily helpful for distributed workers processing different amounts of data.
    """
    if join and is_dist() and isinstance(model, Joinable):
        return Join([model])
    return contextlib.nullcontext()


def maybe_no_sync(model: Module, sync_now: bool) -> ContextManager[None]:
    """
    Returns a no_sync context if in distributed and the model has a no_sync method, else a no-op context.
    """
    if not is_dist() or sync_now or not hasattr(model, "no_sync"):
        return contextlib.nullcontext()
    return model.no_sync()  # type: ignore[no-any-return]


def cpu_rendezvous() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.ones(1, dtype=torch.int64)   # CPU tensor ⇒ Gloo on composable backends
        dist.all_reduce(t, op=dist.ReduceOp.SUM)  # acts as a barrier/rendezvous


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


FTensor = Union[BFloat16Tensor | HalfTensor | FloatTensor | DoubleTensor]
ITensor = Union[CharTensor | ByteTensor | ShortTensor | IntTensor | LongTensor]


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


def find_executable_batch_size(
    function: Optional[Callable[[int, int], Any]] = None,
    starting_batch_size: int = -1,
    starting_gradient_accumulation_steps: int = 1,
) -> Callable[[int, int], Any]:
    """Rerun a function with a smaller mini batch size if an OOM error is encountered.
    """

    if function is None:
        return partial(
            find_executable_batch_size,
            starting_batch_size=starting_batch_size,
            starting_gradient_accumulation_steps=starting_gradient_accumulation_steps,
        )

    batch_size = starting_batch_size
    gradient_accumulation_steps = starting_gradient_accumulation_steps


    def should_reduce_batch_size(exception: Exception) -> bool:
        statements = [
            "CUDA out of memory.",
            "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
            "DefaultCPUAllocator: can't allocate memory",
            "CUDA error: an illegal memory access was encountered",
            "Triton Error [CUDA]: an illegal memory access was encountered",
            "CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
        ]
        if isinstance(exception, RuntimeError) and len(exception.args) == 1:
            return any(err in exception.args[0] for err in statements)
        return False


    def decorator(*args: Any, **kwds: Any) -> Any:
        nonlocal batch_size, gradient_accumulation_steps

        function: Callable[[int, int], Any]

        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())

        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )

        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")

            try:
                return function(batch_size, gradient_accumulation_steps, *args, **kwds)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    gradient_accumulation_steps *= 2
                else:
                    raise

    return decorator


def pformat_dict(d: Mapping[str, int | float]) -> dict[str, str]:
    return {k: f"{v:.3f}" if isinstance(v, float) else f"{v}" for k, v in d.items()}


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


class Trainer:
    """
    Trainer class for training models with PyTorch.
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
        self.scheduler = scheduler if scheduler is not None else LambdaLR(optimizer, lambda _: 1.0)
        self.stopper = stopper if stopper is not None else EarlyStopper(patience=float("inf"))
        self.device = device if device is not None else next(self.model.parameters()).device
        self.padbatch = padbatch.clone().to(self.device) if padbatch is not None else None
        self.monitor = Monitor(device=self.device)
        self.log: list[Mapping[str, int | float]] = []
        self.best_epoch = -1
        self.best_metric = -float("inf") if args.lower_is_worse else float("inf")

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
        shutil.rmtree(self.args.outdir, ignore_errors=True)
        self.args.outdir.mkdir(parents=True, exist_ok=True)
        self.monitor.start()

        tr_report  = {"tr_loss": float("nan"), "tr_time": float("nan")}
        tr_report |= {f"tr_{k}": float("nan") for k, _ in self.monitor.get_report(clear=True).items()}
        vl_report  = self.evaluate()
        vl_report |= {f"vl_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
        report  = {"epoch": 0, "lr": float("nan")}
        report |= tr_report | vl_report
        self._update_cons(report)
        self._update_logs(report)
        self._update_best(report)
        self._update_save(report)

        pbar = list(range(1, self.args.epochs + 1))
        for epoch in pbar:
            self.monitor.clear()
            tr_report  = self.train()
            tr_report |= {f"tr_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
            vl_report  = self.evaluate()
            vl_report |= {f"vl_{k}": v for k, v in self.monitor.get_report(clear=True).items()}
            report  = {"epoch": epoch, "lr": self.scheduler.get_last_lr()[0]}
            report |= tr_report | vl_report
            self._update_cons(report)
            self._update_logs(report)
            self._update_best(report)
            self._update_save(report)
            self.scheduler.step()
            if self.stopper is not None:
                self.stopper.step(vl_report[self.args.metric])
                if self.stopper.stop:
                    break

        self.monitor.stop()

        return self

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        self.optimizer.zero_grad()
        self.model.train()
        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, FOrHSamples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Training...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)

        num_samples = 0
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))

        scaler = GradScaler(self.device.type, enabled=mp_dtype(self.args.mp16, self.device) == torch.float16)

        step = 0
        mini_step = -1
        with maybe_join(self.model):
            for mini_step, batch in iterable:
                print(f"[Worker {rank()}] Trainer::train {mini_step=}")  # FIXME: remove
                sync_gradients = (mini_step + 1) % self.args.gradient_accumulation_steps == 0

                # Compute normalized loss
                with maybe_no_sync(self.model, sync_gradients):
                    outputs = self.forward(batch)
                    loss = self.compute_loss(batch, outputs) / self.args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    results["tr_loss"] += loss.detach() * len(batch)

                # Update model weights
                if sync_gradients:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    step += 1

                num_samples += len(batch)

        if mini_step < 0:
            warnings.warn("No training mini-steps were taken, the training dataset may be empty.")
            num_samples = max(num_samples, 1)

        # Update weights if there are remaining gradients
        if mini_step > 0 and (mini_step + 1) % self.args.gradient_accumulation_steps != 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        # Average statistics over epoch
        report = {}
        for k in results:
            report[k] = results[k].item() / num_samples
        report["tr_time"] = time.time() - t_0

        return report

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.eval()
        dataloader = self.vl_loader
        iterable: Iterable[tuple[int, FOrHSamples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Validating...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)
        iterable = iter(iterable)

        num_samples = 0
        mini_step = -1
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.device))

        batch: FOrHSamples
        padbatch = self.padbatch

        procdev = self.device   # "cpu"
        procgrp = None          # _METRICS_PG

        with torch.no_grad():
            while True:
                # Get the next batch of data or padbatch if this rank is out of data.
                try:
                    mini_step, batch = next(iterable)
                    has_local = torch.tensor([1], dtype=torch.int32, device=procdev)
                    padbatch = batch.clone() if mini_step == 0 and padbatch is None else padbatch
                    real = 1.0
                except StopIteration:
                    if padbatch is None:
                        raise RuntimeError("This rank has no data and we could not create a dummy batch.")
                    mini_step, batch = mini_step + 1, padbatch
                    has_local = torch.tensor([0], dtype=torch.int32, device=procdev)
                    real = 0.0
                print(f"[Worker {rank()}] Trainer::evaluate {mini_step=} {real=}")  # FIXME: remove
                # If all ranks are out of data, all ranks should stop.
                if world_size() > 1:
                    has_any = has_local.clone()
                    dist.all_reduce(has_any, op=dist.ReduceOp.SUM, group=procgrp)
                    if int(has_any[0].item()) == 0:
                        break
                # Run the inputs through the model, but only count the metrics if this rank has real data.
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)
                results["vl_loss"] += loss * len(batch) * real
                for k in metrics:
                    results[f"vl_{k}"] += metrics[k] * len(batch) * real
                num_samples += len(batch) * int(real)

        if mini_step < 0:
            warnings.warn("No validation mini-steps were taken, the training dataset may be empty.")

        print(f"[Worker {rank()}] Trainer::evaluate synchronizing...")  # FIXME: remove
        if is_dist():
            vals = torch.empty((len(results) + 1,), dtype=torch.float64, device=procdev)
            vals[0] = num_samples
            keys = sorted(results.keys())
            for i, k in enumerate(keys, start=1):
                vals[i] = results[k].detach()
            print(f"[Worker {rank()}] Trainer::evaluate checking")  # FIXME: remove
            sizes = [None] * torch.distributed.get_world_size()
            dist.all_gather_object(sizes, 1 + len(keys), group=procgrp)
            assert len(set(sizes)) == 1
            print(f"[Worker {rank()}] Trainer::evaluate reducing")  # FIXME: remove
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=procgrp)
            num_samples = int(vals[0].item())
            for i, k in enumerate(keys, start=1):
                results[k] = vals[i]
        print(f"[Worker {rank()}] Trainer::evaluate normalizing...")  # FIXME: remove

        # Normalize metrics by number of samples
        num_samples = max(num_samples, 1)
        report = {}
        for k in results:
            report[k] = results[k].item() / num_samples
        report["vl_time"] = time.time() - t_0

        return report

    def forward(self, batch: FOrHSamples) -> FTensor:
        """
        Send a batch of inputs forward through the model.
        """
        with torch.autocast(self.device.type, dtype=mp_dtype(self.args.mp16, self.device), enabled=self.args.mp16):
            return self.model(batch.inputs, batch.characteristics)

    def compute_loss(self, batch: FOrHSamples, outputs: FTensor) -> FTensor:
        """
        Compute the loss over a batch of examples.
        """
        with torch.autocast(self.device.type, dtype=mp_dtype(self.args.mp16, self.device), enabled=self.args.mp16):
            return self.loss_fn.forward(outputs, batch.label)

    def compute_metrics(self, batch: FOrHSamples, outputs: FTensor) -> dict[str, Tensor]:
        """
        Compute the validation metrics over a set of examples.

        NOTE: This method (summing over batches) may not be ideal for metrics that
            may be more likely to be not-well defined on smaller sets of examples, e.g., F1.
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

    def _update_save(self, results: Mapping[str, int | float]) -> None:
        # TODO: figure out how to save and load properly (FSDP/DDP/GPU/CPU).
        return
        if local_rank() == 0:
            state_dict = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
            }
            checkpoint_id = str(self.args.outdir / f"ckpt_{results['epoch']}")
            print(f"[rank{dist.get_rank()}] Trainer::_update_save: saving to checkpoint.")
            dist_checkpoint.save(state_dict=state_dict, checkpoint_id=checkpoint_id)

        if dist.is_initialized():
            print(f"[rank{dist.get_rank()}] Trainer::_update_save: synchronizing workers.")
            if "nccl" in dist.get_backend():
                dist.barrier(device_ids=[local_rank()])
            else:
                dist.barrier()

        checkpoints = sorted(self.args.outdir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        for checkpoint in checkpoints:
            e = int(checkpoint.stem.split("_")[1])
            if e not in (self.best_epoch, results["epoch"]):
                checkpoint.unlink()


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
