"""
Train and validation loops.
"""

from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import threading
import time
from typing import Any
from typing import Callable
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
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import str_to_bool
from src.data import FOrHSamples


FTensor = Union[BFloat16Tensor | HalfTensor | FloatTensor | DoubleTensor]
ITensor = Union[CharTensor | ByteTensor | ShortTensor | IntTensor | LongTensor]


@dataclass
class TrainerArgs:
    outdir: Path = Path("./output/tmp")
    device: torch.device = torch.device("cpu")
    epochs: int = 1
    disable_tqdm: bool = False
    logging_steps: int = -1
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                warnings.warn("CUDA device specified but not available, using CPU instead.")
                self.device = torch.device("cpu")
            if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                warnings.warn("CUDA_VISIBLE_DEVICES is not set, which could result in unexpected behavior if multiple GPUs are available.")

        if self.logging_steps > 0:
            warnings.warn(f"Logging every {self.logging_steps} `logging_steps` is enabled, which may slow down training.")

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls(**{k: v for k, v in vars(namespace).items() if k in cls.__dataclass_fields__})


class TrainerArgumentParser(ArgumentParser):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--outdir", type=Path, default=TrainerArgs.outdir)
        self.add_argument("--device", type=torch.device, default=TrainerArgs.device)
        self.add_argument("--epochs", type=int, default=TrainerArgs.epochs)
        self.add_argument("--disable_tqdm", type=str_to_bool, default=TrainerArgs.disable_tqdm)
        self.add_argument("--logging_steps", type=int, default=TrainerArgs.logging_steps)
        self.add_argument("--metric", type=str, default=TrainerArgs.metric)
        self.add_argument("--lower_is_worse", type=str_to_bool, default=TrainerArgs.lower_is_worse)
        self.add_argument("--max_norm", type=float, default=TrainerArgs.max_norm)
        self.add_argument("--gradient_accumulation_steps", type=int, default=TrainerArgs.gradient_accumulation_steps)


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


class Trainer:
    """
    Trainer class for training models.
    """

    PRINTKEYS = ("tr_loss", "tr_gpu_utl", "vl_loss", "vl_acc", "vl_gpu_utl")

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        vl_loader: Collection[FOrHSamples] | DataLoader[FOrHSamples],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
    ) -> None:
        self.args = args
        self.model: Module = model.to(args.device)
        self.tr_loader = tr_loader
        self.vl_loader = vl_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else LambdaLR(optimizer, lambda _: 1.0)
        self.stopper = stopper if stopper is not None else EarlyStopper(patience=float("inf"))
        self.monitor = Monitor(device=args.device)
        self.log: list[Mapping[str, int | float]] = []
        self.best_epoch = -1
        self.best_metric = -float("inf") if args.lower_is_worse else float("inf")

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

        pbar = tqdm(list(range(1, self.args.epochs + 1)), "Epochs", disable=self.args.disable_tqdm, ascii=True)
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

        self.model.train()
        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, FOrHSamples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Training...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)

        num_samples = 0
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.args.device))

        step = 0
        for mini_step, batch in iterable:

            # Compute normalized loss
            outputs = self.forward(batch)
            loss = self.compute_loss(batch, outputs) / self.args.gradient_accumulation_steps
            loss.backward()
            results["tr_loss"] += loss.detach() * len(batch)

            # Update weights every `gradient_accumulation_steps` `mini_steps`
            condition_1 = (mini_step + 1) % self.args.gradient_accumulation_steps == 0
            condition_2 = (mini_step + 1) == len(dataloader)
            if condition_1 or condition_2:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1

            # Perform logging every `logging_steps` `steps`
            condition_1 = self.args.logging_steps > 0
            condition_2 = step > 0
            condition_3 = step % self.args.logging_steps == 0
            if condition_1 and condition_2 and condition_3:
                d = {"step": step}
                for k in results:
                    d[k] = results[k].item() / num_samples
                print(pformat_dict(d))

            num_samples += len(batch)

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

        num_samples = 0
        results: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0, dtype=torch.float64, device=self.args.device))

        with torch.no_grad():
            for mini_step, batch in iterable:
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)
                results["vl_loss"] += loss * len(batch)
                for k in metrics:
                    results[f"vl_{k}"] += metrics[k] * len(batch)
                num_samples += len(batch)

        report = {}
        for k in results:
            report[k] = results[k].item() / num_samples
        report["vl_time"] = time.time() - t_0

        return report

    def forward(self, batch: FOrHSamples) -> FTensor:
        """
        Send a batch of inputs forward through the model.
        """
        return self.model.forward(batch.inputs, batch.characteristics)

    def compute_loss(self, batch: FOrHSamples, outputs: FTensor) -> FTensor:
        """
        Compute the loss over a batch of examples.
        """
        return self.loss_fn.forward(outputs, batch.label)

    def compute_metrics(self, batch: FOrHSamples, outputs: FTensor) -> dict[str, Tensor]:
        """
        Compute the validation metrics over a set of examples.

        NOTE: This method (summing over batches) may not be ideal for metrics that
            may be more likely to be not-well defined on smaller sets of examples, e.g., F1.
        """
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
        d = {k: results[k] for k in self.PRINTKEYS}
        print(pformat_dict(d))

    def _update_best(self, results: Mapping[str, int | float]) -> None:
        if self.args.lower_is_worse and results[self.args.metric] > self.best_metric:
            self.best_epoch = results["epoch"]  # type: ignore[assignment]
            self.best_metric = results[self.args.metric]
        elif not self.args.lower_is_worse and results[self.args.metric] < self.best_metric:
            self.best_epoch = results["epoch"]  # type: ignore[assignment]
            self.best_metric = results[self.args.metric]

    def _update_save(self, results: Mapping[str, int | float]) -> None:
        torch.save(self.model, self.args.outdir / f"model_{results['epoch']}.pth")
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

    def start(self) -> Monitor:
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
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return self

    def clear(self) -> Monitor:
        self._samples = defaultdict(list)
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
