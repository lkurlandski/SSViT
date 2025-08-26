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
from dataclasses import dataclass
from functools import partial
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import Self
from typing import Union
import warnings

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

from src.data import Samples


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
        self.add_argument("--disable_tqdm", action="store_true", default=TrainerArgs.disable_tqdm)
        self.add_argument("--logging_steps", type=int, default=TrainerArgs.logging_steps)
        self.add_argument("--metric", type=str, default=TrainerArgs.metric)
        self.add_argument("--lower_is_worse", action="store_true", default=TrainerArgs.lower_is_worse)
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
    return {k: f"{v:.6f}" if isinstance(v, float) else f"{v}" for k, v in d.items()}


class Trainer:
    """
    Trainer class for training models.
    """

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_loader: Collection[Samples] | DataLoader[Samples],
        vl_loader: Collection[Samples] | DataLoader[Samples],
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
        self.log: list[Mapping[str, int | float]] = []
        self.best_epoch = -1
        self.best_metric = -float("inf") if args.lower_is_worse else float("inf")

    def __call__(self) -> Self:
        shutil.rmtree(self.args.outdir, ignore_errors=True)
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        tr_report = {"tr_loss": float("nan"), "tr_time": float("nan")}
        vl_report = self.evaluate()
        report = {"epoch": 0, "lr": float("nan")} | tr_report | vl_report
        self._update_logs(report)
        self._update_best(report)
        self._update_save(report)

        pbar = tqdm(list(range(1, self.args.epochs + 1)), "Epochs", disable=self.args.disable_tqdm, ascii=True)
        for epoch in pbar:
            tr_report = self.train()
            vl_report = self.evaluate()
            report = {"epoch": epoch, "lr": self.scheduler.get_last_lr()[0]} | tr_report | vl_report
            self._update_logs(report)
            self._update_best(report)
            self._update_save(report)
            self.scheduler.step()
            if self.stopper is not None:
                self.stopper.step(vl_report[self.args.metric])
                if self.stopper.stop:
                    break

        return self

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.train()
        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, Samples]] = enumerate(dataloader)
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
        iterable: Iterable[tuple[int, Samples]] = enumerate(dataloader)
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

    def forward(self, batch: Samples) -> FTensor:
        """
        Send a batch of inputs forward through the model.
        """
        return self.model.forward(batch.inputs, batch.guides.characteristics)

    def compute_loss(self, batch: Samples, outputs: FTensor) -> FTensor:
        """
        Compute the loss over a batch of examples.
        """
        return self.loss_fn.forward(outputs, batch.label)

    def compute_metrics(self, batch: Samples, outputs: FTensor) -> dict[str, float]:
        """
        Compute the validation metrics over a set of examples.

        NOTE: This method (summing over batches) may not be ideal for metrics that
            may be more likely to be not-well defined on smaller sets of examples, e.g., F1.
        """
        labels = batch.label
        preds = torch.argmax(outputs, dim=1)

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

        return {
            "acc": acc,
            "pre": pre,
            "rec": rec,
            "f-1": f_1,
        }

    def _update_logs(self, results: Mapping[str, int | float]) -> None:
        self.log.append(results)
        print(pformat_dict(results))
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")

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
