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
from typing import TypedDict
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
    tr_batch_size: int = 1   # TODO: move
    vl_batch_size: int = 1   # TODO: move
    num_workers: int = 0     # TODO: move
    pin_memory: bool = True  # TODO: move
    disable_tqdm: bool = False
    logging_steps: int = -1
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    find_executable_batch_size: bool = False

    def __post_init__(self) -> None:
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                warnings.warn("CUDA device specified but not available, using CPU instead.")
                self.device = torch.device("cpu")
            if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                warnings.warn("CUDA_VISIBLE_DEVICES is not set, which could result in unexpected behavior if multiple GPUs are available.")

        if self.num_workers > 0:
            if os.environ.get("OMP_NUM_THREADS") is None:
                warnings.warn(f"Parallel data loading is enabled with num_workers={self.num_workers}, but OMP_NUM_THREADS is not set, which could result in CPU oversubscription.")
            if os.environ.get("MKL_NUM_THREADS") is None:
                warnings.warn(f"Parallel data loading is enabled with num_workers={self.num_workers}, but MKL_NUM_THREADS is not set, which could result in CPU oversubscription.")

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls(**{k: v for k, v in vars(namespace).items() if k in cls.__dataclass_fields__})


class TrainerArgumentParser(ArgumentParser):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--outdir", type=Path, default=Path("./output/tmp"))
        self.add_argument("--device", type=torch.device, default=torch.device("cpu"))
        self.add_argument("--epochs", type=int, default=1)
        self.add_argument("--tr_batch_size", type=int, default=1)  # TODO: move
        self.add_argument("--vl_batch_size", type=int, default=1)  # TODO: move
        self.add_argument("--num_workers", type=int, default=0)    # TODO: move
        self.add_argument("--pin_memory", action="store_true")     # TODO: move
        self.add_argument("--disable_tqdm", action="store_true")
        self.add_argument("--logging_steps", type=int, default=-1)
        self.add_argument("--metric", type=str, default="vl_loss")
        self.add_argument("--lower_is_worse", action="store_true")
        self.add_argument("--max_norm", type=float, default=1.0)
        self.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.add_argument("--find_executable_batch_size", action="store_true")


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
        self.scheduler = scheduler if scheduler is not None else LRScheduler(optimizer)
        self.stopper = stopper if stopper is not None else EarlyStopper(patience=float("inf"))
        self.log: list[Mapping[str, int | float]] = []
        self.best_epoch = -1
        self.best_metric = -float("inf") if args.lower_is_worse else float("inf")

    def __call__(self) -> Self:
        shutil.rmtree(self.args.outdir, ignore_errors=True)
        self.args.outdir.mkdir(parents=True, exist_ok=True)

        @find_executable_batch_size(  # type: ignore[call-arg, arg-type]
            starting_batch_size=self.args.vl_batch_size,
            starting_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        def evaluate(batch_size: int, gradient_accumulation_steps: int) -> dict[str, float]:
            nonlocal self
            self.args.vl_batch_size = batch_size
            return self.evaluate()

        @find_executable_batch_size(  # type: ignore[call-arg, arg-type]
            starting_batch_size=self.args.vl_batch_size,
            starting_gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )
        def train(batch_size: int, gradient_accumulation_steps: int) -> dict[str, float]:
            nonlocal self
            self.args.tr_batch_size = batch_size
            self.args.gradient_accumulation_steps = gradient_accumulation_steps
            return self.train()

        if not self.args.find_executable_batch_size:
            evaluate = self.evaluate
            train = self.train

        tr_report = {"tr_loss": float("nan"), "tr_time": float("nan")}
        vl_report = evaluate()
        report = {"epoch": 0, "lr": float("nan")} | tr_report | vl_report
        self._update_logs(report)
        self._update_best(report)
        self._update_save(report)

        pbar = tqdm(list(range(1, self.args.epochs + 1)), "Epochs", disable=self.args.disable_tqdm, ascii=True)
        for epoch in pbar:
            tr_report = train()
            vl_report = evaluate()
            report = {"epoch": 0, "lr": self.scheduler.get_last_lr()[0]} | tr_report | vl_report
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
        results = defaultdict(list)
        report = {}

        self.model.train()
        dataloader = self.tr_loader
        iterable: Iterable[tuple[int, Samples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Training...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)

        step = 0

        for mini_step, batch in iterable:

            batch = batch.decompress()

            # Compute normalized loss
            outputs = self.forward(batch)
            loss = self.compute_loss(batch, outputs) / self.args.gradient_accumulation_steps
            loss.backward()
            results["tr_loss"].append(loss.item())

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
                    start = step - self.args.logging_steps
                    stop = start + self.args.logging_steps
                    d[k] = mean(results[k][start:stop])
                print(pformat_dict(d))

        # Average statistics over epoch
        for k in results:
            report[k] = mean(results[k])
        report["tr_time"] = time.time() - t_0

        return report

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()
        results = defaultdict(list)
        report = {}

        self.model.eval()
        dataloader = self.vl_loader
        iterable: Iterable[tuple[int, Samples]] = enumerate(dataloader)
        iterable = tqdm(iterable, "Validating...", len(dataloader), False, disable=self.args.disable_tqdm, ascii=True)

        with torch.no_grad():
            for mini_step, batch in iterable:
                batch = batch.decompress()
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)
                results["vl_loss"].append(loss.item())
                for k in metrics:
                    results[f"vl_{k}"].append(float(metrics[k]))

        for k in results:
            report[k] = mean(results[k])
        report["vl_time"] = time.time() - t_0

        return report

    def forward(self, batch: Samples) -> FTensor:
        """Send a batch of inputs forward through the model.

        Args:
            batch: batch of inputs.

        Returns:
            tuple: model output(s).
        """
        return self.model.forward(batch.inputs, batch.guides.characteristics)

    def compute_loss(self, batch: Samples, outputs: FTensor) -> FTensor:
        """Compute the loss over a batch of examples.

        Args:
            batch: batch of inputs.
            outputs: model output.

        Returns:
            loss over the batch.
        """
        return self.loss_fn.forward(outputs, batch.label)

    def compute_metrics(self, batch: Samples, outputs: FTensor) -> dict[str, float]:
        """Compute the validation metrics over a set of examples.

        Args:
            batch: batch of inputs.
            outputs: model output.

        Returns:
            metrics for the set.
        """
        return {}

    def _update_logs(self, results: Mapping[str, int | float]) -> None:
        self.log.append(results)
        if self.args.logging_steps > 0:
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
