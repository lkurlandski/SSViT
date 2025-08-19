"""
Train and validation loops.
"""

from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import gc
import inspect
import json
import math
from pathlib import Path
import shutil
from statistics import mean
import sys
import time
from typing import Any
from typing import Callable
from typing import Optional
from typing import Self
from typing import TypeVar
import warnings

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src.architectures import ModelOutput
from src.data import CUDAPrefetcher
from src.data import Samples


@dataclass
class TrainerArgs:
    outdir: Path = Path("./output/tmp")
    device: torch.cuda.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 1
    tr_batch_size: int = 2
    vl_batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = True
    disable_tqdm: bool = False
    logging_steps: int = -1
    metric: str = "vl_loss"
    lower_is_worse: bool = False
    max_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    find_executable_batch_size: bool = False

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls(**{k: v for k, v in vars(namespace).items() if k in cls.__dataclass_fields__})


class TrainerArgumentParser(ArgumentParser):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--outdir", type=Path, default=Path("./output/tmp"))
        self.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.add_argument("--epochs", type=int, default=1)
        self.add_argument("--tr_batch_size", type=int, default=2)
        self.add_argument("--vl_batch_size", type=int, default=2)
        self.add_argument("--num_workers", type=int, default=0)
        self.add_argument("--pin_memory", action="store_true")
        self.add_argument("--disable_tqdm", action="store_true")
        self.add_argument("--logging_steps", type=int, default=-1)
        self.add_argument("--metric", type=str, default="vl_loss")
        self.add_argument("--lower_is_worse", action="store_true")
        self.add_argument("--max_norm", type=float, default=1.0)
        self.add_argument("--gradient_accumulation_steps", type=int, default=1)
        self.add_argument("--find_executable_batch_size", action="store_true")


class EarlyStopper:

    def __init__(self, patience: int = 0, threshold: float = 0.0001, lower_is_worse: bool = False) -> None:
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


class Trainer:
    """
    Trainer class for training models.
    """

    OVERWRITE_OUTDIRS = ("tmp", "test")
    tr_metric_keys = ("tr_loss", "tr_time")

    def __init__(
        self,
        args: TrainerArgs,
        model: Module,
        tr_dataset: Dataset | IterableDataset,
        vl_dataset: Dataset | IterableDataset,
        collate_fn: Callable[[list[Tensor]], Tensor],
        loss_fn: Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        stopper: Optional[EarlyStopper] = None,
    ) -> None:
        self.args = args
        self.model: Module = model.to(args.device)
        self.tr_dataset = tr_dataset
        self.vl_dataset = vl_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.log: list[dict[str, Any]] = []
        self.best_epoch = -1.0
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

        tr_metrics = {k: float("nan") for k in self.tr_metric_keys}
        vl_metrics = evaluate()
        d = {"epoch": 0.0, "learning_rate": float("nan"), "teacher_ratio": float("nan")} | tr_metrics | vl_metrics
        self._update_logs(d)
        self._update_best(d)
        self._update_save(d)

        pbar = self._get_pbar(range(1, self.args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            tr_metrics = train()
            vl_metrics = evaluate()
            learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.optimizer.defaults.get("lr", float("nan"))
            d = {"epoch": float(epoch), "learning_rate": learning_rate} | tr_metrics | vl_metrics
            self._update_logs(d)
            self._update_best(d)
            self._update_save(d)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.stopper is not None:
                self.stopper.step(vl_metrics[self.args.metric])
                if self.stopper.stop:
                    break
            if any(math.isnan(d[m]) or math.isinf(d[m]) for m in ("tr_loss", "vl_loss")):
                raise ValueError("NaN/Inf Loss Detected!")

        return self

    def train(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.train()
        dataloader = self.get_tr_dataloader()
        dataloader = CUDAPrefetcher(dataloader, self.args.device) if torch.cuda.is_available() else dataloader
        num_steps = math.ceil(len(dataloader) / self.args.gradient_accumulation_steps)
        results: defaultdict[str, list[float]] = defaultdict(lambda: [0.0] * num_steps)
        step = 0

        pbar = self._get_pbar(dataloader, total=len(dataloader), desc="Training...", leave=False)
        for mini_step, batch in enumerate(pbar):

            batch = batch.to(self.args.device, non_blocking=True)

            # Compute normalized loss, skip noisy losses
            outputs = self.forward(batch)
            loss, losses = self.compute_loss(batch, outputs)
            loss = loss / self.args.gradient_accumulation_steps
            losses = {k: v / self.args.gradient_accumulation_steps for k, v in losses.items()}
            # NOTE: this could theoretically cause the weights to never step, which should probably be handled.
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                warnings.warn(f"NaN/Inf Loss Detected! {mini_step=} loss={loss.item()}")
                continue
            loss.backward()

            # Add to running metrics
            results["tr_loss"][step] += loss.item()
            for k, v_1 in losses.items():
                results[f"tr_{k}"][step] += v_1

            # Update weights every `gradient_accumulation_steps` `mini_steps`
            condition_1 = (mini_step + 1) % self.args.gradient_accumulation_steps == 0
            condition_2 = (mini_step + 1) == len(dataloader)
            if condition_1 or condition_2:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1

            # Perform logging every `logging_steps` `steps` (not minibatch steps!)
            condition_1 = self.args.logging_steps > 0
            condition_2 = step > 0
            condition_3 = step % self.args.logging_steps == 0
            if condition_1 and condition_2 and condition_3:
                d: dict[str, float] = {"step": step}
                for k, v_2 in results.items():
                    start = step - self.args.logging_steps
                    stop = start + self.args.logging_steps
                    d[f"_{k}"] = mean(v_2[start:stop])
                print(self._fmt_dict(d))

        # Average statistics over epoch
        for k, v_3 in results.items():
            results[k] = mean(v_3)
        results["tr_time"] = time.time() - t_0

        # If all we got was NaN's for Inf's, add to the dict and let the caller handle.
        results["tr_loss"] = results.get("tr_loss", float("nan"))

        return dict(results)

    def evaluate(self) -> dict[str, float]:
        t_0 = time.time()

        self.model.eval()
        dataloader = self.get_vl_dataloader()
        iterable = self.get_wrapped_dataloader(dataloader, total=len(dataloader), desc="Validating...", leave=False)

        with torch.no_grad():
            results: defaultdict[str, list[float]] = defaultdict(list)
            for step, batch in iterable:
                batch = batch.to(self.args.device, non_blocking=True).decompress()
                outputs = self.forward(batch)
                loss = self.compute_loss(batch, outputs)
                metrics = self.compute_metrics(batch, outputs)
                results["loss"].append(loss.item())
                for k in metrics:
                    results[k].append(float(metrics[k]))

        report = {}
        for k in results:
            report[k] = mean(results[k])
        report["time"] = time.time() - t_0

        return dict(report)

    def forward(self, batch: Samples) -> ModelOutput:
        """Send a batch of inputs forward through the model.

        Args:
            batch: batch of inputs.

        Returns:
            tuple: model output(s).
        """
        return self.model.forward(batch.inputs)  # type: ignore[no-any-return]

    def compute_loss(self, batch: Samples, outputs: ModelOutput) -> Tensor:
        """Compute the loss over a batch of examples.

        Args:
            batch: batch of inputs.
            outputs: model output.

        Returns:
            loss over the batch.
        """
        return self.loss_fn.forward(outputs.logits, batch.label)

    def compute_metrics(self, batch: Samples, outputs: ModelOutput) -> dict[str, float]:
        """Compute the validation metrics over a set of examples.

        Args:
            batch: batch of inputs.
            outputs: model output.

        Returns:
            metrics for the set.
        """
        return {}

    def get_dataloader(self, dataset: Dataset | IterableDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

    def get_tr_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.tr_dataset, self.args.tr_batch_size, True)

    def get_vl_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.vl_dataset, self.args.vl_batch_size, False)

    def get_wrapped_dataloader(self, iterable: DataLoader, **kwds) -> Iterable[tuple[int, Samples]]:
        if self.args.device != torch.device("cpu"):
            iterable = CUDAPrefetcher(iterable, self.args.device)
        iterable = enumerate(iterable)
        if not self.args.disable_tqdm:
            iterable = tqdm(iterable, **kwds)
        return iterable

    def _update_logs(self, results: dict[str, float]) -> None:
        self.log.append(results)
        if self.args.logging_steps > 0:
            print(self._fmt_dict(results))
        with open(self.args.outdir / "results.jsonl", "a") as fp:
            fp.write(json.dumps(results) + "\n")

    def _update_best(self, results: dict[str, float]) -> None:
        if self.args.lower_is_worse and results[self.args.metric] > self.best_metric:
            self.best_epoch = results["epoch"]
            self.best_metric = results[self.args.metric]
        elif not self.args.lower_is_worse and results[self.args.metric] < self.best_metric:
            self.best_epoch = results["epoch"]
            self.best_metric = results[self.args.metric]

    def _update_save(self, results: dict[str, float]) -> None:
        torch.save(self.model, self.args.outdir / f"model_{results['epoch']}.pth")
        checkpoints = sorted(self.args.outdir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
        for checkpoint in checkpoints:
            e = int(checkpoint.stem.split("_")[1])
            if e not in (self.best_epoch, results["epoch"]):
                checkpoint.unlink()

    def _get_pbar(self, iterable: Iterable, **kwds) -> tqdm | Iterable:
        if self.args.disable_tqdm:
            return iterable
        return tqdm(iterable, **kwds)

    def _fmt_dict(self, d: Mapping[str, float | int]) -> dict[str, str]:
        return {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in d.items()}
