"""
Tests.
"""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.trainer import EarlyStopper
from src.trainer import TrainerArgs
from src.trainer import Trainer


class MockModel(nn.Module):  # type: ignore[misc]

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(16, 2)

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        return self.layer(x)


class MockDataset(Dataset):  # type: ignore[misc]

    def __init__(self) -> None:
        ...

    def __getitem__(self, _: int) -> tuple[Tensor, Tensor]:
        x = torch.rand(16)
        y = torch.randint(0, 2, (1,)).squeeze()
        return x, y

    def __len__(self) -> int:
        return 27


class MockSamples:

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def clone(self) -> MockSamples:
        return MockSamples(self.x.clone(), self.y.clone())

    @property
    def characteristics(self) -> Tensor:
        return None

    @property
    def inputs(self) -> Tensor:
        return self.x

    @property
    def label(self) -> Tensor:
        return self.y


class MockCollateFn:

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> MockSamples:
        xs, ys = zip(*batch)
        return MockSamples(torch.stack(xs), torch.tensor(ys))


class TestTrainer:

    def create_trainer(self, **kwds: Any) -> Trainer:
        # Reuse the helper from your existing TestTrainer if you prefer
        args = TrainerArgs(**kwds)
        model = MockModel()
        dataset = MockDataset()
        loader = DataLoader(dataset, batch_size=3, collate_fn=MockCollateFn())
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(args, model, loader, loader, loss_fn)
        return trainer

    def _read_jsonl(self, outdir: Path) -> list[dict[str, Any]]:
        path = outdir / "results.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_baseline_hooks_at_step0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            # Only set max_steps so schedule is step-driven; keep eval/log/ckpt enabled with defaults.
            tr = self.create_trainer(
                outdir=outdir,
                max_steps=1,          # stop quickly
                max_epochs=None,      # must be None when max_steps is set
            )
            tr()  # run

            logs = tr.log
            assert len(logs) >= 1, "Expected a baseline log at step 0"
            first = logs[0]
            assert int(first["glbl_step"]) == 0
            assert "epoch" in first
            # baseline was forced to eval+log+ckpt
            assert any(k.startswith("vl_") for k in first), "Baseline should include validation metrics"
            # baseline checkpoint exists
            ckpt0 = outdir / f"checkpoint-{0}"
            # Your code names checkpoints as f"checkpoint-{self.glbl_step}"
            # (baseline: glbl_step==0)
            assert ckpt0.exists(), "Expected baseline checkpoint directory at step 0"

    def test_eval_every_3_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            tr = self.create_trainer(
                outdir=outdir,
                max_steps=10, max_epochs=None,
                eval_steps=3, eval_epochs=None,   # eval every 3 steps
                logg_steps=None, logg_epochs=1.0, # logging defaults (epochly) still OK
                chpt_steps=None, chpt_epochs=None # disable scheduled ckpt (baseline still forces one)
            )
            tr()

            # Count how many logs include validation metrics.
            vl_logs = [r for r in tr.log if any(k.startswith("vl_") for k in r)]
            # Expected evals at steps: 0 (baseline forced), 3, 6, 9
            expected = 1 + len([s for s in range(1, 11) if s % 3 == 0 and s <= tr.max_steps])
            assert len(vl_logs) == expected, f"Expected {expected} eval logs, got {len(vl_logs)}"

            # Sanity: last eval should be at step 9 (<= 10)
            last_vl = max(vl_logs, key=lambda r: r["glbl_step"])
            assert int(last_vl["glbl_step"]) == 9

    def test_logging_only_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            tr = self.create_trainer(
                outdir=outdir,
                max_steps=12, max_epochs=None,
                eval_steps=None, eval_epochs=None,  # disable scheduled eval (baseline still forces eval)
                logg_epochs=0.5, logg_steps=None,   # log every half-epoch
                chpt_steps=None, chpt_epochs=None
            )
            tr()

            # steps_per_epoch = ceil( len(dataset)/batch_size / grad_accum ) = ceil(9/1) = 9
            spe = tr.steps_per_epoch
            assert spe == 9
            # half-epoch → ceil(9*0.5) = 5 → expected logs at 0 (baseline), 5, 10
            logs = tr.log
            glbl_steps = [int(r["glbl_step"]) for r in logs]
            assert 0 in glbl_steps
            assert 5 in glbl_steps
            assert 10 in glbl_steps

            # After baseline, later log entries should not include validation metrics
            # because eval schedule is disabled (only forced at step 0).
            post_baseline = [r for r in logs if int(r["glbl_step"]) > 0]
            assert all(not any(k.startswith("vl_") for k in r) for r in post_baseline), \
                "No scheduled evals expected after baseline"

    def test_checkpoint_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            tr = self.create_trainer(
                outdir=outdir,
                max_steps=10, max_epochs=None,
                chpt_steps=4, chpt_epochs=None,  # checkpoints at 4, 8
                eval_steps=None, eval_epochs=None,
                logg_steps=None, logg_epochs=None,
            )
            tr()

            # baseline checkpoint (0) + 4 + 8
            expected_steps = {0, 4, 8}
            existing = {int(p.name.split("-")[-1]) for p in outdir.iterdir()
                        if p.is_dir() and p.name.startswith("checkpoint-")}
            assert expected_steps.issubset(existing), f"Missing checkpoints for steps {expected_steps - existing}"

    def test_scheduler_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            # Build a trainer with an explicit scheduler (StepLR) stepped every 2 steps.
            model = MockModel()
            dataset = MockDataset()
            loader = DataLoader(dataset, batch_size=3, collate_fn=MockCollateFn())
            optim = AdamW(model.parameters(), lr=1e-3)
            sched = StepLR(optim, step_size=1, gamma=0.5)  # one scheduler.step() halves LR

            args = TrainerArgs(
                outdir=outdir,
                max_steps=5, max_epochs=None,
                schd_steps=2, schd_epochs=None,   # scheduler ticks at steps 2 and 4
                eval_steps=None, eval_epochs=None,
                chpt_steps=None, chpt_epochs=None,
                logg_steps=None, logg_epochs=None,
            )
            tr = Trainer(args, model, loader, loader, nn.CrossEntropyLoss(),
                         optimizer=optim, scheduler=sched)

            start_lr = tr.scheduler.get_last_lr()[0]
            tr()
            final_lr = tr.scheduler.get_last_lr()[0]

            # scheduler should have been stepped twice (at 2 and 4): lr * 0.5 * 0.5
            expected_lr = start_lr * (0.5 ** 2)
            assert math.isclose(final_lr, expected_lr, rel_tol=1e-6), f"{final_lr} vs {expected_lr}"

    def test_max_steps_hard_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            tr = self.create_trainer(outdir=outdir, max_steps=5, max_epochs=None)
            tr()
            assert tr.glbl_step == 5, f"Expected to stop exactly at 5 steps, got {tr.glbl_step}"

    def test_evaluate_restores_train_mode(self) -> None:
        tr = self.create_trainer()
        tr.model.train()
        assert tr.model.training is True
        _ = tr.evaluate()
        # After evaluate, we should be back in train mode
        assert tr.model.training is True

    def test_steps_per_epoch_and_epoch_field(self) -> None:
        tr = self.create_trainer()
        # dataset len = 27, batch_size=3 -> 9 minibatches; grad_accum=1 -> steps/epoch=9
        assert tr.steps_per_epoch == 9
        # mimic three steps and check epoch field math
        tr.glbl_step = 3
        e = tr.glbl_step / tr.steps_per_epoch
        # synthesize a log record like your code
        rec = {"epoch": tr.glbl_step / tr.steps_per_epoch, "glbl_step": tr.glbl_step, "lr": 1.0}
        assert math.isclose(rec["epoch"], e)

    def test_grad_accum_affects_steps_per_epoch(self) -> None:
        # grad_accum=4 → steps_per_epoch = ceil(9/4) = 3
        tr = self.create_trainer(gradient_accumulation_steps=4)
        assert tr.steps_per_epoch == 3

