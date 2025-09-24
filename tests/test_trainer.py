"""
Tests.
"""

from __future__ import annotations
import tempfile
from typing import Any
from typing import Optional

import torch
from torch import Tensor
from torch import nn
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
        args = TrainerArgs(**kwds)
        model = MockModel()
        dataset = MockDataset()
        loader = DataLoader(dataset, batch_size=3, collate_fn=MockCollateFn())
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(args, model, loader, loader, loss_fn)
        return trainer

    def test_evaluate(self) -> None:
        trainer = self.create_trainer()
        report = trainer.evaluate()
        assert isinstance(report, dict)
        for k, v in report.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_train(self) -> None:
        trainer = self.create_trainer()
        trainer.train()

    def test_call(self) -> None:
        trainer = self.create_trainer(max_epochs=2)
        trainer = trainer()
        assert len(trainer.log) == 3

    def test_checkpointing_functional(self) -> None:
        trainer = self.create_trainer()
        with tempfile.TemporaryDirectory() as path:
            trainer.to_checkpoint(path)
            newtrainer = Trainer.from_checkpoint(
                path,
                model=trainer.model,
                wrap_model=lambda model: model,
                tr_loader=trainer.tr_loader,
                vl_loader=trainer.vl_loader,
                loss_fn=trainer.loss_fn
            )
        assert isinstance(newtrainer, Trainer)
        assert isinstance(newtrainer.args, TrainerArgs)
        assert isinstance(newtrainer.model, MockModel)
        assert isinstance(newtrainer.loss_fn, nn.CrossEntropyLoss)
        assert isinstance(newtrainer.optimizer, torch.optim.AdamW)
        assert isinstance(newtrainer.scheduler, torch.optim.lr_scheduler.LambdaLR)
        assert isinstance(newtrainer.stopper, EarlyStopper)

    def test_checkpointing_correct(self) -> None:
        max_epochs = 2
        trainer = self.create_trainer()
        trainer.args.max_epochs = max_epochs
        trainer = trainer()
        with tempfile.TemporaryDirectory() as path:
            trainer.to_checkpoint(path)
            newtrainer = Trainer.from_checkpoint(
                path,
                model=trainer.model,
                wrap_model=lambda model: model,
                tr_loader=trainer.tr_loader,
                vl_loader=trainer.vl_loader,
                loss_fn=trainer.loss_fn
            )
        assert isinstance(newtrainer, Trainer)
        assert newtrainer.args == trainer.args
        assert newtrainer.glbl_step == trainer.glbl_step
        assert all((p1.data != p2.data).sum().item() == 0 for p1, p2 in zip(newtrainer.model.parameters(), trainer.model.parameters()))
