"""
Tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from src.trainer import Trainer  # your Trainer
from src.trainer import TrainerArgs
from tests.trainer.utils import MockDataset
from tests.trainer.utils import MockModel
from tests.trainer.utils import cuda
from tests.trainer.utils import make_device_collate


@cuda
def test_trainer_cuda_end_to_end(tmp_path: Path, cuda0: torch.device) -> None:
    device = cuda0

    model = MockModel().to(device)
    ds = MockDataset()
    collate = make_device_collate(device)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)

    args = TrainerArgs(
        outdir=tmp_path,
        max_steps=8, max_epochs=None,
        eval_steps=4, eval_epochs=None,
        chpt_steps=4, chpt_epochs=None,
        logg_steps=2, logg_epochs=None,
        gradient_accumulation_steps=1,
    )
    tr = Trainer(args, model, loader, loader, nn.CrossEntropyLoss(), padbatch=collate([ds[0]]))  # type: ignore[arg-type]
    tr()

    assert tr.glbl_step == 8
    # steps_per_epoch: ceil(27/4)=7
    assert tr.steps_per_epoch == 7
    last = tr.log[-1]
    assert math.isclose(last["epoch"], last["glbl_step"] / 7, rel_tol=1e-6)
