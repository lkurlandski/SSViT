"""
Worker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.trainer import Trainer
from src.trainer import TrainerArgs
from tests.trainer.utils import MockDataset
from tests.trainer.utils import MockModel
from tests.trainer.utils import make_device_collate


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = MockModel().to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    ds = MockDataset()
    collate = make_device_collate(device)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)

    outdir = Path(os.environ["OUTDIR"])
    args = TrainerArgs(
        outdir=outdir,
        max_steps=6, max_epochs=None,
        eval_steps=3, eval_epochs=None,
        chpt_steps=None, chpt_epochs=None,
        logg_steps=2, logg_epochs=None,
    )
    pad = collate([ds[0]])
    trainer = Trainer(args, ddp_model, loader, loader, nn.CrossEntropyLoss(), padbatch=pad)  # type: ignore[arg-type]
    trainer()
    if rank == 0:
        with open(outdir / "ddp_summary.json", "w") as fp:
            json.dump({"glbl_step": trainer.glbl_step, "n_logs": len(trainer.log)}, fp, indent=2)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
