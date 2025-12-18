"""
Tests.
"""

from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Optional
from typing import Self
from typing import Sequence

import pytest
import torch
from torch import Tensor
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trainer import local_rank
from src.trainer import Trainer
from src.trainer import TrainerArgs


INPUT_DIM = 16
NUM_CLASSES = 2


class MockModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(INPUT_DIM, NUM_CLASSES)

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        out = self.layer(x)
        assert isinstance(out, Tensor)
        return out


class MockDataset(Dataset[tuple[Tensor, Tensor]]):

    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __getitem__(self, _: int) -> tuple[Tensor, Tensor]:
        x = torch.rand(INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (1,)).squeeze()
        return x, y

    def __len__(self) -> int:
        return self.num_samples


class MockBatch:

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def clone(self) -> Self:
        return self.__class__(self.x.clone(), self.y.clone())

    def to(self, device: torch.device, non_blocking: bool) -> Self:
        x = self.x.to(device=device, non_blocking=non_blocking)
        y = self.y.to(device=device, non_blocking=non_blocking)
        return self.__class__(x, y)

    def finalize(self, ftype: torch.dtype, itype: torch.dtype, ltype: torch.dtype) -> Self:
        x = self.x.to(dtype=ftype)
        y = self.y.to(dtype=ltype)
        return self.__class__(x, y)

    def get_label(self) -> Tensor:
        return self.y

    def get_inputs(self) -> Tensor | Sequence[Tensor]:
        return self.x

    def get_guides(self) -> Optional[Tensor] | Sequence[Optional[Tensor]]:
        return None


class MockCollateFn:

    def __call__(self, batch: list[tuple[Tensor, Tensor]]) -> MockBatch:
        xs, ys = zip(*batch)
        return MockBatch(torch.stack(xs), torch.tensor(ys))


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def run_torchrun(module: str, nproc: int, env: dict[str, str], timeout: int = 120) -> tuple[int, str]:
    """
    Launch a Python module, e.g. "tests.trainer.workers.ddp_worker", with torchrun.
    """
    torchrun = str(Path(sys.executable).parent / "torchrun")
    cmd = [
        torchrun,
        "-m",
        "--standalone",
        f"--nproc-per-node={nproc}",
        module,
    ]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("mp16", [False, True])
@pytest.mark.parametrize("ddp", [False, True])
def test(cuda: bool, mp16: bool, ddp: bool) -> None:
    if not cuda and (mp16 or ddp):
        pytest.skip("Running distributed or mixed-precision training without CUDA is not supported.")

    with tempfile.TemporaryDirectory() as outdir:
        if ddp:
            env = os.environ.copy()
            env["OUTDIR"] = str(outdir)
            env["USE_CUDA"] = "1" if cuda else "0"
            env["USE_MP16"] = "1" if mp16 else "0"
            env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
            env["PTW_NUM_THREADS"] = env.get("PTW_NUM_THREADS", "1")
            rc, out = run_torchrun("tests.test_trainer", nproc=2, env=env, timeout=60)
            if rc != 0:
                print(out)
            assert rc == 0
        else:
            main(Path(outdir), cuda, mp16, ddp)


def main(outdir: Path, cuda: bool, mp16: bool, ddp: bool) -> None:
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not cuda and mp16:
        raise RuntimeError("MP16 requires CUDA.")
    if not cuda and ddp:
        raise RuntimeError("DDP requires CUDA.")

    if ddp:
        dist.init_process_group(backend="cuda:nccl")
        torch.cuda.set_device(local_rank())
        device = torch.device(local_rank())
    else:
        device = torch.device("cuda" if cuda else "cpu")

    # Constants.
    num_samples = 511
    batch_size = 7
    gradient_accumulation_steps = 3
    max_epochs = 11.0
    eval_epochs = 0.50
    chpt_epochs = 0.25
    logg_epochs = 0.05

    args = TrainerArgs(
        outdir=Path(outdir),
        disable_tqdm=False,
        mp16=mp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_epochs=max_epochs,
        eval_epochs=eval_epochs,
        chpt_epochs=chpt_epochs,
        logg_epochs=logg_epochs,
    )
    model: MockModel | DistributedDataParallel = MockModel().to(device)
    if ddp:
        model = DistributedDataParallel(model)
    dataset = MockDataset(num_samples)
    collate_fn = MockCollateFn()
    loader: DataLoader[MockBatch] = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # type: ignore[arg-type]
    padbatch = collate_fn([dataset[i] for i in range(batch_size)])
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(args, model, loader, loader, padbatch, loss_fn)
    trainer()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    # OUTDIR="./tmp/output" USE_CUDA="1" USE_MP16="0" CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --start-method 'forkserver' --nnodes 1 --nproc-per-node 2 ./tests/test_trainer.py
    if "OUTDIR" not in os.environ:
        raise RuntimeError("Environment variable 'OUTDIR' not found.")
    if "USE_CUDA" not in os.environ:
        raise RuntimeError("Environment variable 'USE_CUDA' not found.")
    if "USE_MP16" not in os.environ:
        raise RuntimeError("Environment variable 'USE_MP16' not found.")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Environment variable 'LOCAL_RANK' not found.")
    if "RANK" not in os.environ:
        raise RuntimeError("Environment variable 'RANK' not found.")
    main(Path(os.environ["OUTDIR"]), os.environ["USE_CUDA"] == "1", os.environ["USE_MP16"] == "1", True)
