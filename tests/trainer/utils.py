"""
Utilities.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Self
from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.trainer import Trainer
from src.trainer import TrainerArgs


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def cuda_available(min_gpus: int = 1) -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.device_count() >= min_gpus)


cuda = pytest.mark.skipif(not cuda_available(1), reason="CUDA unavailable")


class MockModel(nn.Module):  # type: ignore[misc]

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(16, 2)

    def forward(self, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        return self.layer(x)


class MockDataset(Dataset):  # type: ignore[misc]

    def __len__(self) -> int:
        return 27

    def __getitem__(self, _: int) -> Tuple[Tensor, int]:
        x = torch.rand(16)
        y = int(torch.randint(0, 2, (1,)).item())
        return x, y


class MockSamples:

    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def clone(self) -> MockSamples:
        return MockSamples(self.x.clone(), self.y.clone())

    @property
    def characteristics(self) -> Optional[Tensor]:
        return None

    @property
    def inputs(self) -> Tensor: 
        return self.x

    @property
    def label(self) -> Tensor:
        return self.y

    def to(self, device: str | torch.device) -> Self:
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def make_device_collate(device: torch.device) -> Callable[[list[tuple[Tensor, int]]], MockSamples]:

    def _collate(batch: list[tuple[Tensor, int]]) -> MockSamples:
        xs, ys = zip(*batch)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, device=device)
        return MockSamples(x, y)

    return _collate


def read_jsonl(outdir: Path) -> list[dict[str, Any]]:
    p = outdir / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def create_trainer(*, outdir: Path, device: torch.device, **kwds: Any) -> Trainer:
    model = MockModel().to(device)
    ds = MockDataset()
    collate = make_device_collate(device)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=False)
    args = TrainerArgs(outdir=outdir, **kwds)
    padbatch = collate([ds[0]])
    return Trainer(args, model, loader, loader, nn.CrossEntropyLoss(), padbatch=padbatch)  # type: ignore[arg-type]


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
