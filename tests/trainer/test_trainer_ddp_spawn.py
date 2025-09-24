"""
Tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

from tests.trainer.utils import cuda
from tests.trainer.utils import run_torchrun


@cuda
def test_trainer_ddp_torchrun() -> None:
    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least 1 CUDA device for nccl; switch worker to gloo for CPU runs.")
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        env = os.environ.copy()
        env["OUTDIR"] = str(outdir)

        rc, out = run_torchrun("tests.trainer.workers.ddp_worker", nproc=2, env=env)
        if rc != 0:
            print(out)
        assert rc == 0

        summary = (outdir / "ddp_summary.json").read_text()
        assert '"glbl_step": 6' in summary
