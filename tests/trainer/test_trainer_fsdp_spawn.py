"""
Tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tests.trainer.utils import cuda
from tests.trainer.utils import run_torchrun


@cuda
def test_trainer_fsdp_torchrun() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        env = os.environ.copy()
        env["OUTDIR"] = str(outdir)

        rc, out = run_torchrun("tests.trainer.workers.fsdp_worker", nproc=2, env=env)
        if rc != 0:
            print(out)
        assert rc == 0

        summary = (outdir / "fsdp_summary.json").read_text()
        assert '"glbl_step": 5' in summary
