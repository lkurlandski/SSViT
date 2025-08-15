"""
Tests.
"""

import random
from pathlib import Path
import tempfile
from typing import Any

import pytest
import torch
from torch import LongTensor

from src.binanal import HierarchicalStructureNone
from src.data import BinaryDataset
from src.data import SemanticGuider
from src.data import SemanticGuides
from src.data import StructureMap
from src.data import Preprocessor

from tests import FILES


def get_random_bytes(size: int) -> bytes:
    return bytes([random.randint(0, 255) for _ in range(size)])


def get_random_str(size: int) -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=size))


class TestSemanticGuider:

    @pytest.mark.parametrize("do_parse", [False, True])
    @pytest.mark.parametrize("do_entropy", [False, True])
    @pytest.mark.parametrize("do_characteristics", [False, True])
    def test(self, do_parse: bool, do_entropy: bool, do_characteristics: bool) -> None:
        file = FILES[0]
        b = file.read_bytes()
        guider = SemanticGuider(do_parse=do_parse, do_entropy=do_entropy, do_characteristics=do_characteristics)
        sample = guider(b)
        assert bool(sample.parse is None) != do_parse
        assert bool(sample.entropy is None) != do_entropy
        assert bool(sample.characteristics is None) != do_characteristics


class TestDataset:

    num_samples = 100
    labels = [random.randint(0, 1) for _ in range(num_samples)]

    def check_dataset(self, dataset: BinaryDataset, files: list[Path], buffers: list[bytes]) -> None:
        assert len(dataset) == self.num_samples
        for i, sample in enumerate(dataset):
            n = sample.name
            x = sample.inputs
            y = sample.label
            assert isinstance(n, str)
            assert isinstance(x, LongTensor)
            assert isinstance(y, LongTensor)
            assert len(n) == 64
            assert tuple(x.shape) == (len(buffers[i]),)
            assert tuple(y.shape) == ()
            assert n == files[i].name.split(".")[0]
            assert torch.equal(x, torch.frombuffer(buffers[i], dtype=torch.uint8).to(torch.long))
            assert torch.equal(y, torch.tensor(self.labels[i], dtype=torch.long))

    def populate(self, tmpdir: Path) -> tuple[list[Path], list[bytes]]:
        files = []
        buffers = []
        for _ in range(self.num_samples):
            b = get_random_bytes(random.randint(512, 2048))
            f = tmpdir / f"{get_random_bytes(32).hex()}"
            f.write_bytes(b)
            files.append(f)
            buffers.append(b)
        return files, buffers

    def test_map_binary_dataset(self) -> None:

        class PreprocessorMock(Preprocessor):

            def __call__(self, mv: memoryview, _: Any) -> tuple[LongTensor, SemanticGuides, StructureMap]:
                return (
                    torch.frombuffer(mv, dtype=torch.uint8).to(torch.long),
                    SemanticGuides(None, None, None),
                    StructureMap(torch.full((len(mv), 1), False), {0: HierarchicalStructureNone.ANY})
                )

        preprocessor = PreprocessorMock(False, False, False)
        with tempfile.TemporaryDirectory() as tmpdir:
            files, buffers = self.populate(Path(tmpdir))
            dataset = BinaryDataset(files, self.labels, preprocessor)
            self.check_dataset(dataset, files, buffers)
