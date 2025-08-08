"""
Tests.
"""

import os
from pathlib import Path
from typing import Literal

import lief
import numpy as np
import pytest
import torch
from torch import Tensor
from torch import IntTensor
from torch import LongTensor
from torch import FloatTensor
from torch import DoubleTensor

from src.binanal import patch_binary
from src.binanal import SemanticGuider
from src.binanal import SemanticGuides


FILES = sorted(Path("./tests/data").iterdir())


class TestPatchPEFile:

    @pytest.mark.parametrize("file", FILES)
    def test_patch_none_1(self, file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        new = patch_binary(org, machine=None, subsystem=None)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == org_subsystem
        assert new == org

    @pytest.mark.parametrize("file", FILES)
    def test_patch_same(self, file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        new = patch_binary(org, machine=org_machine, subsystem=org_subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == org_subsystem
        assert new == org

    @pytest.mark.parametrize("file", FILES)
    def test_patch_machine(self, file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd machine
        if org_machine == lief.PE.Header.MACHINE_TYPES.THUMB:
            machine = lief.PE.Header.MACHINE_TYPES.POWERPC
        else:
            machine = lief.PE.Header.MACHINE_TYPES.THUMB
        new = patch_binary(org, machine=machine, subsystem=None)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == machine
        assert new_subsystem == org_subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most two bytes should differ
        assert np.sum(equal) >= len(equal) - 2, f"{np.sum(equal)} {len(equal)}"

    @pytest.mark.parametrize("file", FILES)
    def test_patch_subsystem(self, file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd subsystem
        if org_subsystem == lief.PE.OptionalHeader.SUBSYSTEM.XBOX:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.OS2_CUI
        else:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.XBOX
        new = patch_binary(org, machine=None, subsystem=subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == org_machine
        assert new_subsystem == subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most one byte should differ
        assert np.sum(equal) >= len(equal) - 1, f"{np.sum(equal)} {len(equal)}"

    @pytest.mark.parametrize("file", FILES)
    def test_patch_machine_subsystem(self, file: Path):
        org = file.read_bytes()
        pe = lief.parse(org)
        org_machine, org_subsystem = pe.header.machine, pe.optional_header.subsystem
        # Choose wierd machine and subsystem
        if org_machine == lief.PE.Header.MACHINE_TYPES.THUMB:
            machine = lief.PE.Header.MACHINE_TYPES.POWERPC
        else:
            machine = lief.PE.Header.MACHINE_TYPES.THUMB
        if org_subsystem == lief.PE.OptionalHeader.SUBSYSTEM.XBOX:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.OS2_CUI
        else:
            subsystem = lief.PE.OptionalHeader.SUBSYSTEM.XBOX
        new = patch_binary(org, machine=machine, subsystem=subsystem)
        pe = lief.parse(new)
        new_machine, new_subsystem = pe.header.machine, pe.optional_header.subsystem
        assert new_machine == machine
        assert new_subsystem == subsystem
        assert new != org
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))
        # At most three bytes should differ
        assert np.sum(equal) >= len(equal) - 3, f"{np.sum(equal)} {len(equal)}"


class TestSemanticGuider:

    def path_to_input_type(self, file: Path, input_type: type[str | Path | bytes]) -> str | Path | bytes:
        if input_type == str:
            return str(file)
        elif input_type == Path:
            return file
        else:
            return file.read_bytes()

    @pytest.mark.skip("NotImplemented")
    @pytest.mark.parametrize("file", FILES)
    @pytest.mark.parametrize("input_type", [str, Path, bytes])
    def test_create_parse_guide(self, file: Path, input_type: type[str | Path | bytes]):
        data = self.path_to_input_type(file, input_type)
        x = SemanticGuider.create_parse_guide(data)
        assert isinstance(x, IntTensor)
        assert x.ndim == 2
        assert x.shape[0] == os.path.getsize(file)
        assert x.shape[1] == len(SemanticGuider.PARSEERRORS)
        assert torch.all(torch.isin(x, torch.tensor([0, 1, -1])))

    @pytest.mark.parametrize("file", FILES)
    @pytest.mark.parametrize("input_type", [str, Path, bytes])
    @pytest.mark.parametrize("w", [0, 16, 32])
    def test_create_entropy(self, file: Path, input_type: type[str | Path | bytes], w: int):
        data = self.path_to_input_type(file, input_type)
        x = SemanticGuider.create_entropy_guide(data, w)
        assert isinstance(x, DoubleTensor)
        assert x.ndim == 1
        assert x.shape[0] == len(file.read_bytes())
        assert torch.all(torch.isnan(x[:w])) or w == 0
        assert torch.all(torch.isnan(x[-w + 1:])) or w == 0
        assert torch.all(torch.isfinite(x[w:-w]))

    @pytest.mark.parametrize("file", FILES)
    @pytest.mark.parametrize("input_type", [str, Path, bytes])
    def test_create_characteristics_guide(self, file: Path, input_type: type[str | Path | bytes]):
        data = self.path_to_input_type(file, input_type)
        x = SemanticGuider.create_characteristics_guide(data)
        assert isinstance(x, IntTensor)
        assert x.ndim == 2
        assert x.shape[0] == os.path.getsize(file)
        assert x.shape[1] == len(SemanticGuider.CHARACTERISTICS)
        assert torch.all(torch.isin(x, torch.tensor([0, 1, -1])))

    @pytest.mark.parametrize("do_parse", [False, True])
    @pytest.mark.parametrize("do_entropy", [False, True])
    @pytest.mark.parametrize("do_characteristics", [False, True])
    def test_main(self, do_parse: bool, do_entropy: bool, do_characteristics: bool):
        file = FILES[0]
        b = file.read_bytes()

        guider = SemanticGuider(do_parse=do_parse, do_entropy=do_entropy, do_characteristics=do_characteristics)
        sample = guider(b)
        assert isinstance(sample, SemanticGuides)

        parse = sample.parse
        if do_parse:
            assert isinstance(parse, Tensor)
        else:
            assert parse is None

        entropy = sample.entropy
        if do_entropy:
            assert isinstance(entropy, Tensor)
        else:
            assert entropy is None

        characteristics = sample.characteristics
        if do_characteristics:
            assert isinstance(characteristics, Tensor)
        else:
            assert characteristics is None
