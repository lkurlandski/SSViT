"""
Tests.
"""

from pathlib import Path

import lief
import numpy as np
import pytest

from src.binanal import patch_binary
from src.binanal import SemanticGuider, SemanticSample


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

    # @pytest.mark.parametrize("file", FILES)
    # def test_create_parse_guide(self, file: Path):
    #     x = SemanticGuider.create_parse_guide(file.read_bytes())
    #     assert isinstance(x, np.ndarray)
    #     assert x.ndim == 2
    #     assert x.shape[1] == len(SemanticGuider.PARSE_GUIDE)
    #     assert np.all(np.isin(x, [0, 1, -1]))

    @pytest.mark.parametrize("w", [0, 16, 32])
    @pytest.mark.parametrize("file", FILES)
    def test_create_entropy(self, file: Path, w: int):
        b = file.read_bytes()
        print(len(b))
        x = SemanticGuider.create_entropy_guide(b, w)
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
        assert x.shape[0] == len(file.read_bytes())
        assert np.all(np.isnan(x[:w])) or w == 0
        assert np.all(np.isnan(x[-w + 1:])) or w == 0
        assert np.all(np.isfinite(x[w:-w]))

    @pytest.mark.parametrize("file", FILES)
    def test_create_characteristics_guide(self, file: Path):
        b = file.read_bytes()
        x = SemanticGuider.create_characteristics_guide(b)
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        assert x.shape[0] == len(b)
        assert x.shape[1] == len(SemanticGuider.CHARACTERISTICS)
        assert np.all(np.isin(x, [0, 1, -1]))

    @pytest.mark.parametrize("do_parse", [False, True])
    @pytest.mark.parametrize("do_entropy", [False, True])
    @pytest.mark.parametrize("do_characteristics", [False, True])
    def test_main(self, do_parse: bool, do_entropy: bool, do_characteristics: bool):
        file = FILES[0]
        b = file.read_bytes()

        guider = SemanticGuider(do_parse=do_parse, do_entropy=do_entropy, do_characteristics=do_characteristics)
        sample = guider(b)
        assert isinstance(sample, SemanticSample)

        buffer = sample.buffer
        assert isinstance(buffer, bytes)
        assert buffer == b

        parse = sample.parse
        if do_parse:
            assert isinstance(parse, np.ndarray)
            assert parse.shape == (len(b),)
            assert parse.dtype == np.int32
        else:
            assert parse is None

        entropy = sample.entropy
        if do_entropy:
            assert isinstance(entropy, np.ndarray)
            assert entropy.shape == (len(b),)
            assert entropy.dtype == np.float64
        else:
            assert entropy is None

        characteristics = sample.characteristics
        if do_characteristics:
            assert isinstance(characteristics, np.ndarray)
            assert characteristics.shape == (len(b), len(SemanticGuider.CHARACTERISTICS))
            assert characteristics.dtype == np.int32
        else:
            assert characteristics is None
