"""
Tests.
"""

import math
import os
from pathlib import Path
import sys
import tempfile

import lief
import numpy as np
from numpy import typing as npt
import pytest

from src.binanal import patch_binary
from src.binanal import ParserGuider
from src.binanal import CharacteristicGuider
from src.binanal import StructureParser
from src.binanal import BinaryCreator

from tests import FILES


class TestPatchPEFile:

    @pytest.mark.parametrize("file", FILES)
    def test_patch_none_1(self, file: Path) -> None:
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
    def test_patch_same(self, file: Path) -> None:
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
    def test_patch_machine(self, file: Path) -> None:
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
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))  # type: ignore[no-untyped-call]
        # At most two bytes should differ
        assert np.sum(equal) >= len(equal) - 2, f"{np.sum(equal)} {len(equal)}"

    @pytest.mark.parametrize("file", FILES)
    def test_patch_subsystem(self, file: Path) -> None:
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
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))  # type: ignore[no-untyped-call]
        # At most one byte should differ
        assert np.sum(equal) >= len(equal) - 1, f"{np.sum(equal)} {len(equal)}"

    @pytest.mark.parametrize("file", FILES)
    def test_patch_machine_subsystem(self, file: Path) -> None:
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
        equal = np.equal(np.frombuffer(org, dtype=np.uint8), np.frombuffer(new, dtype=np.uint8))  # type: ignore[no-untyped-call]
        # At most three bytes should differ
        assert np.sum(equal) >= len(equal) - 3, f"{np.sum(equal)} {len(equal)}"


def _path_to_input_type(file: Path, input_type: type[str | Path | bytes]) -> str | Path | bytes:
    if input_type == str:
        return str(file)
    elif input_type == Path:
        return file
    else:
        return file.read_bytes()


class TestParserGuider:

    @pytest.mark.parametrize("file", FILES)
    def test_build_simple_guide(self, file: Path) -> None:
        parser = ParserGuider(file)
        g = parser(True)
        assert g.ndim == 2
        assert g.shape[0] == os.path.getsize(file)
        assert g.shape[1] == len(ParserGuider.PARSEERRORS)

    @pytest.mark.parametrize("file", FILES)
    def test_build_complex_guide(self, file: Path) -> None:
        parser = ParserGuider(file)
        g = parser(False)
        assert g.ndim == 2
        assert g.shape[0] == os.path.getsize(file)
        assert g.shape[1] == len(ParserGuider.PARSEERRORS)


class TestCharacteristicGuider:

    @pytest.mark.parametrize("file", FILES)
    @pytest.mark.parametrize("input_type", [str, Path, bytes])
    def test_bool(self, file: Path, input_type: type[str | Path | bytes]) -> None:
        data = _path_to_input_type(file, input_type)
        x = CharacteristicGuider(data, use_packed=False)()
        assert x.ndim == 2
        assert x.shape[0] == os.path.getsize(file)
        assert x.shape[1] == len(CharacteristicGuider.CHARACTERISTICS)
        assert np.issubdtype(x.dtype, np.bool_)

    @pytest.mark.parametrize("file", FILES)
    @pytest.mark.parametrize("input_type", [str, Path, bytes])
    def test_bit(self, file: Path, input_type: type[str | Path | bytes]) -> None:
        data = _path_to_input_type(file, input_type)
        x = CharacteristicGuider(data, use_packed=True)()
        assert x.ndim == 2
        assert x.shape[0] == math.ceil(os.path.getsize(file) / 8)
        assert x.shape[1] == len(CharacteristicGuider.CHARACTERISTICS)
        assert np.issubdtype(x.dtype, np.uint8)

    @pytest.mark.parametrize("file", FILES)
    def test_equivalence(self, file: Path, input_type: type[str | Path | bytes] = str) -> None:
        data = _path_to_input_type(file, input_type)
        x_1 = CharacteristicGuider(data, use_packed=False)()
        x_2 = CharacteristicGuider(data, use_packed=True)()

        z = np.packbits(x_1, axis=0, bitorder="little")
        assert z.shape == x_2.shape
        assert np.all(x_2 == z)

        z = np.unpackbits(x_2, axis=0, count=x_1.shape[0], bitorder="little")
        assert z.shape == x_1.shape
        assert np.all(x_1 == z)


class TestStructureParser:

    def _create_section_with_characteristics(self, name: str, characteristics: int) -> lief.PE.Section:
        section = lief.PE.Section(name)
        section.characteristics = characteristics
        return section

    def _check_ranges(self, ranges: list[tuple[int, int]], size: int = sys.maxsize) -> None:
        assert isinstance(ranges, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges), f"{ranges}"
        assert all(isinstance(start, int) and isinstance(end, int) for start, end in ranges)
        assert all(0 <= start < end <= size for start, end in ranges)

    def _get_pe_size(self, pe: lief.PE.Binary) -> int:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pe.write(tmp_file.name)
            return os.path.getsize(tmp_file.name)

    def _get_pe_bytes(self, pe: lief.PE.Binary) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pe.write(tmp_file.name)
            return Path(tmp_file.name).read_bytes()

    def _valid_sections(self, pe: lief.PE.Binary, size: int) -> list[lief.PE.Section]:
        return [s for s in pe.sections if s.sizeof_raw_data > 0 and s.offset < size]

    def _valid_directories(self, pe: lief.PE.Binary, size: int) -> list[lief.PE.DataDirectory]:
        v = []
        for d in pe.data_directories:
            if d.size <= 0:
                continue
            if d.rva <= 0:
                continue
            off = d.rva if d.type == lief.PE.DataDirectory.TYPES.CERTIFICATE_TABLE else pe.rva_to_offset(d.rva)
            if off >= size:
                continue
            v.append(d)
        return v

    # ---------- PE Logic ----------
    def test_is_code_a(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert StructureParser._is_code(section)

    def test_is_code_b(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_READ)
        assert not StructureParser._is_code(section)

    def test_is_code_c(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert not StructureParser._is_code(section)

    def test_is_data_a(self) -> None:
        section = self._create_section_with_characteristics(".data", lief.PE.Section.CHARACTERISTICS.MEM_READ)
        assert StructureParser._is_data(section)

    def test_is_data_b(self) -> None:
        section = self._create_section_with_characteristics(".data", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert StructureParser._is_data(section)

    def test_is_data_c(self) -> None:
        section = self._create_section_with_characteristics(".data", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert not StructureParser._is_data(section)

    def test_is_data_d(self) -> None:
        section = self._create_section_with_characteristics(".data", 0)
        assert not StructureParser._is_data(section)

    def test_is_rdata_a(self) -> None:
        section = self._create_section_with_characteristics(".rdata", lief.PE.Section.CHARACTERISTICS.MEM_READ)
        assert StructureParser._is_rdata(section)

    def test_is_rdata_b(self) -> None:
        section = self._create_section_with_characteristics(".rdata", lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert not StructureParser._is_rdata(section)

    def test_is_rdata_c(self) -> None:
        section = self._create_section_with_characteristics(".rdata", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert not StructureParser._is_rdata(section)

    def test_is_wdata_a(self) -> None:
        section = self._create_section_with_characteristics(".wdata", lief.PE.Section.CHARACTERISTICS.MEM_READ)
        assert not StructureParser._is_wdata(section)

    def test_is_wdata_b(self) -> None:
        section = self._create_section_with_characteristics(".wdata", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert StructureParser._is_wdata(section)

    def test_is_wdata_c(self) -> None:
        section = self._create_section_with_characteristics(".wdata", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert not StructureParser._is_wdata(section)

    def test_is_rcode_a(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert StructureParser._is_rcode(section)

    def test_is_rcode_b(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert not StructureParser._is_rcode(section)

    def test_is_rcode_c(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_WRITE | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert not StructureParser._is_rcode(section)

    def test_is_wcode_a(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_WRITE | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert StructureParser._is_wcode(section)

    def test_is_wcode_b(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
        assert not StructureParser._is_wcode(section)

    def test_is_wcode_c(self) -> None:
        section = self._create_section_with_characteristics(".text", lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
        assert not StructureParser._is_wcode(section)

    # ---------- ANY ----------
    @pytest.mark.parametrize("file", FILES)
    def test_get_any_real(self, file: Path) -> None:
        size = os.path.getsize(file)
        parser = StructureParser(file)
        ranges = parser.get_any()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0] == (0, size)

    def test_get_any_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_any()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0] == (0, size)

    # ---------- COARSE ----------
    @pytest.mark.parametrize("file", FILES)
    def test_get_headers_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_headers()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == 1
        assert ranges[0] == (0, parser.pe.sizeof_headers)

    def test_get_headers_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_headers()    
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0] == (0, pe.sizeof_headers)

    @pytest.mark.parametrize("file", FILES)
    def test_get_sections_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_sections()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == len(self._valid_sections(parser.pe, parser.size))

    def test_get_sections_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_sections()
        self._check_ranges(ranges, size)
        assert len(ranges) == 2
        assert ranges[0][0] == pe.sizeof_headers
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".text").content)
        assert ranges[1][0] == ranges[0][1]
        assert ranges[1][1] == size - len(pe.overlay)

    @pytest.mark.parametrize("file", FILES)
    def test_get_overlay_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_overlay()
        self._check_ranges(ranges, os.path.getsize(file))
        if parser.pe.overlay_offset > 0:
            assert len(ranges) == 1
            assert ranges[0][0] == parser.pe.overlay_offset
            assert ranges[0][1] == ranges[0][0] + len(parser.pe.overlay)
        else:
            assert len(ranges) == 0

    def test_get_overlay_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_overlay()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == size - len(pe.overlay)
        assert ranges[0][1] == size
        assert byte[ranges[0][0]:ranges[0][1]] == b"OVERLAY"

    # ---------- MIDDLE ----------
    @pytest.mark.parametrize("file", FILES)
    def test_get_code_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_code()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len([s for s in parser.pe.sections if s.sizeof_raw_data > 0])

    def test_get_code_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_section_text(".code").add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_code()
        self._check_ranges(ranges, size)
        assert len(ranges) == 2
        assert ranges[0][0] == pe.sizeof_headers
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".text").content)
        assert ranges[1][0] == ranges[0][1] + len(pe.get_section(".data").content)
        assert ranges[1][1] == ranges[1][0] + len(pe.get_section(".code").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_data_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_data()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_data_synth(self) -> None:
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_section_data(".data2").add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_data()
        self._check_ranges(ranges, size)
        assert len(ranges) == 2
        assert ranges[0][0] == pe.sizeof_headers + len(pe.get_section(".text").content)
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".data").content)
        assert ranges[1][0] == ranges[0][1]
        assert ranges[1][1] == ranges[1][0] + len(pe.get_section(".data2").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_directory_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_directory()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == len(self._valid_directories(parser.pe, parser.size))

    def test_get_directory_synth(self) -> None:  # TODO: enhance
        pe, byte = BinaryCreator().add_section_text().add_section_data().add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_data()
        self._check_ranges(ranges, size)

    @pytest.mark.parametrize("file", FILES)
    def test_get_othersec_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_othersec()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_othersec_synth(self) -> None:
        section = lief.PE.Section(".unk")
        section.characteristics = 0
        section.content = bytearray(0x200)
        pe, byte = BinaryCreator().add_section_text().add_section(section).add_overlay(b"OVERLAY")()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_othersec()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == pe.sizeof_headers + len(pe.get_section(".text").content)
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".unk").content)

    # ---------- FINE ----------
    @pytest.mark.parametrize("file", FILES)
    def test_get_rcode_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_rcode()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_rcode_synth(self) -> None:
        c_wcode = lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE | lief.PE.Section.CHARACTERISTICS.MEM_WRITE
        c_rcode = lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE
        pe, byte = BinaryCreator().add_section_text(".wcode", characteristics=c_wcode).add_section_data().add_section_text(".rcode", characteristics=c_rcode)()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_rcode()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == pe.sizeof_headers + len(pe.get_section(".wcode").content) + len(pe.get_section(".data").content)
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".rcode").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_wcode_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_wcode()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_wcode_synth(self) -> None:
        c_wcode = lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE | lief.PE.Section.CHARACTERISTICS.MEM_WRITE
        c_rcode = lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE
        pe, byte = BinaryCreator().add_section_text(".rcode", characteristics=c_rcode).add_section_data().add_section_text(".wcode", characteristics=c_wcode)()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_wcode()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == pe.sizeof_headers + len(pe.get_section(".rcode").content) + len(pe.get_section(".data").content)
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".wcode").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_rdata_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_rdata()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_rdata_synth(self) -> None:
        c_rdata = lief.PE.Section.CHARACTERISTICS.MEM_READ
        c_wdata = lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE
        s_rdata = lief.PE.Section(list(bytearray(0x200)), ".rdata", c_rdata)
        s_wdata = lief.PE.Section(list(bytearray(0x200)), ".wdata", c_wdata)
        pe, byte = BinaryCreator().add_section(s_rdata).add_section(s_wdata).add_section_text()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_rdata()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == pe.sizeof_headers
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".rdata").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_wdata_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_wdata()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_sections(parser.pe, parser.size))

    def test_get_wdata_synth(self) -> None:
        c_rdata = lief.PE.Section.CHARACTERISTICS.MEM_READ
        c_wdata = lief.PE.Section.CHARACTERISTICS.MEM_READ | lief.PE.Section.CHARACTERISTICS.MEM_WRITE
        s_rdata = lief.PE.Section(list(bytearray(0x200)), ".wdata", c_wdata)
        s_wdata = lief.PE.Section(list(bytearray(0x200)), ".rdata", c_rdata)
        pe, byte = BinaryCreator().add_section(s_rdata).add_section(s_wdata).add_section_text()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_wdata()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        assert ranges[0][0] == pe.sizeof_headers
        assert ranges[0][1] == ranges[0][0] + len(pe.get_section(".wdata").content)

    @pytest.mark.parametrize("file", FILES)
    def test_get_dos_header_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_dos_header()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    def test_get_dos_header_synth(self) -> None:
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_dos_header()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    @pytest.mark.parametrize("file", FILES)
    def test_get_coff_header_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_coff_header()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    def test_get_coff_header_synth(self) -> None:
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_coff_header()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    @pytest.mark.parametrize("file", FILES)
    def test_get_optional_header_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_optional_header()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    def test_get_optional_header_synth(self) -> None:
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_optional_header()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    @pytest.mark.parametrize("file", FILES)
    def test_get_section_table_real(self, file: Path) -> None:
        parser = StructureParser(file)
        ranges = parser.get_section_table()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    def test_get_section_table_synth(self) -> None:
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_section_table()
        self._check_ranges(ranges, size)
        assert len(ranges) == 0
        # No point in redoing the calculation here to check

        pe, byte = BinaryCreator().add_section_text()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_section_table()
        self._check_ranges(ranges, size)
        assert len(ranges) == 1
        # No point in redoing the calculation here to check

    @pytest.mark.parametrize("file", FILES)
    def test_get_idata_real(self, file: Path) -> None:  # TODO: enhance
        parser = StructureParser(file)
        ranges = parser.get_idata()
        self._check_ranges(ranges, os.path.getsize(file))
        # assert len(ranges) == 1

    def test_get_idata_synth(self) -> None:  # TODO: enhance
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_idata()
        self._check_ranges(ranges, size)
        # assert len(ranges) == 1

    @pytest.mark.parametrize("file", FILES)
    def test_get_edata_real(self, file: Path) -> None:  # TODO: enhance
        parser = StructureParser(file)
        ranges = parser.get_edata()
        self._check_ranges(ranges, os.path.getsize(file))
        # assert len(ranges) == 1

    def test_get_edata_synth(self) -> None:
        pe, byte = BinaryCreator()()  # TODO: enhance
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_edata()
        self._check_ranges(ranges, size)
        # assert len(ranges) == 1

    @pytest.mark.parametrize("file", FILES)
    def test_get_reloc_real(self, file: Path) -> None:  # TODO: enhance
        parser = StructureParser(file)
        ranges = parser.get_reloc()
        self._check_ranges(ranges, os.path.getsize(file))
        # assert len(ranges) == 1

    def test_get_reloc_synth(self) -> None:  # TODO: enhance
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_reloc()
        self._check_ranges(ranges, size)
        # assert len(ranges) == 1

    @pytest.mark.parametrize("file", FILES)
    def test_get_clr_real(self, file: Path) -> None:  # TODO: enhance
        parser = StructureParser(file)
        ranges = parser.get_clr()
        self._check_ranges(ranges, os.path.getsize(file))
        # assert len(ranges) == 1

    def test_get_clr_synth(self) -> None:  # TODO: enhance
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_clr()
        self._check_ranges(ranges, size)
        # assert len(ranges) == 1

    @pytest.mark.parametrize("file", FILES)
    def test_get_otherdir_real(self, file: Path) -> None:  # TODO: enhance
        parser = StructureParser(file)
        ranges = parser.get_otherdir()
        self._check_ranges(ranges, os.path.getsize(file))
        assert len(ranges) <= len(self._valid_directories(parser.pe, parser.size))

    def test_get_otherdir_synth(self) -> None:  # TODO: enhance
        pe, byte = BinaryCreator()()
        size = len(byte)
        parser = StructureParser(pe, size)
        ranges = parser.get_otherdir()
        self._check_ranges(ranges, size)
        assert len(ranges) <= len(self._valid_directories(parser.pe, parser.size))
