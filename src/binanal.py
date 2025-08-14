"""
Binary analysis.
"""

from __future__ import annotations
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import enum
import hashlib
import os
from pathlib import Path
import struct
from typing import Callable
from typing import Optional
from typing import Iterable
from typing import Self
import warnings

import lief
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy
import torch
from torch import BoolTensor
from torch import IntTensor
from torch import FloatTensor
from torch import DoubleTensor


StrPath = str | os.PathLike[str]
LiefParse = StrPath | bytes
Range = tuple[int, int]


# Machine types, from most to least common for PE files.
MACHINES = [
    lief.PE.Header.MACHINE_TYPES.I386,
    lief.PE.Header.MACHINE_TYPES.AMD64,
    lief.PE.Header.MACHINE_TYPES.ARM64,
]
for v in lief.PE.Header.MACHINE_TYPES.__dict__.values():
    if type(v).__name__ == "MACHINE_TYPES" and v not in MACHINES:
        MACHINES.append(v)
MACHINES = tuple(MACHINES)

# Subsystem types, from most to least common for PE files.
SUBSYSTEMS = [
    lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_GUI,
    lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_CUI,
    lief.PE.OptionalHeader.SUBSYSTEM.NATIVE,
]
for v in lief.PE.OptionalHeader.SUBSYSTEM.__dict__.values():
    if type(v).__name__ == "SUBSYSTEM" and v not in SUBSYSTEMS:
        SUBSYSTEMS.append(v)
SUBSYSTEMS = tuple(SUBSYSTEMS)


def get_ranges_numpy(x: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Detects the ranges of consecutive True values in a boolean numpy array.
    """
    if x.dtype != np.bool_ or x.ndim != 1:  # type: ignore[comparison-overlap]
        raise TypeError(f"Expected a 1D boolean numpy array, got {type(x)} with shape {x.shape}.")

    padded = np.pad(x, (1, 1), mode='constant', constant_values=False)
    diff = np.diff(padded.astype(int))
    lo = np.where(diff == 1)[0]
    hi = np.where(diff == -1)[0]
    return lo, hi


def is_section_executable(section: lief.PE.Section) -> bool:
    for c in section.characteristics_lists:
        if "MEM_EXECUTE" in c:
            return True
        if "CNT_CODE" in c:
            return True
    return False


def is_dotnet(raw: str | Path | bytes) -> bool:
    pe: lief.PE.Binary = lief.parse(raw)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    dd: lief.PE.DataDirectory = pe.data_directory(lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER)
    if not isinstance(dd, lief.PE.DataDirectory):
        raise RuntimeError(f"Expected lief.PE.DataDirectory, got {type(pe)}")
    return bool(dd.rva != 0) and bool(dd.size != 0)


def get_machine_and_subsystem(data: str | Path | bytes) -> tuple[lief.PE.Header.MACHINE_TYPES, lief.PE.OptionalHeader.SUBSYSTEM]:
    pe = lief.parse(data)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    return pe.header.machine, pe.optional_header.subsystem


def patch_binary(
    src: str | Path | bytes,
    machine: Optional[lief.PE.Header.MACHINE_TYPES] = None,
    subsystem: Optional[lief.PE.OptionalHeader.SUBSYSTEM] = None,
) -> bytes:
    if isinstance(src, (str, Path)):
        data = bytearray(Path(src).read_bytes())
    elif isinstance(src, bytes):
        data = bytearray(src)
    else:
        raise TypeError(f"Unsupported type for src: {type(src)}. Expected str, Path, or bytes.")

    # Locate headers.
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    pe_sig_off   = e_lfanew
    filehdr_off  = pe_sig_off + 4          # after "PE\0\0"
    opthdr_off   = filehdr_off + 20        # IMAGE_FILE_HEADER is 20 bytes

    # Patch Machine (IMAGE_FILE_HEADER.Machine).
    if machine is not None:
        if hasattr(machine, "value"):  # LIEF enum
            machine = machine.value
        struct.pack_into("<H", data, filehdr_off + 0, machine)

    # Patch Subsystem (IMAGE_OPTIONAL_HEADER.Subsystem).
    subsys_rel = 68
    if subsystem is not None:
        if hasattr(subsystem, "value"):
            subsystem = subsystem.value
        struct.pack_into("<H", data, opthdr_off + subsys_rel, subsystem)

    return bytes(data)


def rearm_disarmed_binary(src: str | Path | bytes, sha: str, check: bool = True, verbose: bool = False) -> bytes:
    """
    Rearm a Sorel binary by patching its machine and subsystem until the target SHA-256 matches.
    """
    if isinstance(src, (str, Path)):
        if not Path(src).exists():
            raise FileNotFoundError(f"Source file {src} does not exist.")
        data = Path(src).read_bytes()
    elif isinstance(src, bytes):
        data = src
    else:
        raise TypeError(f"Unsupported type for src: {type(src)}. Expected str, Path, or bytes.")

    if verbose:
        machine, subsystem = get_machine_and_subsystem(data)
        print(f"Target SHA-256: {sha}")
        print(f"Current Machine: {machine}")
        print(f"Current Subsystem: {subsystem}")

    for machine in MACHINES:
        for subsystem in SUBSYSTEMS:
            patched = patch_binary(data, machine=machine, subsystem=subsystem)
            s = hashlib.sha256(patched).hexdigest()
            if verbose:
                print(f"Ran Machine: {machine}, Subsystem: {subsystem} -> SHA-256: {s}")
            if s == sha:
                if check:
                    machine_, subsystem_ = get_machine_and_subsystem(patched)
                    if machine_ != machine or subsystem_ != subsystem:
                        raise ValueError(f"Patched binary does not match expected machine and subsystem: {machine_}, {subsystem_}")
                    equal = np.equal(np.frombuffer(data, dtype=np.uint8), np.frombuffer(patched, dtype=np.uint8))  # type: ignore[no-untyped-call]
                    if len(equal) - np.sum(equal) > 3:
                        raise ValueError(f"Patched binary differs from original by more than 3 bytes: {len(equal) - np.sum(equal)} differences.")
                return patched

    raise RuntimeError(f"Could not find matching machine and subsystem for SHA-256: {sha}")


class BinaryCreator:

    def __init__(
        self,
        type_: lief.PE.PE_TYPE = lief.PE.PE_TYPE.PE32,
        machine: lief.PE.Header.MACHINE_TYPES = lief.PE.Header.MACHINE_TYPES.I386,
        subsystem: lief.PE.OptionalHeader.SUBSYSTEM = lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_CUI,
    ) -> None:
        self.pe = lief.PE.Binary(type_)
        self.pe.header.machine = machine
        self.pe.header.sizeof_optional_header = 0xE0 if type_ == lief.PE.PE_TYPE.PE32 else 0xF0
        self.pe.optional_header.subsystem = subsystem
        self.pe.optional_header.major_operating_system_version = 4
        self.pe.optional_header.minor_operating_system_version = 0
        self.pe.optional_header.major_subsystem_version = 4
        self.pe.optional_header.minor_subsystem_version = 0
        self.pe.optional_header.file_alignment = 0x200
        self.pe.optional_header.section_alignment = 0x1000
        self.pe.optional_header.addressof_entrypoint = 0x1000
        self.pe.optional_header.numberof_rva_and_size = 16

        self.overlay = b""

    def __call__(self) -> tuple[lief.PE.Binary, bytes]:
        builder = lief.PE.Builder(self.pe)
        builder.build()
        pe_bytes = bytes(builder.get_build())
        pe_bytes += self.overlay
        pe: lief.PE.Binary = lief.parse(pe_bytes)
        if pe is None:
            raise RuntimeError(f"lief.parse({type(pe_bytes)}) returned None")
        if pe.overlay_offset == 0:
            warnings.warn("PE overlay offset is 0, which may indicate that the overlay was not added correctly.")
        elif bytes(pe.overlay) != self.overlay:
            raise RuntimeError("Overlay mismatch after building the PE binary.")
        return pe, pe_bytes

    def add_overlay(self, overlay: bytes) -> Self:
        self.overlay += overlay
        return self

    def add_section(self, section: lief.PE.Section, type_: lief.PE.SECTION_TYPES = lief.PE.SECTION_TYPES.UNKNOWN) -> Self:
        self.pe.add_section(section, type_)
        return self

    def add_section_text(self, name: str = ".text", content: Optional[Sequence[int]] = None, characteristics: Optional[int] = None) -> Self:
        section = lief.PE.Section(name)
        if content is None:
            content = bytearray(0x200)
            content[0] = 0xC3  # RET
            content = list(content)
        section.content = content
        if characteristics is None:
            characteristics = (
                int(lief.PE.Section.CHARACTERISTICS.CNT_CODE) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_READ) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
            )
        section.characteristics = characteristics
        self.add_section(section, lief.PE.SECTION_TYPES.TEXT)
        return self

    def add_section_data(self, name: str = ".data", content: Optional[Sequence[int]] = None, characteristics: Optional[int] = None) -> Self:
        if characteristics is not None:
            warnings.warn("Characteristics may be ignored when using the lief.PE.SECTION_TYPES.DATA type.")
        section = lief.PE.Section(name)
        if content is None:
            content = bytearray(0x200)
            content[0] = 0x00
            content = list(content)
        section.content = content
        if characteristics is None:
            characteristics = (
                int(lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_READ) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
            )
        section.characteristics = characteristics
        self.add_section(section, lief.PE.SECTION_TYPES.DATA)
        return self


@dataclass(frozen=True, slots=True)
class SemanticGuides:
    parse: Optional[BoolTensor] = None
    entropy: Optional[DoubleTensor] = None
    characteristics: Optional[BoolTensor] = None

    def __post_init__(self) -> None:
        lengths = [len(x) for x in (self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"SemanticSample buffers have different lengths: {lengths}")


class SemanticGuider:
    """
    Semantic guides to acompany a byte stream.
    """

    CHARACTERISTICS = [
        lief.PE.Section.CHARACTERISTICS.CNT_CODE,
        lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA,
        lief.PE.Section.CHARACTERISTICS.CNT_UNINITIALIZED_DATA,
        lief.PE.Section.CHARACTERISTICS.GPREL,
        lief.PE.Section.CHARACTERISTICS.LNK_NRELOC_OVFL,
        lief.PE.Section.CHARACTERISTICS.MEM_DISCARDABLE,
        lief.PE.Section.CHARACTERISTICS.MEM_NOT_CACHED,
        lief.PE.Section.CHARACTERISTICS.MEM_NOT_PAGED,
        lief.PE.Section.CHARACTERISTICS.MEM_SHARED,
        lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE,
        lief.PE.Section.CHARACTERISTICS.MEM_READ,
        lief.PE.Section.CHARACTERISTICS.MEM_WRITE,
    ]

    def __init__(self, do_parse: bool = False, do_entropy: bool = False, do_characteristics: bool = False, window: int = 256, simple: bool = False) -> None:
        self.do_parse = do_parse
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.window = window
        self.simple = simple

    def __call__(self, data: LiefParse) -> SemanticGuides:
        parse = None
        if self.do_parse:
            try:
                parse = self.create_parse_guide(data, self.simple)
            except Exception as err:
                print(f"Error creating parse guide: {err}")

        entropy = None
        if self.do_entropy:
            try:
                entropy = self.create_entropy_guide(data, self.window)
            except Exception as err:
                print(f"Error creating entropy guide: {err}")

        characteristics = None
        if self.do_characteristics:
            try:
                characteristics = self.create_characteristics_guide(data)
            except Exception as err:
                print(f"Error creating permission guide: {err}")

        return SemanticGuides(parse, entropy, characteristics)

    @staticmethod
    def create_parse_guide(data: LiefParse, simple: bool) -> BoolTensor:
        return ParserGuider(data)(simple=simple)

    @staticmethod
    def create_entropy_guide(data: LiefParse, w: int, e: float = 1e-8) -> FloatTensor:
        b = Path(data).read_bytes() if isinstance(data, (str, Path)) else data
        x = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
        x = entropy(sliding_window_view(x, w * 2 + 1) + e, axis=1, nan_policy="raise")
        if np.any(np.isnan(x)):
            raise RuntimeError("Entropy calculation resulted in NaN values.")
        x = np.concatenate((np.full(w, np.nan), x, np.full(w, np.nan)))
        x = torch.from_numpy(x)
        return x

    @staticmethod
    def create_characteristics_guide(data: LiefParse) -> BoolTensor:
        size = os.path.getsize(data) if isinstance(data, (str, os.PathLike)) else len(data)
        x = torch.full((size, len(SemanticGuider.CHARACTERISTICS)), False, dtype=torch.bool)
        pe = lief.parse(data)
        if pe is None:
            raise RuntimeError(f"lief.parse({type(data)}) return None")
        for section in pe.sections:
            offset = section.offset
            size = section.size
            for i, c in enumerate(SemanticGuider.CHARACTERISTICS):
                x[offset:offset + size, i] = True if section.has_characteristic(c) else False
        return x


class ParserGuider:

    PARSEERRORS = [
        lief.lief_errors.asn1_bad_tag,
        lief.lief_errors.build_error,
        lief.lief_errors.conversion_error,
        lief.lief_errors.corrupted,
        lief.lief_errors.data_too_large,
        lief.lief_errors.file_error,
        lief.lief_errors.file_format_error,
        lief.lief_errors.not_found,
        lief.lief_errors.not_implemented,
        lief.lief_errors.not_supported,
        lief.lief_errors.parsing_error,
        lief.lief_errors.read_error,
        lief.lief_errors.read_out_of_bound,
        lief.lief_errors.require_extended_version
    ]
    PARSEERRORS = [Exception]  # FIXME: remove.

    def __init__(self, data: LiefParse) -> None:
        warnings.warn("ParserGuider not yet fully operational. All errors marked in same channel.")
        self.data = data
        self.pe: Optional[lief.PE.Binary] = None
        self.guide = torch.zeros((len(self), len(ParserGuider.PARSEERRORS)), dtype=torch.bool)

    def __call__(self, simple: bool) -> BoolTensor:
        try:
            self.pe = lief.parse(self.data)
        except Exception as e:
            self._mark_guide(0, len(self), e)
            return self.guide

        if self.pe is None:
            raise RuntimeError(f"lief.parse({type(self.data)}) return None")

        if simple:
            self.build_simple_guide()
        else:
            self.build_complex_guide()
        return self.guide

    def __len__(self) -> int:
        return os.path.getsize(self.data) if isinstance(self.data, (str, os.PathLike)) else len(self.data)

    def build_simple_guide(self) -> None:
        if self.pe is None:
            raise RuntimeError("ParserGuider has not been called yet.")

        probes: list[tuple[str, tuple[int, int]]] = [
            ("header", (0, min(0x40, len(self)))),
            ("optional_header", (0, min(0x100, len(self)))),
            ("sections", (0x200, len(self))),
            ("data_directories", (0, min(0x200, len(self)))),
            ("imports", (0, len(self))),
            ("exports", (0, len(self))),
            ("resources", (0, len(self))),
        ]
        for attr, (lo, hi) in probes:
            try:
                _ = getattr(self.pe, attr)
                if attr == "sections":
                    for s in self.pe.sections:
                        _ = (s.name, s.size)
                elif attr == "data_directories":
                    for dd in self.pe.data_directories[:8]:
                        _ = (dd.type, dd.size)
                elif attr == "imports":
                    for im in self.pe.imports[:5]:
                        _ = im.name
                elif attr == "exports":
                    _ = self.pe.has_exports
                elif attr == "resources":
                    _ = self.pe.has_resources
            except Exception as e:
                self._mark_guide(lo, hi, e)

    def build_complex_guide(self) -> None:
        if self.pe is None:
            raise RuntimeError("ParserGuider has not been called yet.")

        # Check the header
        lo, hi = self._compute_coff_header_region_boundaries(self.pe, len(self))
        try:
            getattr(self.pe, "header")
        except Exception as e:
            self._mark_guide(lo, hi, e)

        # Check the optional header
        lo, hi = self._compute_optional_header_boundaries(self.pe, len(self))
        try:
            getattr(self.pe, "optional_header")
        except Exception as e:
            self._mark_guide(lo, hi, e)

        # Check the section table
        lo, hi = self._compute_section_table_boundaries(self.pe, len(self))
        try:
            getattr(self.pe, "sections")
        except Exception as e:
            self._mark_guide(lo, hi, e)
        else:
            # Check the sections
            if self.pe.sections is not None:
                for s in self.pe.sections:
                    lo = s.pointerto_raw_data
                    hi = min(lo + s.size, len(self))
                    try:
                        # Touch properties that often trigger lazy reads/validations
                        _ = (s.name, s.size, s.virtual_address, s.content)
                    except Exception as e:
                        self._mark_guide(lo, hi, e)

        # Check the data directories
        lo, hi = self._compute_optional_header_boundaries(self.pe, len(self))
        try:
            getattr(self.pe, "data_directories")
        except Exception as e:
            self._mark_guide(lo, hi, e)
        else:
            # Check the data directory information
            if self.pe.data_directories is not None:
                interesting = {
                    lief.PE.DataDirectory.TYPES.IMPORT_TABLE: "imports",
                    lief.PE.DataDirectory.TYPES.EXPORT_TABLE: "exports",
                    lief.PE.DataDirectory.TYPES.RESOURCE_TABLE: "resources",
                }
                for k, attr_name in interesting.items():
                    dir_entry = self.pe.data_directory(k)
                    rva = getattr(dir_entry, "rva", 0)
                    size = getattr(dir_entry, "size", 0)
                    try:
                        _ = getattr(self.pe, attr_name)
                        # Poke a few fields to trigger lazy parsing
                        if attr_name == "imports":
                            next(iter(self.pe.imports))
                        elif attr_name == "exports":
                            _ = self.pe.has_exports
                        elif attr_name == "resources":
                            _ = self.pe.has_resources
                    except Exception as e_obj:
                        if not bool(rva) or not bool(size) or not bool(self.pe.rva_to_offset(rva)):
                            lo, hi = self._compute_optional_header_boundaries(self.pe, len(self))
                        else:
                            off = self.pe.rva_to_offset(rva)
                            if off is None or off < 0:
                                return None
                            lo = self.pe.rva_to_offset(rva)
                            hi = min(lo + size, len(self))
                        self._mark_guide(lo, hi, e_obj)

    @staticmethod
    def get_e_lfanew(pe: lief.PE.Binary, size: int) -> int:
        dos = pe.dos_header
        if dos is None:
            return -1
        e_lfanew: Optional[int] = dos.addressof_new_exeheader
        if e_lfanew is None:
            return -1
        if e_lfanew < 0:
            return -1
        if e_lfanew >= size - 4:
            return -1
        return e_lfanew

    @staticmethod
    def _compute_coff_header_region_boundaries(pe: lief.PE.Binary, size: int) -> tuple[int, int]:
        if (e_lfanew := ParserGuider.get_e_lfanew(pe, size)) < 0:
            return 0, min(0x40, size)
        lo = e_lfanew + 4
        hi = min(lo + 20, size)
        return lo, hi

    @staticmethod
    def _compute_optional_header_boundaries(pe: lief.PE.Binary, size: int) -> tuple[int, int]:
        if ParserGuider.get_e_lfanew(pe, size) < 0:
            return 0, min(0x100, size)
        _, lo = ParserGuider._compute_coff_header_region_boundaries(pe, size)
        size_opt = pe.header.sizeof_optional_header if pe.header is not None else 0
        hi = min(size_opt, size)
        return lo, hi

    @staticmethod
    def _compute_section_table_boundaries(pe: lief.PE.Binary, size: int) -> tuple[int, int]:
        if ParserGuider.get_e_lfanew(pe, size) < 0:
            return min(size, 0x100), min(size, 0x200)
        _,  lo  = ParserGuider._compute_optional_header_boundaries(pe, size)
        nsects = pe.header.numberof_sections if pe.header is not None else 0
        hi = min(lo + 40 * nsects, size)
        return lo, hi

    def _mark_guide(self, start: int, end: int, error: Exception) -> None:
        self.guide[start:end, self._lief_exc_idx(error)] = True

    def _lief_exc_idx(self, e: Exception) -> int:
        for i, t in enumerate(ParserGuider.PARSEERRORS):
            if isinstance(e, t):
                return i
        raise TypeError(f"Unknown LIEF exception type: {type(e)}. Expected one of {ParserGuider.PARSEERRORS}.")


class HierarchicalLevel(enum.Enum):
    NONE = "none"
    COARSE = "coarse"
    MIDDLE = "middle"
    FINE = "fine"


class HierarchicalStructure(enum.Enum):
    ...


class HierarchicalStructureNone(HierarchicalStructure):
    ANY = "any"


class HierarchicalStructureCoarse(HierarchicalStructure):
    HEADERS = "headers"  # Headers      (ANY)
    SECTION = "sections" # Sections     (ANY)
    OVERLAY = "overlay"  # Overlay data (ANY)
    OTHER   = "other"    # All other bytes  (ANY)


class HierarchicalStructureMiddle(HierarchicalStructure):
    HEADERS = "headers"     # Headers               (ANY)
    OVERLAY = "overlay"     # Overlay data          (ANY)
    OTHER = "other"         # All other bytes       (ANY)
    CODE = "code"           # Code sections         (SECTION)
    DATA = "data"           # Data sections         (SECTION)
    DIRECTORY = "directory" # Resouce directories   (SECTION)
    OTHERSEC = "othersec"   # No clean split        (SECTION)


class HierarchicalStructureFine(HierarchicalStructure):
    # TODO: Should we consider using the lief.PE.SECTION_TYPES?
    OVERLAY = "overlay"          # Overlay data     (ANY)
    OTHER = "other"              # All other bytes  (ANY)
    DOS_HEADER  = "dos_header"   # DOS header       (HEADER)
    COFF_HEADER = "coff_header"  # COFF header      (HEADER)
    OPTN_HEADER = "optn_header"  # Optional header  (HEADER)
    SECTN_TABLE = "sectn_table"  # Section table    (HEADER)
    OTHERSEC = "othersec"        # No clean split   (SECTION)
    RDATA = "rdata"              # Read-only data   (DATA)
    WDATA = "wdata"              # Writeable data   (DATA)
    RCODE = "rcode"              # Read-only code   (CODE)
    WCODE = "wcode"              # Writeable code   (CODE)
    IDATA = "idata"              # Import data      (DIRECTORY)
    EDATA = "edata"              # Export data      (DIRECTORY)
    RELOC = "reloc"              # Relocation data  (DIRECTORY)
    CLR   = "clr"                # CLR/.NET data    (DIRECTORY)
    OTHERDIR = "otherdir"        # Other directory  (DIRECTORY)


class StructureParser:

    _SCN_MEM_EXECUTE = 0x20000000
    _SCN_MEM_READ    = 0x40000000
    _SCN_MEM_WRITE   = 0x80000000

    # _SCN_CNT_CODE    = 0x00000020
    # _SCN_CNT_IN_DATA = 0x00000040
    # _SCN_CNT_UN_DATA = 0x00000080

    def __init__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> None:
        self.pe: lief.PE.Binary = data if isinstance(data, lief.PE.Binary) else lief.parse(data)
        self.size: int = size if size is not None else (len(data) if isinstance(data, bytes) else os.path.getsize(data))
        if self.pe is None:
            raise RuntimeError(f"lief.parse({type(data)}) return None")
        if self.size is None:
            raise ValueError("Size must be provided if data is a lief.PE.Binary object.")

        self.functions = {
            HierarchicalStructureNone.ANY: self.get_any,
            HierarchicalStructureCoarse.HEADERS: self.get_headers,
            HierarchicalStructureCoarse.SECTION: self.get_sections,
            HierarchicalStructureCoarse.OVERLAY: self.get_overlay,
            HierarchicalStructureCoarse.OTHER: self.get_other,
            HierarchicalStructureMiddle.CODE: self.get_code,
            HierarchicalStructureMiddle.DATA: self.get_data,
            HierarchicalStructureMiddle.DIRECTORY: self.get_directory,
            HierarchicalStructureMiddle.OTHERSEC: self.get_othersec,
            HierarchicalStructureFine.DOS_HEADER: self.get_dos_header,
            HierarchicalStructureFine.COFF_HEADER: self.get_coff_header,
            HierarchicalStructureFine.OPTN_HEADER: self.get_optional_header,
            HierarchicalStructureFine.SECTN_TABLE: self.get_section_table,
            HierarchicalStructureFine.RDATA: self.get_rdata,
            HierarchicalStructureFine.WDATA: self.get_wdata,
            HierarchicalStructureFine.RCODE: self.get_rcode,
            HierarchicalStructureFine.WCODE: self.get_wcode,
            HierarchicalStructureFine.IDATA: self.get_idata,
            HierarchicalStructureFine.EDATA: self.get_edata,
            HierarchicalStructureFine.RELOC: self.get_reloc,
            HierarchicalStructureFine.CLR: self.get_clr,
            HierarchicalStructureFine.OTHERDIR: self.get_otherdir,
        }

    def __call__(self, structure: HierarchicalStructure) -> list[Range]:
        if self.pe is None:
            return []
        return self._norm(self.functions[structure]())

    # ---------- Range Helpers ----------
    def _norm(self, ranges: Iterable[Range]) -> list[Range]:
        ranges = [r for r in ranges if r is not None]
        ranges = [self._clip(*r) for r in ranges]
        ranges = [r for r in ranges if r is not None]
        return ranges

    def _clip(self, start: int, end: int) -> Optional[Range]:
        """Clip range to be within the file size. Return None if invalid.
        """
        if start is None or end is None:  # TODO: remove?
            return None
        start = max(0, start)  # TODO: add warnings for stuff out of bounds.
        end   = min(self.size, end)
        if end <= start:
            return None
        return (start, end)

    # ---------- PE Helpers ----------
    def _sec_raw_range(self, s: lief.PE.Section) -> Optional[Range]:
        """Get the raw data range of a section if it has raw data.
        """
        if s.sizeof_raw_data == 0:
            return None
        return (int(s.pointerto_raw_data), int(s.pointerto_raw_data + s.sizeof_raw_data))

    def _select_sections(self, function: Callable[[lief.PE.Section], bool]) -> list[Range]:
        out = []
        for s in self.pe.sections:
            if function(s):
                r = self._sec_raw_range(s)
                if r:
                    out.append(r)
        return out

    def _dirs(self) -> list[lief.PE.DataDirectory]:
        try:
            return list(self.pe.data_directories)
        except Exception:
            print("An error occurred while accessing data directories.")
            return []

    def _dir_range(self, d: lief.PE.DataDirectory) -> Optional[Range]:
        try:
            # SECURITY (certificate table) is special: VirtualAddress is a FILE OFFSET (not an RVA)
            if d.type == lief.PE.DataDirectory.TYPES.CERTIFICATE_TABLE:
                off = int(d.rva)
                return (off, off + int(d.size))
            if d.size == 0:
                return None
            if d.rva == 0:  # TODO: is this really an invalid RVA?
                return None
            off = self.pe.rva_to_offset(int(d.rva))
            if off is None or off <= 0:
                return None
            return (int(off), int(off + d.size))
        except Exception:
            print(f"An error occurred while computing range for data directory {d.type}.")
            return None

    def _dir_type_range(self, types: set[lief.PE.DataDirectory.TYPES]) -> list[Range]:
        out = []
        for d in self._dirs():
            if d.type in types:
                r = self._dir_range(d)
                if r:
                    out.append(r)
        return out

    # ---------- PE Logic ----------
    @staticmethod
    def _is_code(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_executable = bool(c & StructureParser._SCN_MEM_EXECUTE)
        is_code_section = False # bool(c & StructureParser._SCN_CNT_CODE)
        return is_executable or is_code_section

    @staticmethod
    def _is_rcode(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_code(s) and not is_writeable

    @staticmethod
    def _is_wcode(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_code(s) and is_writeable

    @staticmethod
    def _is_data(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_readable = bool(c & StructureParser._SCN_MEM_READ)
        is_executable = bool(c & StructureParser._SCN_MEM_EXECUTE)
        is_initialized = False # bool(c & StructureParser._SCN_CNT_IN_DATA)
        is_uninitialized = False # bool(c & StructureParser._SCN_CNT_UN_DATA)
        return is_readable and (not is_executable or is_initialized or is_uninitialized)

    @staticmethod
    def _is_rdata(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        print(f"StructureParser::_is_rdata {StructureParser._is_data(s) = } {is_writeable = }")
        return StructureParser._is_data(s) and not is_writeable

    @staticmethod
    def _is_wdata(s: lief.PE.Section) -> bool:
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_data(s) and is_writeable

    # ---------- ANY ----------
    def get_any(self) -> list[Range]:
        return self._norm([(0, self.size)])

    # ---------- COARSE ----------
    def get_headers(self) -> list[Range]:
        return self._norm([(0, self.pe.sizeof_headers)])

    def get_sections(self) -> list[Range]:
        out = []
        for s in self.pe.sections:
            r = self._sec_raw_range(s)
            if r:
                out.append(r)
        return self._norm(out)

    def get_overlay(self) -> list[Range]:
        sec_end = 0
        for s in self.pe.sections:
            if s.sizeof_raw_data and s.pointerto_raw_data:
                sec_end = max(sec_end, int(s.pointerto_raw_data + s.sizeof_raw_data))
        if self.size > sec_end:
            return self._norm([(sec_end, self.size)])
        return []

    def get_other(self) -> list[Range]:
        covered = np.array([False] * self.size, dtype=bool)
        for lo, hi in self.get_headers() + self.get_sections() + self.get_overlay():
            covered[lo:hi] = True
        lo, hi = get_ranges_numpy(covered == False)
        return self._norm([(int(l), int(o)) for l, o in zip(lo, hi, strict=True)])

    # ---------- MIDDLE ----------
    def get_code(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_code))

    def get_data(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_data))

    def get_directory(self) -> list[Range]:
        warnings.warn("StructureParser::get_directory() has not been tested thoroughly.")
        out = []
        for d in self._dirs():
            r = self._dir_range(d)
            if r:
                out.append(r)
        return self._norm(out)

    def get_othersec(self) -> list[Range]:
        allsec = self.get_sections()
        for b in self.get_code() + self.get_data() + self.get_directory():
            if b in allsec:
                allsec.remove(b)
        return self._norm(allsec)

    # ---------- FINE ----------
    def get_dos_header(self) -> list[Range]:
        try:
            e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
            dos_start = 0
            dos_end   = e_lfanew + 64  # DOS header is 64 bytes long
            return self._norm([(dos_start, dos_end)])
        except Exception:  # TODO: why catch?
            warnings.warn("An error occurred while computing DOS header range.")
            return []

    def get_coff_header(self) -> list[Range]:
        try:
            e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
            coff_start = e_lfanew + 4  # skip "PE\0\0"
            coff_end   = coff_start + 20
            return self._norm([(coff_start, coff_end)])
        except Exception:  # TODO: why catch?
            warnings.warn("An error occurred while computing COFF header range.")
            return []

    def get_optional_header(self) -> list[Range]:
        try:
            e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
            coff_start = e_lfanew + 4
            coff_end   = coff_start + 20
            opt_start  = coff_end
            opt_end    = opt_start + int(self.pe.header.sizeof_optional_header)
            return self._norm([(opt_start, opt_end)])
        except Exception:  # TODO: why catch?
            warnings.warn("An error occurred while computing optional header range.")
            return []

    def get_section_table(self) -> list[Range]:
        try:
            e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
            coff_start = e_lfanew + 4
            coff_end   = coff_start + 20
            opt_start  = coff_end
            opt_end    = opt_start + int(self.pe.header.sizeof_optional_header)
            nsects     = int(self.pe.header.numberof_sections)
            sectab_start = opt_end
            sectab_end   = sectab_start + 40 * nsects
            return self._norm([(sectab_start, sectab_end)])
        except Exception:  # TODO: why catch?
            warnings.warn("An error occurred while computing section table range.")
            return []

    def get_rdata(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_rdata))

    def get_wdata(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_wdata))

    def get_rcode(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_rcode))

    def get_wcode(self) -> list[Range]:
        return self._norm(self._select_sections(self._is_wcode))

    def get_idata(self) -> list[Range]:
        warnings.warn("StructureParser::get_idata() has not been tested thoroughly.")
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.IMPORT_TABLE}))

    def get_edata(self) -> list[Range]:
        warnings.warn("StructureParser::get_edata() has not been tested thoroughly.")
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.EXPORT_TABLE}))

    def get_reloc(self) -> list[Range]:
        warnings.warn("StructureParser::get_reloc() has not been tested thoroughly.")
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.BASE_RELOCATION_TABLE}))

    def get_clr(self) -> list[Range]:
        warnings.warn("StructureParser::get_clr() has not been tested thoroughly.")
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER}))

    def get_otherdir(self) -> list[Range]:
        warnings.warn("StructureParser::get_otherdir() has not been tested thoroughly.")
        alldir = self.get_directory()
        for b in self.get_idata() + self.get_edata() + self.get_reloc() + self.get_clr():
            if b in alldir:
                alldir.remove(b)
        return self._norm(alldir)


@dataclass(frozen=True, slots=True)
class StructureMap:
    index: IntTensor
    lexicon: Mapping[int, HierarchicalStructure]

    def __post_init__(self) -> None:
        if set(torch.unique(self.index)) != set(self.lexicon.keys()):
            raise ValueError("StructureMap index does not match lexicon keys.")


class StructurePartitioner:
    """
    Partitions a binary into hierarchical structures.
    """

    def __init__(self, level: HierarchicalLevel = HierarchicalLevel.NONE) -> None:
        self.level = level

    def __call__(self, data: LiefParse) -> StructureMap:
        parser = StructureParser(data)

        if self.level == HierarchicalLevel.NONE:
            structures = list(HierarchicalStructureNone)
        if self.level == HierarchicalLevel.COARSE:
            structures = list(HierarchicalStructureCoarse)
        elif self.level == HierarchicalLevel.MIDDLE:
            structures = list(HierarchicalStructureMiddle)
        elif self.level == HierarchicalLevel.FINE:
            structures = list(HierarchicalStructureFine)
        else:
            raise TypeError(f"Unknown HierarchicalLevel: {self.level}. Expected one of {list(HierarchicalLevel)}.")

        index = torch.zeros(parser.size, dtype=torch.int32)
        lexicon = {}
        for i, s in enumerate(structures):
            bounds = parser(s)
            for l, u in bounds:
                index[l:u] = i
            lexicon[i] = s
        return StructureMap(index, lexicon)
