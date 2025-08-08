"""
Binary analysis.
"""

from dataclasses import dataclass
import enum
import hashlib
import os
from pathlib import Path
import struct
from typing import Optional
import warnings

import lief
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy
import torch
from torch import BoolTensor
from torch import IntTensor
from torch import FloatTensor
from torch import DoubleTensor


StrPath = str | os.PathLike[str]
LiefParse = StrPath | bytes


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


@dataclass(frozen=True, slots=True)
class SemanticGuides:
    parse: Optional[IntTensor] = None
    entropy: Optional[DoubleTensor] = None
    characteristics: Optional[IntTensor] = None

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
    def create_characteristics_guide(data: LiefParse) -> IntTensor:
        size = os.path.getsize(data) if isinstance(data, (str, os.PathLike)) else len(data)
        x = torch.full((size, len(SemanticGuider.CHARACTERISTICS)), -1, dtype=torch.int32)
        pe = lief.parse(data)
        if pe is None:
            raise RuntimeError(f"lief.parse({type(data)}) return None")
        for section in pe.sections:
            offset = section.offset
            size = section.size
            for i, c in enumerate(SemanticGuider.CHARACTERISTICS):
                x[offset:offset + size, i] = 1 if section.has_characteristic(c) else 0
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


@dataclass(frozen=True, slots=True)
class StructureMap:
    index: IntTensor
    lexicon: dict[int, str]

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
        size = os.path.getsize(data) if isinstance(data, (str, os.PathLike)) else len(data)

        if self.level == HierarchicalLevel.NONE:
            index = torch.zeros(size, dtype=torch.int32)
            lexicon = {0: "none"}
        elif self.level == HierarchicalLevel.COARSE:
            raise NotImplementedError("Coarse level partitioning is not implemented yet.")
        elif self.level == HierarchicalLevel.MIDDLE:
            raise NotImplementedError("Middle level partitioning is not implemented yet.")
        elif self.level == HierarchicalLevel.FINE:
            raise NotImplementedError("Fine level partitioning is not implemented yet.")
        else:
            raise ValueError(f"Unknown hierarchical level: {self.level}")

        return StructureMap(index, lexicon)
