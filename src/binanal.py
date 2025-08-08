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

import lief
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy
import torch
from torch import IntTensor
from torch import LongTensor
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

    def __init__(self, do_parse: bool = False, do_entropy: bool = False, do_characteristics: bool = False, window: int = 256) -> None:
        self.do_parse = do_parse
        self.do_entropy = do_entropy
        self.do_characteristics = do_characteristics
        self.window = window

    def __call__(self, data: LiefParse) -> SemanticGuides:
        parse = None
        if self.do_parse:
            try:
                parse = self.create_parse_guide(data)
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
    def create_parse_guide(data: LiefParse) -> IntTensor:
        try:
            pe = lief.parse(data)
        except lief.lief_errors.asn1_bad_tag:
            ...

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
            raise RuntimeError(f"Failed to parse binary with lief.")
        for section in pe.sections:
            offset = section.offset
            size = section.size
            for i, c in enumerate(SemanticGuider.CHARACTERISTICS):
                x[offset:offset + size, i] = 1 if section.has_characteristic(c) else 0
        return x


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
