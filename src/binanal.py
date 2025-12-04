"""
Binary analysis.
"""

from __future__ import annotations
import enum
from functools import cache
import hashlib
import io
import os
from pathlib import Path
import struct
from typing import Callable
from typing import Optional
from typing import Iterable
from typing import Self
from typing import Sequence
import warnings

import lief
import numba as nb
import numpy as np
import numpy.typing as npt


StrPath = str | os.PathLike[str]
LiefParse = str | io.IOBase | os.PathLike[str] | bytes | memoryview | bytearray | list[int]
Range = tuple[int, int]

NPUInt = np.uint8 | np.uint16 | np.uint32 | np.uint64
NPSInt = np.int8 | np.int16 | np.int32 | np.int64
NPInt = NPUInt | NPSInt
NPFloat = np.float16 | np.float32 | np.float64


def get_ranges_numpy(x: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Detects the ranges of consecutive True values in a boolean numpy array.
    """
    if x.dtype != np.bool_ or x.ndim != 1:
        raise TypeError(f"Expected a 1D boolean numpy array, got {type(x)} with shape {x.shape}.")

    n = x.size
    if n == 0:
        z = np.empty(0, dtype=np.int32)
        return z, z

    # Boolean-only edge detection:
    prev = np.empty_like(x)
    prev[0] = False
    prev[1:] = x[:-1]

    nxt = np.empty_like(x)
    nxt[:-1] = x[1:]
    nxt[-1] = False

    lo = np.flatnonzero(x & ~prev).astype(np.int32, copy=False)
    hi = (np.flatnonzero(x & ~nxt) + 1).astype(np.int32, copy=False)
    return lo, hi


def get_machine_and_subsystem(data: LiefParse) -> tuple[lief.PE.Header.MACHINE_TYPES, lief.PE.OptionalHeader.SUBSYSTEM]:
    data = io.BytesIO(data) if isinstance(data, bytes) else data
    pe = lief.PE.parse(data)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    return pe.header.machine, pe.optional_header.subsystem


def patch_binary(
    data: LiefParse,
    machine: Optional[lief.PE.Header.MACHINE_TYPES | int] = None,
    subsystem: Optional[lief.PE.OptionalHeader.SUBSYSTEM | int] = None,
) -> bytes:
    if isinstance(data, (str, os.PathLike)):
        data = bytearray(Path(data).read_bytes())
    elif isinstance(data, bytes):
        data = bytearray(data)
    elif isinstance(data, io.IOBase):
        data = bytearray(data.read())
    else:
        raise TypeError(f"Unsupported input type {type(data)}")

    # Locate headers.
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    pe_sig_off   = e_lfanew
    filehdr_off  = pe_sig_off + 4          # after "PE\0\0"
    opthdr_off   = filehdr_off + 20        # IMAGE_FILE_HEADER is 20 bytes

    # Patch Machine (IMAGE_FILE_HEADER.Machine).
    if machine is not None:
        struct.pack_into("<H", data, filehdr_off + 0, machine.value if isinstance(machine, enum.Enum) else machine)

    # Patch Subsystem (IMAGE_OPTIONAL_HEADER.Subsystem).
    subsys_rel = 68
    if subsystem is not None:
        struct.pack_into("<H", data, opthdr_off + subsys_rel, subsystem.value if isinstance(subsystem, enum.Enum) else subsystem)

    return bytes(data)


def get_timestamp(data: LiefParse) -> int:
    """
    Return the unix timestamp from the PE header.
    """
    pe = _parse_pe_and_get_size(data)[0]
    return int(pe.header.time_date_stamps)


def rearm_disarmed_binary(src: str | Path | bytes, sha: str) -> bytes:
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

    # Machine types, from most to least common for PE files.
    machines = [
        lief.PE.Header.MACHINE_TYPES.I386,
        lief.PE.Header.MACHINE_TYPES.AMD64,
        lief.PE.Header.MACHINE_TYPES.ARM64,
    ]
    for v in lief.PE.Header.MACHINE_TYPES.__dict__.values():
        if type(v).__name__ == "MACHINE_TYPES" and v not in machines:
            machines.append(v)

    # Subsystem types, from most to least common for PE files.
    subsystems = [
        lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_GUI,
        lief.PE.OptionalHeader.SUBSYSTEM.WINDOWS_CUI,
        lief.PE.OptionalHeader.SUBSYSTEM.NATIVE,
    ]
    for v in lief.PE.OptionalHeader.SUBSYSTEM.__dict__.values():
        if type(v).__name__ == "SUBSYSTEM" and v not in subsystems:
            subsystems.append(v)

    for machine in machines:
        for subsystem in subsystems:
            patched = patch_binary(data, machine=machine, subsystem=subsystem)
            s = hashlib.sha256(patched).hexdigest()
            if s == sha:
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
        pe = _parse_pe_and_get_size(pe_bytes)[0]
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

    def add_section_text(self, name: str = ".text", content: Optional[bytes | bytearray] = None, characteristics: Optional[int] = None) -> Self:
        section = lief.PE.Section(name)
        if content is None:
            content = bytearray(0x200)
            content[0] = 0xC3  # RET
        section.content = memoryview(content)
        if characteristics is None:
            characteristics = (
                int(lief.PE.Section.CHARACTERISTICS.CNT_CODE) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_READ) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
            )
        section.characteristics = characteristics
        self.add_section(section, lief.PE.SECTION_TYPES.TEXT)
        return self

    def add_section_data(self, name: str = ".data", content: Optional[bytes | bytearray] = None, characteristics: Optional[int] = None) -> Self:
        if characteristics is not None:
            warnings.warn("Characteristics may be ignored when using the lief.PE.SECTION_TYPES.DATA type.")
        section = lief.PE.Section(name)
        if content is None:
            content = bytearray(0x200)
            content[0] = 0x00
        section.content = memoryview(content)
        if characteristics is None:
            characteristics = (
                int(lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_READ) |
                int(lief.PE.Section.CHARACTERISTICS.MEM_WRITE)
            )
        section.characteristics = characteristics
        self.add_section(section, lief.PE.SECTION_TYPES.DATA)
        return self


def _parse_pe_and_get_size(data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> tuple[lief.PE.Binary, int]:
    # Its usually much faster to parse the binary from a file than any buffer.

    pe: Optional[lief.PE.Binary]
    sz: Optional[int]

    if isinstance(data, lief.PE.Binary):
        pe = data
        sz = size
    elif isinstance(data, (str, os.PathLike)):
        pe = lief.PE.parse(data)
        sz = os.path.getsize(data)
    elif isinstance(data, memoryview):
        pe = lief.PE.parse(data)
        sz = len(data)
    elif isinstance(data, bytes):
        pe = lief.PE.parse(io.BytesIO(data))
        sz = len(data)
    elif isinstance(data, io.IOBase):
        pe = lief.PE.parse(data)
        sz = size
    else:
        raise TypeError(f"Unsupported input type {type(data)}.")

    if pe is None:
        raise RuntimeError(f"lief.PE.parse({type(data)}) returned None")
    if sz is None:
        raise RuntimeError(f"Size must be provided when data is {type(data)}.")

    return pe, sz


def _get_size_of_liefparse(data: LiefParse) -> int:

    sz: Optional[int] = None

    if isinstance(data, (str, os.PathLike)):
        sz = os.path.getsize(data)
    elif isinstance(data, (memoryview, bytes, bytearray, list)):
        sz = len(data)
    elif isinstance(data, io.IOBase):
        warnings.warn("Using io.IOBase, size will be determined by seeking to the end. This has not been tested.")
        data.seek(0, io.SEEK_END)
        sz = data.tell()
        data.seek(0, io.SEEK_SET)
    else:
        raise TypeError(f"Unsupported input type {type(data)}.")

    if sz is None:
        raise RuntimeError(f"Size must be provided when data is {type(data)}.")

    return sz


# List of relevant section characteristics to track.
# We ignore aligment and reserved (for future use).
# See https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-flags
CHARACTERISTICS = (
    # lief.PE.Section.CHARACTERISTICS.TYPE_NO_PAD,               # alignment
    lief.PE.Section.CHARACTERISTICS.CNT_CODE,
    lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA,
    lief.PE.Section.CHARACTERISTICS.CNT_UNINITIALIZED_DATA,
    # lief.PE.Section.CHARACTERISTICS.LNK_OTHER,                 # reserved
    lief.PE.Section.CHARACTERISTICS.LNK_INFO,
    lief.PE.Section.CHARACTERISTICS.LNK_REMOVE,
    lief.PE.Section.CHARACTERISTICS.LNK_COMDAT,
    lief.PE.Section.CHARACTERISTICS.GPREL,
    # lief.PE.Section.CHARACTERISTICS.MEM_PURGEABLE,             # reserved
    # lief.PE.Section.CHARACTERISTICS.MEM_16BIT,                 # reserved
    # lief.PE.Section.CHARACTERISTICS.MEM_LOCKED,                # reserved
    # lief.PE.Section.CHARACTERISTICS.MEM_PRELOAD,               # reserved
    # lief.PE.Section.CHARACTERISTICS.ALIGN_1BYTES,              # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_2BYTES,              # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_4BYTES,              # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_8BYTES,              # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_16BYTES,             # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_32BYTES,             # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_64BYTES,             # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_128BYTES,            # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_256BYTES,            # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_512BYTES,            # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_1024BYTES,           # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_2048BYTES,           # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_4096BYTES,           # alignment
    # lief.PE.Section.CHARACTERISTICS.ALIGN_8192BYTES,           # alignment
    lief.PE.Section.CHARACTERISTICS.LNK_NRELOC_OVFL,
    lief.PE.Section.CHARACTERISTICS.MEM_DISCARDABLE,
    lief.PE.Section.CHARACTERISTICS.MEM_NOT_CACHED,
    lief.PE.Section.CHARACTERISTICS.MEM_NOT_PAGED,
    lief.PE.Section.CHARACTERISTICS.MEM_SHARED,
    lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE,
    lief.PE.Section.CHARACTERISTICS.MEM_READ,
    lief.PE.Section.CHARACTERISTICS.MEM_WRITE,
)


class CharacteristicGuider:

    def __init__(
        self,
        data: LiefParse | lief.PE.Binary,
        size: Optional[int] = None,
        use_packed: bool = False,
        which_characteristics: Sequence[lief.PE.Section.CHARACTERISTICS] = CHARACTERISTICS,
    ) -> None:
        self.pe, self.size = _parse_pe_and_get_size(data, size)
        self.use_packed = use_packed
        self.which_characteristics = which_characteristics

    def _get_characteristic_offsets(self) -> dict[lief.PE.Section.CHARACTERISTICS, list[tuple[int, int]]]:
        offsets: dict[lief.PE.Section.CHARACTERISTICS, list[tuple[int, int]]] = {c: [] for c in self.which_characteristics}

        for section in self.pe.sections:
            offset = section.offset
            size = section.size
            lo = max(0, offset)
            hi = min(self.size, offset + size)
            if hi <= lo:
                continue

            for c in self.which_characteristics:
                if section.has_characteristic(c):
                    offsets[c].append((lo, hi))

        return offsets

    @staticmethod
    def _get_bool_mask_(
        size: int,
        offsets: npt.ArrayLike,
        sizes: npt.ArrayLike,
        flags: npt.ArrayLike,
        mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> npt.NDArray[np.bool_]:
        offsets = np.array(offsets)
        sizes = np.array(sizes)
        flags = np.array(flags)
        if not np.issubdtype(offsets.dtype, np.integer) or not offsets.ndim == 1:
            raise TypeError(f"Expected 1D integer dtype for offsets, got {offsets.dtype} and {offsets.shape}.")
        if not np.issubdtype(sizes.dtype, np.integer) or not sizes.ndim == 1:
            raise TypeError(f"Expected 1D integer dtype for sizes, got {sizes.dtype} and {sizes.shape}.")
        if not (np.issubdtype(flags.dtype, np.integer) or np.issubdtype(flags.dtype, np.bool_)) or not flags.ndim == 2:
            raise TypeError(f"Expected 2D integer/bool dtype for flags, got {flags.dtype} and {flags.shape}.")
        offsets = offsets.astype(np.int64, copy=False)
        sizes = sizes.astype(np.int64, copy=False)
        flags = flags.astype(np.uint8, copy=False)

        if mask is None:
            mask = np.full((size, flags.shape[1]), False, dtype=bool)

        for i in range(offsets.shape[0]):
            o = offsets[i]
            s = sizes[i]
            j = flags[i]
            mask[o:s,j] = 1.0

        return mask

    def _get_bool_mask(self) -> npt.NDArray[np.bool_]:
        # NOTE: allocating the (T, C) array is quite expensive.
        x = np.full((self.size, len(self.which_characteristics)), False, dtype=bool)
        for section in self.pe.sections:
            offset = section.offset
            size = section.size
            lo = max(0, offset)
            hi = min(self.size, offset + size)
            if hi <= lo:
                continue

            for i, c in enumerate(self.which_characteristics):
                x[lo:hi, i] |= section.has_characteristic(c)

        return x

    @staticmethod
    def _get_bit_mask_(
        size: int,
        offsets: npt.NDArray[np.int64],
        sizes: npt.NDArray[np.int64],
        flags: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        x = np.zeros(((size + 7) // 8, flags.shape[1]), dtype=np.uint8)
        _paint_packed_or(x, offsets, sizes, flags, size)
        return x

    def _get_bit_mask(self) -> npt.NDArray[np.uint8]:
        x = np.zeros(((self.size + 7) // 8, len(self.which_characteristics)), dtype=np.uint8)

        sections = list(self.pe.sections)
        n_sec = len(sections)
        K = len(self.which_characteristics)

        offsets = np.empty(n_sec, dtype=np.int64)
        sizes   = np.empty(n_sec, dtype=np.int64)
        flags   = np.zeros((n_sec, K), dtype=np.uint8)

        for s, sec in enumerate(sections):
            offsets[s] = int(getattr(sec, "offset", 0) or 0)
            sizes[s]   = int(getattr(sec, "size",   0) or 0)
            for i, c in enumerate(self.which_characteristics):
                flags[s, i] = 1 if sec.has_characteristic(c) else 0

        _paint_packed_or(x, offsets, sizes, flags, self.size)
        return x

    def __call__(self) -> npt.NDArray[np.bool_] | npt.NDArray[np.uint8]:
        if self.use_packed:
            return self._get_bit_mask()
        return self._get_bool_mask()


@nb.njit(cache=True, parallel=False, fastmath=True)  # type: ignore[misc]
def _paint_bool_or(x: npt.NDArray[np.uint8], offsets: npt.NDArray[np.int64], sizes: npt.NDArray[np.int64], flags: npt.NDArray[np.uint8], T: int) -> None:
    """
    This is the equivalent of _paint_bool_or that could be used for the boolean version of the CharacteristicGuider.

    NOTE: This has not been tested.
    """
    n_sec, C = flags.shape
    for s in range(n_sec):
        lo = offsets[s]
        hi = lo + sizes[s]
        if lo < 0:
            lo = 0
        if hi > T:
            hi = T
        if hi <= lo:
            continue
        for k in range(C):
            if flags[s, k] != 0:
                for j in range(lo, hi):
                    x[j, k] = True


@nb.njit(cache=True, parallel=False, fastmath=True)  # type: ignore[misc]
def _paint_packed_or(x: npt.NDArray[np.uint8], offsets: npt.NDArray[np.int64], sizes: npt.NDArray[np.int64], flags: npt.NDArray[np.uint8], T: int) -> None:
    """
    I honestly have no idea how this works, but it seems to be correct and is super fast.

    See commit 2b74f189e5ca5fcd9f25e855d25d40877a16399b for the native numpy version.

    Using njit(parellel=True) appeared to be slower for sequences of length ~1MiB.
    """
    n_sec, C = flags.shape
    for s in range(n_sec):
        lo = offsets[s]
        hi = lo + sizes[s]
        if lo < 0:
            lo = 0
        if hi > T:
            hi = T
        if hi <= lo:
            continue

        last = hi - 1
        b0 = lo >> 3
        b1 = last >> 3
        start_bit = lo & 7
        end_bits = (last & 7) + 1

        for k in nb.prange(C):
            if flags[s, k] == 0:
                continue
            if b0 == b1:
                m = ((0xFF << start_bit) & 0xFF) & ((1 << end_bits) - 1)
                x[b0, k] = x[b0, k] | m
            else:
                m0 = (0xFF << start_bit) & 0xFF
                x[b0, k] = x[b0, k] | m0
                for b in range(b0 + 1, b1):
                    x[b, k] = x[b, k] | 0xFF
                m1 = (1 << end_bits) - 1
                x[b1, k] = x[b1, k] | m1


class ParserGuider:

    # NOTE: I couldn't figure out how to properly catch/handle lief errors.

    # PARSEERRORS = [
    #     lief.lief_errors.asn1_bad_tag,
    #     lief.lief_errors.build_error,
    #     lief.lief_errors.conversion_error,
    #     lief.lief_errors.corrupted,
    #     lief.lief_errors.data_too_large,
    #     lief.lief_errors.file_error,
    #     lief.lief_errors.file_format_error,
    #     lief.lief_errors.not_found,
    #     lief.lief_errors.not_implemented,
    #     lief.lief_errors.not_supported,
    #     lief.lief_errors.parsing_error,
    #     lief.lief_errors.read_error,
    #     lief.lief_errors.read_out_of_bound,
    #     lief.lief_errors.require_extended_version
    # ]

    PARSEERRORS = [Exception]

    def __init__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> None:
        warnings.warn("ParserGuider not yet fully operational. All errors marked in same channel.")
        self.pe, self.size = _parse_pe_and_get_size(data, size)
        self.guide = np.zeros((self.size, len(ParserGuider.PARSEERRORS)), dtype=bool)

    def __call__(self, simple: bool) -> npt.NDArray[np.bool_]:
        if simple:
            self.build_simple_guide()
        else:
            self.build_complex_guide()
        return self.guide

    def build_simple_guide(self) -> None:
        if self.pe is None:
            raise RuntimeError("ParserGuider has not been called yet.")

        probes: list[tuple[str, tuple[int, int]]] = [
            ("header", (0, min(0x40, self.size))),
            ("optional_header", (0, min(0x100, self.size))),
            ("sections", (0x200, self.size)),
            ("data_directories", (0, min(0x200, self.size))),
            ("imports", (0, self.size)),
            ("exports", (0, self.size)),
            ("resources", (0, self.size)),
        ]
        for attr, (lo, hi) in probes:
            try:
                _ = getattr(self.pe, attr)
                if attr == "sections":
                    for s in self.pe.sections:
                        _ = (s.name, s.size)
                elif attr == "data_directories":
                    for dd in self.pe.data_directories:
                        _ = (dd.type, dd.size)
                elif attr == "imports":
                    for im in self.pe.imports:
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
        lo, hi = self._compute_coff_header_region_boundaries(self.pe, self.size)
        try:
            getattr(self.pe, "header")
        except Exception as e:
            self._mark_guide(lo, hi, e)

        # Check the optional header
        lo, hi = self._compute_optional_header_boundaries(self.pe, self.size)
        try:
            getattr(self.pe, "optional_header")
        except Exception as e:
            self._mark_guide(lo, hi, e)

        # Check the section table
        lo, hi = self._compute_section_table_boundaries(self.pe, self.size)
        try:
            getattr(self.pe, "sections")
        except Exception as e:
            self._mark_guide(lo, hi, e)
        else:
            # Check the sections
            if self.pe.sections is not None:
                for s in self.pe.sections:
                    lo = s.pointerto_raw_data
                    hi = min(lo + s.size, self.size)
                    try:
                        # Touch properties that often trigger lazy reads/validations
                        _ = (s.name, s.size, s.virtual_address, s.content)
                    except Exception as e:
                        self._mark_guide(lo, hi, e)

        # Check the data directories
        lo, hi = self._compute_optional_header_boundaries(self.pe, self.size)
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
                            lo, hi = self._compute_optional_header_boundaries(self.pe, self.size)
                        else:
                            off = self.pe.rva_to_offset(rva)
                            if off is None or off < 0:
                                return None
                            lo = self.pe.rva_to_offset(rva)
                            hi = min(lo + size, self.size)
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
    HEADERS   = "headers"   # Headers                (ANY)
    SECTION   = "sections"  # Sections               (ANY)
    OVERLAY   = "overlay"   # Overlay data           (ANY)
    DIRECTORY = "directory" # Directories            (ANY)
    OTHER     = "other"     # All other bytes        (ANY)


class HierarchicalStructureMiddle(HierarchicalStructure):
    HEADERS   = "headers"   # Headers               (ANY)
    OVERLAY   = "overlay"   # Overlay data          (ANY)
    DIRECTORY = "directory" # Directories           (ANY)
    OTHER     = "other"     # All other bytes       (ANY)
    DNETCODE  = "dnetcode"  # .NET code sections    (SECTION)
    DNETDATA  = "dnetdata"  # .NET data sections    (SECTION)
    CODE      = "code"      # Code sections         (SECTION)
    DATA      = "data"      # Data sections         (SECTION)


class HierarchicalStructureFine(HierarchicalStructure):
    OVERLAY = "overlay"          # Overlay data     (ANY)
    OTHER   = "other"            # All other bytes  (ANY)
    DOS_HEADER  = "dos_header"   # DOS header       (HEADER)
    DOS_STUB    = "dos_stub"     # DOS stub         (HEADER)
    COFF_HEADER = "coff_header"  # COFF header      (HEADER)
    OPTN_HEADER = "optn_header"  # Optional header  (HEADER)
    SECTN_TABLE = "sectn_table"  # Section table    (HEADER)
    RDNETCODE = "rdnetcode"      # Read-only .NET code (DNETCODE)
    WDNETCODE = "wdnetcode"      # Writeable .NET code (DNETCODE)
    RDNETDATA = "rdnetdata"      # Read-only .NET data (DNETDATA)
    WDNETDATA = "wdnetdata"      # Writeable .NET data (DNETDATA)
    RCODE = "rcode"              # Read-only code   (CODE)
    WCODE = "wcode"              # Writeable code   (CODE)
    RDATA = "rdata"              # Read-only data   (DATA)
    WDATA = "wdata"              # Writeable data   (DATA)
    IDATA    = "idata"           # Import descriptors         (DIRECTORY)
    DELAYIMP = "delayimp"        # Delay import descriptors   (DIRECTORY)
    EDATA    = "edata"           # Export descriptors         (DIRECTORY)
    RESOURCE = "resource"        # Resource directory (.rsrc) (DIRECTORY)
    TLS      = "tls"             # TLS directory / callbacks  (DIRECTORY)
    LOADCFG  = "loadcfg"         # Load config directory      (DIRECTORY)
    RELOC    = "reloc"           # Relocation table           (DIRECTORY)
    DEBUG    = "debug"           # Debug directory            (DIRECTORY)
    CLR      = "clr"             # CLR/.NET header/metadata   (DIRECTORY)
    OTHERDIR = "otherdir"        # Any other data directory   (DIRECTORY)


LEVEL_STRUCTURE_MAP: dict[HierarchicalLevel, type[HierarchicalStructure]] = {
    HierarchicalLevel.NONE: HierarchicalStructureNone,
    HierarchicalLevel.COARSE: HierarchicalStructureCoarse,
    HierarchicalLevel.MIDDLE: HierarchicalStructureMiddle,
    HierarchicalLevel.FINE: HierarchicalStructureFine,
}


DIRECTORY_STRUCTURES = (
    HierarchicalStructureCoarse.DIRECTORY,
    HierarchicalStructureMiddle.DIRECTORY,
    HierarchicalStructureFine.IDATA,
    HierarchicalStructureFine.DELAYIMP,
    HierarchicalStructureFine.EDATA,
    HierarchicalStructureFine.RESOURCE,
    HierarchicalStructureFine.TLS,
    HierarchicalStructureFine.LOADCFG,
    HierarchicalStructureFine.RELOC,
    HierarchicalStructureFine.DEBUG,
    HierarchicalStructureFine.CLR,
    HierarchicalStructureFine.OTHERDIR,
)


class StructureParser:

    _SCN_MEM_EXECUTE = 0x20000000
    _SCN_MEM_READ    = 0x40000000
    _SCN_MEM_WRITE   = 0x80000000

    def __init__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> None:
        self._pe, self._size = _parse_pe_and_get_size(data, size)

    @property
    def pe(self) -> lief.PE.Binary:
        return self._pe

    @property
    def size(self) -> int:
        return self._size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructureParser):
            return NotImplemented
        return (self.pe is other.pe) and (self.size == other.size)

    def __hash__(self) -> int:
        return hash((id(self.pe), self.size))

    def __call__(self, structure: HierarchicalStructure) -> list[Range]:
        match structure:
            # None
            case HierarchicalStructureNone.ANY:
                ranges = self.get_any()
            # Coarse
            case HierarchicalStructureCoarse.HEADERS:
                ranges = self.get_headers()
            case HierarchicalStructureCoarse.SECTION:
                ranges = self.get_sections()
            case HierarchicalStructureCoarse.OVERLAY:
                ranges = self.get_overlay()
            case HierarchicalStructureCoarse.DIRECTORY:
                ranges = self.get_directory()
            case HierarchicalStructureCoarse.OTHER:
                ranges = self.get_other()
            # Middle
            case HierarchicalStructureMiddle.HEADERS:
                ranges = self.get_headers()
            case HierarchicalStructureMiddle.OVERLAY:
                ranges = self.get_overlay()
            case HierarchicalStructureMiddle.OTHER:
                ranges = self.get_other()
            case HierarchicalStructureMiddle.DIRECTORY:
                ranges = self.get_directory()
            case HierarchicalStructureMiddle.DNETCODE:
                ranges = self.get_dnet_code()
            case HierarchicalStructureMiddle.DNETDATA:
                ranges = self.get_dnet_data()
            case HierarchicalStructureMiddle.CODE:
                ranges = self.get_code()
            case HierarchicalStructureMiddle.DATA:
                ranges = self.get_data()
            # Fine
            case HierarchicalStructureFine.OVERLAY:
                ranges = self.get_overlay()
            case HierarchicalStructureFine.OTHER:
                ranges = self.get_other()
            case HierarchicalStructureFine.DOS_HEADER:
                ranges = self.get_dos_header()
            case HierarchicalStructureFine.DOS_STUB:
                ranges = self.get_dos_stub()
            case HierarchicalStructureFine.COFF_HEADER:
                ranges = self.get_coff_header()
            case HierarchicalStructureFine.OPTN_HEADER:
                ranges = self.get_optional_header()
            case HierarchicalStructureFine.SECTN_TABLE:
                ranges = self.get_section_table()
            case HierarchicalStructureFine.RDNETCODE:
                ranges = self.get_rdnet_code()
            case HierarchicalStructureFine.WDNETCODE:
                ranges = self.get_wdnet_code()
            case HierarchicalStructureFine.RDNETDATA:
                ranges = self.get_rdnet_data()
            case HierarchicalStructureFine.WDNETDATA:
                ranges = self.get_wdnet_data()
            case HierarchicalStructureFine.RDATA:
                ranges = self.get_rdata()
            case HierarchicalStructureFine.WDATA:
                ranges = self.get_wdata()
            case HierarchicalStructureFine.RCODE:
                ranges = self.get_rcode()
            case HierarchicalStructureFine.WCODE:
                ranges = self.get_wcode()
            case HierarchicalStructureFine.IDATA:
                ranges = self.get_idata()
            case HierarchicalStructureFine.DELAYIMP:
                ranges = self.get_delayimp()
            case HierarchicalStructureFine.EDATA:
                ranges = self.get_edata()
            case HierarchicalStructureFine.RESOURCE:
                ranges = self.get_resource()
            case HierarchicalStructureFine.TLS:
                ranges = self.get_tls()
            case HierarchicalStructureFine.LOADCFG:
                ranges = self.get_loadcfg()
            case HierarchicalStructureFine.RELOC:
                ranges = self.get_reloc()
            case HierarchicalStructureFine.DEBUG:
                ranges = self.get_debug()
            case HierarchicalStructureFine.CLR:
                ranges = self.get_clr()
            case HierarchicalStructureFine.OTHERDIR:
                ranges = self.get_otherdir()
            case _:
                raise ValueError(f"Unknown structure type: {structure}.")

        return self._norm(ranges)

    # ---------- Range Helpers ----------
    def _norm(self, ranges: Iterable[Range]) -> list[Range]:
        """Normalize ranges, clipping to file size."""
        ranges = [r for r in ranges if r is not None]
        ranges = [self._clip(*r) for r in ranges]
        ranges = [r for r in ranges if r is not None]
        return ranges

    def _clip(self, start: int, end: int) -> Optional[Range]:
        """Clip range to be within the file size. Return None if invalid."""
        if start is None or end is None:
            return None
        start = max(0, start)
        end   = min(self.size, end)
        if end <= start:
            return None
        return (start, end)

    @staticmethod
    def _ranges_overlap(a: Range, b: Range) -> bool:
        """Return True if byte ranges a and b overlap at all."""
        a_lo, a_hi = a
        b_lo, b_hi = b
        return not (a_hi <= b_lo or b_hi <= a_lo)

    # ---------- PE Helpers ----------
    def _sec_raw_range(self, s: lief.PE.Section) -> Optional[Range]:
        """Get the raw data range of a section if it has raw data."""
        if s.sizeof_raw_data == 0:
            return None
        return (int(s.pointerto_raw_data), int(s.pointerto_raw_data + s.sizeof_raw_data))

    def _select_sections(self, function: Callable[[lief.PE.Section], bool]) -> list[Range]:
        """Filter sections by a function and return their raw ranges."""
        out = []
        for s in self.pe.sections:
            if function(s):
                r = self._sec_raw_range(s)
                if r:
                    out.append(r)
        return out

    def _dir_range(self, d: lief.PE.DataDirectory) -> Optional[Range]:
        """Return the raw range of the directory."""
        try:
            # SECURITY (certificate table) is special: VirtualAddress is a FILE OFFSET (not an RVA)
            if d.type == lief.PE.DataDirectory.TYPES.CERTIFICATE_TABLE:
                off = int(d.rva)
                return (off, off + int(d.size))
            if d.size == 0:
                return None
            if d.rva == 0:
                return None
            off = self.pe.rva_to_offset(int(d.rva))
            if off is None or off < 0:
                return None
            return (int(off), int(off + d.size))
        except Exception:
            print(f"An error occurred while computing range for data directory {d.type}.")
            return None

    def _dir_type_range(self, types: set[lief.PE.DataDirectory.TYPES]) -> list[Range]:
        """Return the raw ranges of all data directories of the given types."""
        out = []
        for d in self.pe.data_directories:
            if d.type in types:
                r = self._dir_range(d)
                if r:
                    out.append(r)
        return out

    def _rva_size_to_range(self, rva: int, size: int) -> Optional[Range]:
        """Convert an RVA+size pair into a raw file (offset, offset+size) range."""
        if not rva or not size:
            return None
        off = self.pe.rva_to_offset(int(rva))
        if off is None or off < 0:
            return None
        return (int(off), int(off + size))

    @cache
    def _get_dotnet_sections(self) -> list[lief.PE.Section]:
        """Identify sections that belong to the .NET / CLR world."""
        # Get the CLR directory.
        for clr_dir in self.pe.data_directories:
            if clr_dir.type == lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER:
                break
        else:
            return []
        if clr_dir.size == 0 or clr_dir.rva == 0:
            return []
        clr_rva  = int(clr_dir.rva)
        clr_size = int(clr_dir.size)

        # Parse IMAGE_COR20_HEADER
        def _u32(off: int) -> int:
            return int.from_bytes(header_bytes[off : off + 4], "little", signed=False)

        HEADER_LEN = 0x48
        try:
            header_bytes = bytes(self.pe.get_content_from_virtual_address(clr_rva, HEADER_LEN))
        except Exception:
            warnings.warn("A CLR Header was located but we failed to read its bytes.")
            return []
        header_bytes = header_bytes.ljust(HEADER_LEN, b"\x00")

        MetaDataRVA                     = _u32(0x08)
        MetaDataSize                    = _u32(0x0C)
        ResourcesRVA                    = _u32(0x18)
        ResourcesSize                   = _u32(0x1C)
        StrongNameSignatureRVA          = _u32(0x20)
        StrongNameSignatureSize         = _u32(0x24)
        CodeManagerTableRVA             = _u32(0x28)
        CodeManagerTableSize            = _u32(0x2C)
        VTableFixupsRVA                 = _u32(0x30)
        VTableFixupsSize                = _u32(0x34)
        ExportAddressTableJumpsRVA      = _u32(0x38)
        ExportAddressTableJumpsSize     = _u32(0x3C)
        ManagedNativeHeaderRVA          = _u32(0x40)
        ManagedNativeHeaderSize         = _u32(0x44)

        # Gather candidate .NET regions.
        def _add_region_from_rva_size(accum_ranges: list[Range], rva: int, sz: int) -> None:
            if not rva or not sz:
                return
            rng = self._rva_size_to_range(int(rva), int(sz))
            if rng is not None:
                accum_ranges.append(rng)

        dotnet_file_ranges: list[Range] = []

        _add_region_from_rva_size(dotnet_file_ranges, clr_rva, clr_size)
        _add_region_from_rva_size(dotnet_file_ranges, MetaDataRVA, MetaDataSize)
        _add_region_from_rva_size(dotnet_file_ranges, ResourcesRVA, ResourcesSize)
        _add_region_from_rva_size(dotnet_file_ranges, StrongNameSignatureRVA, StrongNameSignatureSize)
        _add_region_from_rva_size(dotnet_file_ranges, CodeManagerTableRVA, CodeManagerTableSize)
        _add_region_from_rva_size(dotnet_file_ranges, VTableFixupsRVA, VTableFixupsSize)
        _add_region_from_rva_size(dotnet_file_ranges, ExportAddressTableJumpsRVA, ExportAddressTableJumpsSize)
        _add_region_from_rva_size(dotnet_file_ranges, ManagedNativeHeaderRVA, ManagedNativeHeaderSize)

        if not dotnet_file_ranges:
            return []

        # Map ranges to sections.
        dotnet_secs: set[lief.PE.Section] = set()
        for sec in self.pe.sections:
            sec_range = self._sec_raw_range(sec)
            if sec_range is None:
                continue
            for dr in dotnet_file_ranges:
                if self._ranges_overlap(sec_range, dr):
                    dotnet_secs.add(sec)
                    break

        return list(dotnet_secs)

    # ---------- PE Section Logic ----------
    @staticmethod
    def _is_code(s: lief.PE.Section) -> bool:
        """Returns True if the section contains "code" else False."""
        c = int(s.characteristics)
        is_executable = bool(c & StructureParser._SCN_MEM_EXECUTE)
        is_code_section = False # bool(c & StructureParser._SCN_CNT_CODE)
        return is_executable or is_code_section

    @staticmethod
    def _is_rcode(s: lief.PE.Section) -> bool:
        """Returns True if the section contains read-only "code" else False."""
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_code(s) and not is_writeable

    @staticmethod
    def _is_wcode(s: lief.PE.Section) -> bool:
        """Returns True if the section contains writeable "code" else False."""
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_code(s) and is_writeable

    @staticmethod
    def _is_data(s: lief.PE.Section) -> bool:
        """Returns True if the section contains "data" else False."""
        c = int(s.characteristics)
        is_readable = bool(c & StructureParser._SCN_MEM_READ)
        is_executable = bool(c & StructureParser._SCN_MEM_EXECUTE)
        is_initialized = False # bool(c & StructureParser._SCN_CNT_IN_DATA)
        is_uninitialized = False # bool(c & StructureParser._SCN_CNT_UN_DATA)
        return is_readable and (not is_executable or is_initialized or is_uninitialized)

    @staticmethod
    def _is_rdata(s: lief.PE.Section) -> bool:
        """Returns True if the section contains read-only "data" else False."""
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_data(s) and not is_writeable

    @staticmethod
    def _is_wdata(s: lief.PE.Section) -> bool:
        """Returns True if the section contains writeable "data" else False."""
        c = int(s.characteristics)
        is_writeable = bool(c & StructureParser._SCN_MEM_WRITE)
        return StructureParser._is_data(s) and is_writeable

    def _is_dotnet(self, s: lief.PE.Section) -> bool:
        """Returns True if the section is part of the .NET runtime else False."""
        return s in self._get_dotnet_sections()

    # ---------- ANY ---------- #

    def get_any(self) -> list[Range]:
        return self._norm([(0, self.size)])

    @cache
    def get_overlay(self) -> list[Range]:
        sec_end = 0
        for s in self.pe.sections:
            if s.sizeof_raw_data and s.pointerto_raw_data:
                sec_end = max(sec_end, int(s.pointerto_raw_data + s.sizeof_raw_data))
        if self.size > sec_end:
            return self._norm([(sec_end, self.size)])
        return []

    @cache
    def get_other(self) -> list[Range]:

        def _normalize_and_merge(ranges: Iterable[Range], size: int) -> list[Range]:
            """Clamp to [0,size), drop empties, sort, and merge overlaps/adjacent."""
            if size <= 0:
                return []
            norm: list[Range] = []
            for lo, hi in ranges:
                if hi <= 0 or lo >= size:
                    continue
                lo = 0 if lo < 0 else lo
                hi = size if hi > size else hi
                if hi > lo:
                    norm.append((lo, hi))
            if not norm:
                return []

            norm.sort()
            merged: list[Range] = []
            cur_lo, cur_hi = norm[0]
            for lo, hi in norm[1:]:
                if lo <= cur_hi:
                    if hi > cur_hi:
                        cur_hi = hi
                else:
                    merged.append((cur_lo, cur_hi))
                    cur_lo, cur_hi = lo, hi
            merged.append((cur_lo, cur_hi))
            return merged

        def _complement_of_merged(merged: list[Range], size: int) -> list[Range]:
            """Complement of a merged, sorted, non-overlapping interval set within [0,size)."""
            if size <= 0:
                return []
            other: list[Range] = []
            cursor = 0
            for lo, hi in merged:
                if cursor < lo:
                    other.append((cursor, lo))
                cursor = hi
            if cursor < size:
                other.append((cursor, size))
            return other

        ranges = self.get_headers() + self.get_sections() + self.get_overlay() + self.get_directory()
        merged = _normalize_and_merge(ranges, self.size)
        other = _complement_of_merged(merged, self.size)
        return self._norm(other)

    # ---------- HEADER ---------- #

    def get_headers(self) -> list[Range]:
        return self._norm([(0, self.pe.sizeof_headers)])

    def get_dos_header(self) -> list[Range]:
        return self._norm([(0, 64)])

    def get_dos_stub(self) -> list[Range]:
        e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
        stub_start = 64
        stub_end   = e_lfanew
        return self._norm([(stub_start, stub_end)])

    def get_coff_header(self) -> list[Range]:
        e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
        coff_start = e_lfanew + 4
        coff_end   = coff_start + 20
        return self._norm([(coff_start, coff_end)])

    def get_optional_header(self) -> list[Range]:
        e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
        coff_start = e_lfanew + 4
        coff_end   = coff_start + 20
        opt_start  = coff_end
        opt_end    = opt_start + int(self.pe.header.sizeof_optional_header)
        return self._norm([(opt_start, opt_end)])

    def get_section_table(self) -> list[Range]:
        e_lfanew = int(self.pe.dos_header.addressof_new_exeheader)
        coff_start = e_lfanew + 4
        coff_end   = coff_start + 20
        opt_start  = coff_end
        opt_end    = opt_start + int(self.pe.header.sizeof_optional_header)
        nsects     = int(self.pe.header.numberof_sections)
        sectab_start = opt_end
        sectab_end   = sectab_start + 40 * nsects
        return self._norm([(sectab_start, sectab_end)])

    # --------- SECTION ---------- #

    def get_sections(self) -> list[Range]:
        out = []
        for s in self.pe.sections:
            r = self._sec_raw_range(s)
            if r:
                out.append(r)
        return self._norm(out)

    def get_dnet_code(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_code(s)))

    def get_rdnet_code(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_rcode(s)))

    def get_wdnet_code(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_wcode(s)))

    def get_dnet_data(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_data(s)))

    def get_rdnet_data(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_rdata(s)))

    def get_wdnet_data(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: self._is_dotnet(s) and self._is_wdata(s)))

    def get_code(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_code(s)))

    def get_rcode(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_rcode(s)))

    def get_wcode(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_wcode(s)))

    def get_data(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_data(s)))

    def get_rdata(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_rdata(s)))

    def get_wdata(self) -> list[Range]:
        return self._norm(self._select_sections(lambda s: not self._is_dotnet(s) and self._is_wdata(s)))

    # --------- DIRECTORY ---------- #

    def get_directory(self) -> list[Range]:
        out = []
        for d in self.pe.data_directories:
            r = self._dir_range(d)
            if r:
                out.append(r)
        return self._norm(out)

    def get_idata(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.IMPORT_TABLE}))

    def get_edata(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.EXPORT_TABLE}))

    def get_reloc(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.BASE_RELOCATION_TABLE}))

    def get_clr(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER}))

    def get_resource(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.RESOURCE_TABLE}))

    def get_tls(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.TLS_TABLE}))

    def get_loadcfg(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.LOAD_CONFIG_TABLE}))

    def get_delayimp(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.DELAY_IMPORT_DESCRIPTOR}))

    def get_debug(self) -> list[Range]:
        return self._norm(self._dir_type_range({lief.PE.DataDirectory.TYPES.DEBUG_DIR}))

    def get_otherdir(self) -> list[Range]:
        special = {
            lief.PE.DataDirectory.TYPES.IMPORT_TABLE,
            lief.PE.DataDirectory.TYPES.EXPORT_TABLE,
            lief.PE.DataDirectory.TYPES.BASE_RELOCATION_TABLE,
            lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER,
            lief.PE.DataDirectory.TYPES.RESOURCE_TABLE,
            lief.PE.DataDirectory.TYPES.TLS_TABLE,
            lief.PE.DataDirectory.TYPES.LOAD_CONFIG_TABLE,
            lief.PE.DataDirectory.TYPES.DELAY_IMPORT_DESCRIPTOR,
            lief.PE.DataDirectory.TYPES.DEBUG_DIR,
        }
        other_ranges = []
        for d in self.pe.data_directories:
            if d.type in special:
                continue
            r = self._dir_range(d)
            if r:
                other_ranges.append(r)
        return self._norm(other_ranges)
