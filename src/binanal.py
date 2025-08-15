"""
Binary analysis.
"""

from __future__ import annotations
from collections.abc import Sequence
import enum
import hashlib
import io
import os
from pathlib import Path
import struct
from typing import Callable
from typing import Optional
from typing import Iterable
from typing import Self
import warnings

import lief
import numba as nb
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy


StrPath = str | os.PathLike[str]
LiefParse = str | io.IOBase | os.PathLike | bytes
Range = tuple[int, int]


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


def is_dotnet(data: LiefParse) -> bool:
    data = io.BytesIO(data) if isinstance(data, bytes) else data
    pe = lief.PE.parse(data)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    dd: lief.PE.DataDirectory = pe.data_directory(lief.PE.DataDirectory.TYPES.CLR_RUNTIME_HEADER)
    if not isinstance(dd, lief.PE.DataDirectory):
        raise RuntimeError(f"Expected lief.PE.DataDirectory, got {type(pe)}")
    return bool(dd.rva != 0) and bool(dd.size != 0)


def get_machine_and_subsystem(data: LiefParse) -> tuple[lief.PE.Header.MACHINE_TYPES, lief.PE.OptionalHeader.SUBSYSTEM]:
    data = io.BytesIO(data) if isinstance(data, bytes) else data
    pe = lief.PE.parse(data)
    if not isinstance(pe, lief.PE.Binary):
        raise RuntimeError(f"Expected lief.PE.Binary, got {type(pe)}")
    return pe.header.machine, pe.optional_header.subsystem


def patch_binary(
    data: LiefParse,
    machine: Optional[lief.PE.Header.MACHINE_TYPES] = None,
    subsystem: Optional[lief.PE.OptionalHeader.SUBSYSTEM] = None,
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


def _parse_pe_and_get_size(data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> tuple[lief.PE.Binary, int]:

    pe: Optional[lief.PE.Binary]
    sz: Optional[int]

    if isinstance(data, lief.PE.Binary):
        pe = data
        sz = size
    elif isinstance(data, (str, os.PathLike)):
        pe = lief.PE.parse(data)
        sz = os.path.getsize(data)
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


class CharacteristicGuider:

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

    def __init__(self, data: LiefParse | lief.PE.Binary, size: Optional[int] = None) -> None:
        self.pe, self.size = _parse_pe_and_get_size(data, size)

    def __call__(self) -> npt.NDArray[np.bool_]:
        x = np.full((self.size, len(self.CHARACTERISTICS)), False, dtype=bool)
        for section in self.pe.sections:
            offset = section.offset
            size = section.size
            for i, c in enumerate(self.CHARACTERISTICS):
                x[offset:offset + size, i] = True if section.has_characteristic(c) else False
        return x


class EntropyGuider:

    def __init__(self, data: LiefParse | npt.NDArray[np.uint8]) -> None:
        self.data = self._get_array(data)

    def __call__(self, fast: bool = True, hist: bool = True, radius: int = 256) -> npt.NDArray[np.float64]:
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if not hist and not fast:
            return self.compute_entropy(self.data, radius)
        if not hist and fast:
            return self.compute_entropy_rolling(self.data, radius)
        if hist and not fast:
            return self.compute_histogram_entropy(self.data, radius)
        if hist and fast:
            return self.compute_histogram_entropy_rolling(self.data, radius)  # type: ignore[no-any-return]
        raise RuntimeError("unreachable")

    # -----------------------------
    # 0) SciPy-equivalent entropy (slow, reference implementation)
    # H = log(S) - (1/S) * sum(z * log z), where z = x + eps
    # -----------------------------
    @staticmethod
    def compute_entropy_scipy(x: npt.NDArray[np.uint8], radius: int, epsilon: float = 1e-8) -> npt.NDArray[np.float64]:
        if radius <= 0 or len(x) <= 0:
            raise ValueError(f"The `radius` must be > 0 and `x` must not be empty. Got radius {radius} and length of x {len(x)}.")

        n = int(x.size)
        W = 2 * radius + 1
        out = np.full(n, np.nan, dtype=np.float64)
        win = sliding_window_view(x.astype(np.float64), W) + epsilon
        H = entropy(win, axis=1)
        out[radius:n - radius] = H
        return out

    # -----------------------------
    # 1) Value-weighted (slow, SciPy-equivalent semantics)
    # H = log(S) - (1/S) * sum(z * log z), where z = x + eps
    # -----------------------------
    @staticmethod
    def compute_entropy(x: npt.NDArray[np.uint8], radius: int, epsilon: float = 1e-8) -> npt.NDArray[np.float64]:
        if radius <= 0 or len(x) <= 0:
            raise ValueError(f"The `radius` must be > 0 and `x` must not be empty. Got radius {radius} and length of x {len(x)}.")

        n = int(x.size)
        W = 2 * radius + 1
        out = np.full(n, np.nan, dtype=np.float64)
        win = sliding_window_view(x.astype(np.float64), W) + epsilon  # shape: (n-W+1, W)
        S = win.sum(axis=1)
        T = (win * np.log(win)).sum(axis=1)
        H = np.log(S) - (T / S)
        out[radius:n - radius] = H
        return out

    # -----------------------------
    # 2) Value-weighted (fast rolling, O(N), identical semantics)
    # Precompute tables for z=byte+eps to avoid per-step logs.
    # -----------------------------
    @staticmethod
    def compute_entropy_rolling(x: npt.NDArray[np.uint8], radius: int, epsilon: float = 1e-8) -> npt.NDArray[np.float64]:
        if radius <= 0 or len(x) <= 0:
            raise ValueError(f"The `radius` must be > 0 and `x` must not be empty. Got radius {radius} and length of x {len(x)}.")

        n = int(x.size)
        W = 2 * radius + 1
        out = np.full(n, np.nan, dtype=np.float64)

        # LUTs for z and z*log z at byte values with the same epsilon
        z_vals = np.arange(256, dtype=np.float64) + float(epsilon)
        z_logz_lut = z_vals * np.log(z_vals)

        # Build arrays to scan once
        x_float = x.astype(np.float64)
        t_vals  = z_logz_lut[x]                       # length N

        # Prefix sums with a leading 0 for easy slicing
        S_ps = np.empty(n + 1, dtype=np.float64); S_ps[0] = 0.0
        T_ps = np.empty(n + 1, dtype=np.float64); T_ps[0] = 0.0
        np.cumsum(x_float, out=S_ps[1:])              # sum of raw bytes
        np.cumsum(t_vals,  out=T_ps[1:])              # sum of z*log z

        # Window sums via prefix differences (valid positions: starts 0..n-W)
        S_win = (S_ps[W:] - S_ps[:-W]) + W * float(epsilon)
        T_win =  T_ps[W:] - T_ps[:-W]

        H = np.log(S_win) - (T_win / S_win)
        out[radius:n - radius] = H
        return out

    # -----------------------------
    # 3) Byte-frequency (slow reference, per-window bincount)
    # H = log(W) - (1/W) * sum_c c*log(c), with 0*log 0 := 0
    # -----------------------------
    @staticmethod
    def compute_histogram_entropy(x: npt.NDArray[np.uint8], radius: int) -> npt.NDArray[np.float64]:
        if radius <= 0 or len(x) <= 0:
            raise ValueError(f"The `radius` must be > 0 and `x` must not be empty. Got radius {radius} and length of x {len(x)}.")

        n = int(x.size)
        W = 2 * radius + 1
        out = np.full(n, np.nan, dtype=np.float64)

        logW = np.log(W)
        for c in range(radius, n - radius):
            counts = np.bincount(x[c - radius:c + radius + 1], minlength=256)
            # sum c*log(c) with 0*log 0 := 0 (mask zeros)
            nz = counts[counts > 0].astype(np.float64)
            s_clogc = float((nz * np.log(nz)).sum())
            out[c] = logW - (s_clogc / W)
        return out

    # -----------------------------
    # 4) Byte-frequency (fast rolling, O(N))
    # Maintain counts[256] and s_clogc = sum c*log(c).
    # Only two bins change per step.
    # -----------------------------
    @staticmethod
    @nb.njit(cache=True, nogil=True, fastmath=True)  # type: ignore[misc]
    def compute_histogram_entropy_rolling(x: npt.NDArray[np.uint8], radius: int) -> npt.NDArray[np.float64]:
        if radius <= 0 or len(x) <= 0:
            raise ValueError(f"The `radius` must be > 0 and `x` must not be empty. Got radius {radius} and length of x {len(x)}.")

        n = int(x.size)
        W = 2 * radius + 1
        out = np.full(n, np.nan, dtype=np.float64)

        # log table for k=0..W with log(0)=0 to realize 0*log 0 := 0
        log_tbl = np.empty(W + 1, dtype=np.float64)
        log_tbl[0] = 0.0
        log_tbl[1:] = np.log(np.arange(1, W + 1, dtype=np.float64))
        logW = np.log(W)

        counts = np.zeros(256, dtype=np.int32)
        # initial histogram over [0:W)
        for v in x[:W]:
            counts[int(v)] += 1
        # initial s_clogc
        s_clogc = float((counts.astype(np.float64) * log_tbl[counts]).sum())
        out[radius] = logW - (s_clogc / W)

        # slide window
        for i in range(W, n):
            v_out = int(x[i - W])
            v_in  = int(x[i])
            if v_out != v_in:
                co = counts[v_out]
                ci = counts[v_in]
                # remove old contributions
                s_clogc -= co * log_tbl[co]
                s_clogc -= ci * log_tbl[ci]
                # update counts
                co -= 1; ci += 1
                counts[v_out] = co
                counts[v_in]  = ci
                # add new contributions
                s_clogc += co * log_tbl[co]
                s_clogc += ci * log_tbl[ci]
            out[i - radius] = logW - (s_clogc / W)

        return out

    # -----------------------------
    # Helper to convert input data to a numpy array of uint8.
    # Supports str, Path, bytes, bytearray, memoryview, io.BytesIO, and np.ndarray.
    # Raises TypeError for unsupported types.
    # Raises ValueError if the input is not a 1D array of integers.
    # -----------------------------
    @staticmethod
    def _get_array(data: LiefParse | npt.NDArray[np.int_] | npt.NDArray[np.uint]) -> npt.NDArray[np.uint8]:
        x: npt.NDArray[np.uint8]
        if isinstance(data, (str, os.PathLike)):
            x = np.memmap(data, mode="r", dtype=np.uint8)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            x = np.frombuffer(data, dtype=np.uint8)
        elif isinstance(data, io.BytesIO):
            x = np.frombuffer(data.getbuffer(), dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            if not np.issubdtype(data.dtype, np.integer) or data.ndim != 1:
                raise ValueError(f"Expected a 1D array of integers, got {data.dtype} with shape {data.shape}.")
            x = data.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported input type {type(data)}.")
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

    def __init__(self, data: LiefParse | lief.PE.Binary) -> None:
        warnings.warn("ParserGuider not yet fully operational. All errors marked in same channel.")
        self.pe, self.size = _parse_pe_and_get_size(data)
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
        self.pe, self.size = _parse_pe_and_get_size(data, size)

        self.functions = {
            HierarchicalStructureNone.ANY: self.get_any,

            HierarchicalStructureCoarse.HEADERS: self.get_headers,
            HierarchicalStructureCoarse.SECTION: self.get_sections,
            HierarchicalStructureCoarse.OVERLAY: self.get_overlay,
            HierarchicalStructureCoarse.OTHER: self.get_other,

            HierarchicalStructureMiddle.HEADERS: self.get_headers,
            HierarchicalStructureMiddle.CODE: self.get_code,
            HierarchicalStructureMiddle.DATA: self.get_data,
            HierarchicalStructureMiddle.DIRECTORY: self.get_directory,
            HierarchicalStructureMiddle.OTHERSEC: self.get_othersec,
            HierarchicalStructureMiddle.OVERLAY: self.get_overlay,
            HierarchicalStructureMiddle.OTHER: self.get_other,

            HierarchicalStructureFine.DOS_HEADER: self.get_dos_header,
            HierarchicalStructureFine.COFF_HEADER: self.get_coff_header,
            HierarchicalStructureFine.OPTN_HEADER: self.get_optional_header,
            HierarchicalStructureFine.SECTN_TABLE: self.get_section_table,
            HierarchicalStructureFine.RDATA: self.get_rdata,
            HierarchicalStructureFine.WDATA: self.get_wdata,
            HierarchicalStructureFine.RCODE: self.get_rcode,
            HierarchicalStructureFine.WCODE: self.get_wcode,
            HierarchicalStructureFine.OTHERSEC: self.get_othersec,
            HierarchicalStructureFine.IDATA: self.get_idata,
            HierarchicalStructureFine.EDATA: self.get_edata,
            HierarchicalStructureFine.RELOC: self.get_reloc,
            HierarchicalStructureFine.CLR: self.get_clr,
            HierarchicalStructureFine.OTHERDIR: self.get_otherdir,
            HierarchicalStructureFine.OVERLAY: self.get_overlay,
            HierarchicalStructureFine.OTHER: self.get_other,
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
