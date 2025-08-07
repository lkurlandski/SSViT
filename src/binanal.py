"""
Binary analysis.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Optional

import lief
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import entropy


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


@dataclass(frozen=True)
class SemanticSample:
    buffer: bytes
    parse: Optional[npt.NDArray[np.int32]] = None
    entropy: Optional[npt.NDArray[np.float64]] = None
    characteristics: Optional[npt.NDArray[np.int32]] = None

    def __post_init__(self) -> None:
        lengths = [len(x) for x in (self.buffer, self.parse, self.entropy, self.characteristics) if x is not None]
        if len(set(lengths)) > 1:
            raise ValueError(f"SemanticSample buffers have different lengths: {lengths}")


class SemanticGuider:
    """
    Semantic guides to acompany a byte stream.
    """

    CHARACTERISTICS = [
        # lief.PE.Section.CHARACTERISTICS.TYPE_NO_PAD,
        lief.PE.Section.CHARACTERISTICS.CNT_CODE,
        lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA,
        lief.PE.Section.CHARACTERISTICS.CNT_UNINITIALIZED_DATA,
        # lief.PE.Section.CHARACTERISTICS.LNK_OTHER,
        # lief.PE.Section.CHARACTERISTICS.LNK_INFO,
        # lief.PE.Section.CHARACTERISTICS.LNK_REMOVE,
        # lief.PE.Section.CHARACTERISTICS.LNK_COMDAT,
        lief.PE.Section.CHARACTERISTICS.GPREL,
        # lief.PE.Section.CHARACTERISTICS.MEM_PURGEABLE,  # Reserved for future use
        # lief.PE.Section.CHARACTERISTICS.MEM_16BIT,      # Reserved for future use
        # lief.PE.Section.CHARACTERISTICS.MEM_LOCKED,     # Reserved for future use
        # lief.PE.Section.CHARACTERISTICS.MEM_PRELOAD,    # Reserved for future use
        # lief.PE.Section.CHARACTERISTICS.ALIGN_1BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_2BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_4BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_8BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_16BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_32BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_64BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_128BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_256BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_512BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_1024BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_2048BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_4096BYTES,
        # lief.PE.Section.CHARACTERISTICS.ALIGN_8192BYTES,
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

    def __call__(self, b: bytes) -> SemanticSample:
        parse = None
        if self.do_parse:
            try:
                parse = self.create_parse_guide(b)
            except Exception as err:
                print(f"Error creating parse guide: {err}")

        entropy = None
        if self.do_entropy:
            try:
                entropy = self.create_entropy_guide(b, self.window)
            except Exception as err:
                print(f"Error creating entropy guide: {err}")

        characteristics = None
        if self.do_characteristics:
            try:
                characteristics = self.create_characteristics_guide(b)
            except Exception as err:
                print(f"Error creating permission guide: {err}")

        return SemanticSample(b, parse, entropy, characteristics)

    @staticmethod
    def create_parse_guide(b: bytes) -> npt.NDArray[np.int32]:
        import warnings
        warnings.warn("SemanticGuider.create_parse_guide is not implemented yet. Returning dummy data.")
        return np.random.randint(0, 255, size=len(b), dtype=np.int32)

    @staticmethod
    def create_entropy_guide(b: bytes, w: int, e: float = 1e-8) -> npt.NDArray[np.float64]:
        """
        Args:
            b: The byte stream to analyze.
            w: The window size on each side for entropy calculation.
            e: A small value to avoid division by zero in entropy calculation.
        Returns:
            A numpy array of entropy values for each byte in the stream. The beginning and end of the array
            will be NaN-padded to account for the window size unless `w` is 0.
        """
        x = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
        x = entropy(sliding_window_view(x, w * 2 + 1) + e, axis=1, nan_policy="raise")
        if np.any(np.isnan(x)):
            raise RuntimeError("Entropy calculation resulted in NaN values.")
        x: npt.NDArray[np.float64] = np.concatenate((np.full(w, np.nan), x, np.full(w, np.nan)))
        return x

    @staticmethod
    def create_characteristics_guide(b: bytes) -> npt.NDArray[np.int32]:
        """
        Args:
            b: The byte stream to analyze.
        Returns:
            A numpy array indicating the presence (1) or absense (0) of each characteristic in the
            sections of the binary. Bytes outside of section boundaries will be set to -1.
        """
        x = np.full((len(b), len(SemanticGuider.CHARACTERISTICS)), -1, dtype=np.int32)
        pe = lief.parse(b)
        if pe is None:
            raise RuntimeError("Failed to parse binary with lief.")
        for section in pe.sections:
            offset = section.offset
            size = section.size
            for i, c in enumerate(SemanticGuider.CHARACTERISTICS):
                x[offset:offset + size, i] = 1 if section.has_characteristic(c) else 0
        return x
