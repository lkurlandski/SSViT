"""
Aid and abet the main module.
"""

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import Field
from enum import Enum
from functools import partial
import json
import os
import sys
from types import GenericAlias
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Self
from typing import Sequence
from typing import Union
from typing import get_type_hints
from typing import get_args
from typing import get_origin
import warnings

import lief
import torch

from src.binanal import HierarchicalLevel
from src.binanal import CHARACTERISTICS
from src.trainer import TrainerArgs
from src.utils import str_to_bool


class Design(Enum):
    FLAT         = "flat"          # Flat
    HIERARCHICAL = "hierarchical"  # Hierarchical
    STRUCTURAL   = "structural"    # Structured


class Architecture(Enum):
    MCV = "mcv"  # Original MalConv
    MC2 = "mc2"  # Low-memory MalConv
    MCG = "mcg"  # Low-memory MalConv with Gating
    VIT = "vit"  # Vision Transformer


class PatcherArchitecture(Enum):
    BAS = "bas"      # Basic
    CNV = "cnv"      # Convolutional
    HCV = "hcv"      # Hierarchical Convolutional
    MEM = "mem"      # Low-Memory
    EXP = "exp"      # Low-Memory with Experts
    DWC = "dwc"      # Depthwise Convolutional


class PositionalEncodingArchitecture(Enum):
    NONE     = "none"      # No positional encoding
    FIXED    = "fixed"     # Sinusoidal absolute positional encoding
    LEARNED  = "learned"   # Learned absolute positional encoding


class PatchPositionalEncodingArchitecture(Enum):
    NONE = "none" # None
    REL  = "rel"  # Relative
    ABS  = "abs"  # Absolute
    BTH  = "bth"  # Relative and Absolute


class Scheduler(Enum):
    NONE = "none"  # Fixed learning rate
    CUST = "cust"  # Custom schduler
    OCLR = "oclr"  # One Cycle Learning Rate


def any_to_section_characteristic(s: lief.PE.Section.CHARACTERISTICS | int | str) -> lief.PE.Section.CHARACTERISTICS:
    if isinstance(s, lief.PE.Section.CHARACTERISTICS):
        return s
    elif isinstance(s, int):
        return lief.PE.Section.CHARACTERISTICS(s)
    elif isinstance(s, str):
        return lief.PE.Section.CHARACTERISTICS[s]
    else:
        raise TypeError(f"Cannot convert {s!r} to lief.PE.Section.CHARACTERISTICS.")


# TODO: figure out how to elegantly combine different dataclasses into a new class.
# TODO: write an ArgumentParser that takes a dataclass and generates arguments.
@dataclass
class MainArgs:
    design: Design = Design.FLAT
    arch: Architecture = Architecture.MCV
    parch: PatcherArchitecture = PatcherArchitecture.MEM
    posenc: PositionalEncodingArchitecture = PositionalEncodingArchitecture.FIXED
    patchposenc: PatchPositionalEncodingArchitecture = PatchPositionalEncodingArchitecture.NONE
    model_config_str: str = "{}"
    seed: int = 0
    do_parser: bool = False
    do_entropy: bool = False
    which_characteristics: tuple[lief.PE.Section.CHARACTERISTICS, ...] = tuple()
    level: HierarchicalLevel = HierarchicalLevel.NONE
    share_embeddings: bool = False
    share_patchers: bool = False
    ignore_directory_structures: bool = True
    tr_num_samples: Optional[int] = None
    vl_num_samples: Optional[int] = None
    ts_num_samples: Optional[int] = None
    max_length: Optional[int] = None
    max_length_per_structure: Optional[int] = None
    max_structures: Optional[int] = None
    num_streams: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    muddy_padded: bool = True
    prefetch_factor: int = 1
    tr_batch_size: int = 1
    vl_batch_size: int = -1
    ts_batch_size: int = -1
    sched: Scheduler = Scheduler.NONE
    lr_beg: float = 1e-5
    lr_max: float = 1e-3
    lr_end: float = 1e-6
    warmup_ratio: float = 0.00
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    device: torch.device = torch.device("cpu")
    ddp: bool = False
    fsdp: bool = False
    fsdp_offload: bool = True
    tf32: bool = False
    enable_checkpoint: bool = False
    enable_compile: bool = False
    static_shapes_bin_patcher_seq_lengths: bool = False
    static_shapes_bin_patcher_batch_sizes: bool = False
    static_shapes_bin_backbone_batch_sizes: bool = False
    static_shapes_bin_backbone_seq_lengths: bool = False
    find_unused_parameters: bool = False
    resume: bool = False
    resume_checkpoint: Optional[str] = None
    embedding_freeze: bool = False

    def __post_init__(self) -> None:
        if self.tr_batch_size == 1 or self.vl_batch_size == 1 or self.ts_batch_size == 1:
            raise NotImplementedError("Batch size of 1 is not supported right now. See https://docs.pytorch.org/docs/stable/data#working-with-collate-fn.")
        self.vl_batch_size = self.tr_batch_size if self.vl_batch_size < 1 else self.vl_batch_size
        self.ts_batch_size = self.vl_batch_size if self.ts_batch_size < 1 else self.ts_batch_size
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        if self.ddp and self.fsdp:
            raise ValueError("ddp and fsdp cannot both be True.")
        if "RANK" in os.environ and not (self.ddp or self.fsdp):
            raise ValueError("If running with torchrun, either --ddp or --fsdp must be set.")
        if "RANK" not in os.environ and (self.ddp or self.fsdp):
            raise ValueError("If --ddp or --fsdp is set, the script must be launched with torchrun.")
        self.pin_memory = self.pin_memory and self.device.type == "cuda"
        self.num_streams = 0 if self.device.type == "cpu" else self.num_streams
        self.prefetch_factor = max(1, self.prefetch_factor) if self.num_workers > 0 else 0
        if self.fsdp:
            warnings.warn("The Trainer::evaluate method currently hangs with FSDP modules.")
        if self.num_streams > 0:
            warnings.warn("Using multiple data streams leads to substantially larger memory usage than zero streams.")

    @property
    def model_config(self) -> dict[str, Any]:
        cls: Optional[str] = None
        try:
            d = json.loads(self.model_config_str)
        except Exception as err:
            cls = err.__class__.__name__
        else:
            if not isinstance(d, dict):
                cls = "TypeError"
        if cls is not None:
            raise RuntimeError(f"A {cls} occurred while parsing model_config_str: `{self.model_config_str!r}`")
        assert isinstance(d, dict)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls.from_dict(vars(namespace))


def create_argument_parser_from_dataclass(*objs: type) -> ArgumentParser:
    """
    Creates an ArgumentParser from dataclass(es) with unique field names.
    """

    alltypes: dict[str, Any] = {}
    allfields: list[Field[Any]] = []
    allnames: set[str] = set()
    for obj in objs:
        fields_ = fields(obj)
        names = [f.name for f in fields_]
        if any(n in allnames for n in names):
            raise ValueError(f"Duplicate field names found: {names} in {allnames}.")
        allnames.update(names)
        allfields.extend(list(fields_))
        # NOTE: this does not correctly extract the types from foreign modules; they are just strings.
        hints = get_type_hints(obj, globalns=vars(sys.modules[obj.__module__]), include_extras=True)
        alltypes.update(hints)

    def _unwrap_optional(t: Any) -> tuple[Any, bool]:
        if get_origin(t) is Union:
            args = [a for a in get_args(t) if a is not type(None)]
            if len(args) == 1:
                return args[0], True
        return t, False

    def _maybe_cast_str(f: Callable[[str], Any], x: str) -> Any:
        if x.lower() == "none":
            return None
        return f(x)

    parser = ArgumentParser()
    for f in allfields:
        # print(f"Adding argument {f.name} of type {f.type} {type(f.type)=} with default {f.default}.")
        argname = f"--{f.name}"
        if (isinstance(f.type, GenericAlias) and isinstance(f.type.__origin__, type) and issubclass(f.type.__origin__, Sequence)):
            # This is the branch for multiple-value arguments. Its rather limited right now.
            type_ = f.type.__args__[0]
            choices = None
            if isinstance(f.type.__args__[0], type) and issubclass(f.type.__args__[0], Enum):
                if f.type.__args__[0] is lief.PE.Section.CHARACTERISTICS:
                    choices = [c for c in CHARACTERISTICS]
                    type_ = any_to_section_characteristic
                else:
                    choices = list(f.type.__args__[0])
            parser.add_argument(argname, type=type_, nargs="*", choices=choices, default=f.default)
        elif f.type is bool:
            parser.add_argument(argname, type=str_to_bool, default=f.default)
        elif f.type == Optional[bool]:
            parser.add_argument(argname, type=partial(_maybe_cast_str, str_to_bool), default=f.default)
        elif f.type == Optional[int]:
            parser.add_argument(argname, type=partial(_maybe_cast_str, int), default=f.default)
        elif f.type == Optional[float]:
            parser.add_argument(argname, type=partial(_maybe_cast_str, float), default=f.default)
        elif f.type == Optional[str]:
            parser.add_argument(argname, type=partial(_maybe_cast_str, str), default=f.default)
        elif isinstance(f.type, type) and issubclass(f.type, Enum):
            parser.add_argument(argname, type=f.type, choices=list(f.type), default=f.default)
        elif isinstance(f.type, type):
            parser.add_argument(argname, type=f.type, default=f.default)
        elif isinstance(f.type, str):
            type_, optional_ = _unwrap_optional(alltypes[f.name])
            if type_ is bool:
                type_ = str_to_bool
            elif type_ is int:
                type_ = int
            elif type_ is float:
                type_ = float
            elif type_ is str:
                type_ = str
            if optional_:
                type_ = partial(_maybe_cast_str, type_)
            parser.add_argument(argname, type=type_, default=f.default)
        else:
            raise ValueError(f"Cannot determine type of field {f}.")

    return parser


class _FlatDataclassWrapper:
    """
    Not intended to be used directly. Use `flatten_dataclasses` instead.
    """

    def __init__(self, *objs: type) -> None:
        # Avoid recursion in __setattr__.
        object.__setattr__(self, "_objs", objs)
        # Ensure all field names are unique.
        allnames: set[str] = set()
        for obj in objs:
            names = [f.name for f in fields(obj)]
            if any(n in allnames for n in names):
                raise ValueError(f"Duplicate field names found: {names} in {allnames}.")
            allnames.update(names)

    def __getattribute__(self, name: str) -> Any:
        # Internals
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        # Dataclasses
        objs = object.__getattribute__(self, "_objs")
        for obj in objs:
            if hasattr(obj, name):
                return getattr(obj, name)
        # Other
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Could not find {name!r} in any wrapped dataclass.")

    def __setattr__(self, name: str, value: Any) -> None:
        # Internals
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # Dataclasses
        for obj in object.__getattribute__(self, "_objs"):
            if hasattr(obj, name):
                setattr(obj, name, value)
                return
        # Other
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        # Internals
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        # Dataclasses
        for obj in object.__getattribute__(self, "_objs"):
            if hasattr(obj, name):
                delattr(obj, name)
                return
        # Other
        object.__delattr__(self, name)

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        for obj in object.__getattribute__(self, "_objs"):
            names.update(f.name for f in fields(obj))
        return sorted(names)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        parts = []
        parts.append(f"{self.__class__.__name__}(")
        for obj in object.__getattribute__(self, "_objs"):
            parts.append(f"  {obj.__class__.__name__}(")
            for f in fields(obj):
                parts.append(f"    {f.name}={getattr(obj, f.name)!r},")
            parts.append("  ),")
        parts.append(")")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for obj in object.__getattribute__(self, "_objs"):
            d.update({f.name: getattr(obj, f.name) for f in fields(obj)})
        return d


class _MTArgs(MainArgs, TrainerArgs):  # type: ignore[misc]
    """
    This has gotten so stupid, but I just don't care. Any way, don't create one of these. Ever.
    """

    def __init__(self, *args: Any, **kwds: Any) -> None:
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        return {}


def flatten_dataclasses(*objs: object) -> _FlatDataclassWrapper:
    """
    Provides a mypy compatible flattened view over multiple dataclasses with unique field names.
    """
    bases: list[type] = [_FlatDataclassWrapper]
    for obj in objs:
        if not hasattr(obj, "__dataclass_fields__"):
            raise TypeError(f"{obj} is not a dataclass.")
        bases.append(obj.__class__)
    Args = type("Args", tuple(bases), {})
    args: _FlatDataclassWrapper = Args(*objs)
    return args
