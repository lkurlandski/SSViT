"""
Aid and abet the main module.
"""

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import Field
from enum import Enum
import os
import sys
from typing import Any
from typing import Literal
from typing import Optional
from typing import Self
from typing import Union
from typing import get_type_hints
from typing import get_args
from typing import get_origin

import torch

from src.binanal import HierarchicalLevel
from src.trainer import TrainerArgs
from src.utils import str_to_bool


class Architecture(Enum):
    MALCONV = "malconv"
    VIT     = "vit"


class ModelSize(Enum):
    SM = "sm"
    MD = "md"
    LG = "lg"


# TODO: figure out how to elegantly combine different dataclasses into a new class.
# TODO: write an ArgumentParser that takes a dataclass and generates arguments.
@dataclass
class MainArgs:
    arch: Architecture = Architecture.MALCONV
    size: ModelSize = ModelSize.SM
    seed: int = 0
    do_parser: bool = False
    do_entropy: bool = False
    do_characteristics: bool = False
    level: HierarchicalLevel = HierarchicalLevel.NONE
    tr_num_samples: Optional[int] = None
    vl_num_samples: Optional[int] = None
    max_length: Optional[int] = None
    num_streams: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 1
    tr_batch_size: int = 1
    vl_batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: torch.device = torch.device("cpu")
    ddp: bool = False
    fsdp: bool = False
    fsdp_offload: bool = True
    tf32: bool = False

    def __post_init__(self) -> None:
        if self.tr_batch_size == 1 or self.vl_batch_size == 1:
            raise NotImplementedError("Batch size of 1 is not supported right now. See https://docs.pytorch.org/docs/stable/data#working-with-collate-fn.")
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

    parser = ArgumentParser()
    for f in allfields:
        # print(f"Adding argument {f.name} of type {f.type} {type(f.type)=} with default {f.default}.")
        argname = f"--{f.name}"
        if f.type == bool:
            parser.add_argument(argname, type=str_to_bool, default=f.default)
        elif f.type == Optional[bool]:
            parser.add_argument(argname, type=lambda x: None if x.lower() == "none" else str_to_bool(x), default=f.default)
        elif f.type == Optional[int]:
            parser.add_argument(argname, type=lambda x: None if x.lower() == "none" else int(x), default=f.default)
        elif f.type == Optional[float]:
            parser.add_argument(argname, type=lambda x: None if x.lower() == "none" else float(x), default=f.default)
        elif f.type == Optional[str]:
            parser.add_argument(argname, type=lambda x: None if x.lower() == "none" else str(x), default=f.default)
        elif isinstance(f.type, type) and issubclass(f.type, Enum):
            parser.add_argument(argname, type=f.type, choices=list(f.type), default=f.default)
        elif isinstance(f.type, type):
            parser.add_argument(argname, type=f.type, default=f.default)
        elif isinstance(f.type, str):
            # FIXME: this will not handle the special behaviors above!
            type_, _ = _unwrap_optional(alltypes[f.name])
            if type_ == bool:
                type_ = str_to_bool
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
