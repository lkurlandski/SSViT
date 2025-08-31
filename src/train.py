"""
Train and validate models.
"""

from argparse import ArgumentParser
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import Field
from enum import Enum
import math
import os
from pathlib import Path
from pprint import pprint
import sys
import time
from typing import Any
from typing import Optional
from typing import Self
from typing import Union
from typing import get_type_hints
from typing import get_args
from typing import get_origin
import warnings

import lief
import numpy as np
import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import OffloadPolicy
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures import get_model_input_lengths
from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import MalConv
from src.architectures import ViT
from src.architectures import Classifier
from src.architectures import PatchEncoder
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier
from src.binanal import HierarchicalLevel
from src.binanal import CharacteristicGuider
from src.binanal import LEVEL_STRUCTURE_MAP
from src.binanal import HierarchicalStructure
from src.data import BinaryDataset
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.data import CUDAPrefetcher
from src.data import Preprocessor
from src.data import GroupedLengthBatchSampler
from src.trainer import Trainer
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.utils import seed_everything
from src.utils import get_optimal_num_worker_threads
from src.utils import str_to_bool


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


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
            type_, _ = _unwrap_optional(alltypes[f.name])
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


def get_model(arch: Architecture, size: ModelSize, do_characteristics: bool, level: HierarchicalLevel) -> Classifier | HierarchicalMalConvClassifier | HierarchicalViTClassifier:

    f = None
    if size == ModelSize.SM:
        f = 1
    if size == ModelSize.MD:
        f = 2
    if size == ModelSize.LG:
        f = 4
    if f is None:
        raise ValueError(f"{size}")

    HierarchicalStructureCls: type[HierarchicalStructure] = LEVEL_STRUCTURE_MAP[level]
    num_structures = len(HierarchicalStructureCls)
    if num_structures <= 0:
        raise RuntimeError(f"{level} yielded no structures.")

    # Embedding
    padding_idx    = 0
    num_embeddings = 256 + 8
    embedding_dim  = 4 * f
    # FiLM
    guide_dim      = len(CharacteristicGuider.CHARACTERISTICS)
    guide_hidden   = 4 * f
    # Patcher
    num_patches    = 256
    patch_size     = None
    # MalConv
    mcnv_channels  = 128
    mcnv_kernel    = 512
    mcnv_stride    = 512
    # ViT
    vit_d_model    = 128 * f
    vit_nhead      = 2 * f
    vit_feedfrwd   = 4 * vit_d_model
    vit_layers     = 2 * f
    # Head
    num_classes    = 2
    clf_hidden     = 64 * f
    clf_layers     = 2
    clf_input_size = -1

    embedding = [Embedding(num_embeddings, embedding_dim, padding_idx) for _ in range(num_structures)]

    if do_characteristics:
        filmer = [FiLM(guide_dim, embedding_dim, guide_hidden) for _ in range(num_structures)]
    else:
        filmer = [FiLMNoP(guide_dim, embedding_dim, guide_hidden) for _ in range(num_structures)]

    patcher: list[Optional[PatchEncoder]]
    backbone: list[MalConv | ViT]
    if arch == Architecture.MALCONV:
        patcher = [None for _ in range(num_structures)]
        backbone = [MalConv(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride) for _ in range(num_structures)]
        clf_input_size = mcnv_channels
    elif arch == Architecture.VIT:
        patcher = [PatchEncoder(embedding_dim, vit_d_model, num_patches=num_patches, patch_size=patch_size) for _ in range(num_structures)]
        backbone = [ViT(vit_d_model, vit_d_model, vit_nhead, vit_feedfrwd, num_layers=vit_layers)]  # Only one ViT backbone
        clf_input_size = vit_d_model
    else:
        raise NotImplementedError(f"{arch}")

    head = ClassifificationHead(clf_input_size, num_classes, clf_hidden, clf_layers)

    if num_structures == 1:
        return Classifier(embedding[0], filmer[0], patcher[0], backbone[0], head)

    if arch == Architecture.MALCONV:
        return HierarchicalMalConvClassifier(embedding, filmer, backbone, head)

    if arch == Architecture.VIT:
        return HierarchicalViTClassifier(embedding, filmer, patcher, backbone[0], head)  # type: ignore[arg-type]  # Only one ViT backbone

    raise ValueError(f"Invalid combination of {arch=} and {level=}.")


def get_materials(tr_num_samples: Optional[int] = None, vl_num_samples: Optional[int] = None) -> tuple[list[Path], list[Path], list[int], list[int]]:
    # TODO: define a temporal (not random) train/test split.
    benfiles = list(filter(lambda f: f.is_file(), Path("./data/ass").rglob("*")))
    benlabels = [0] * len(benfiles)
    malfiles = list(filter(lambda f: f.is_file(), Path("./data/sor").rglob("*")))
    mallabels = [1] * len(malfiles)
    files = benfiles + malfiles
    labels = benlabels + mallabels
    print(f"{len(benfiles)=}")
    print(f"{len(malfiles)=}")

    idx = np.arange(len(files))
    tr_idx = np.random.choice(idx, size=int(0.8 * len(files)), replace=False)
    vl_idx = np.setdiff1d(idx, tr_idx)  # type: ignore[no-untyped-call]
    vl_idx = np.random.permutation(vl_idx)

    tr_files  = [files[i] for i in tr_idx]
    vl_files  = [files[i] for i in vl_idx]
    tr_labels = [labels[i] for i in tr_idx]
    vl_labels = [labels[i] for i in vl_idx]

    if tr_num_samples is not None:
        if tr_num_samples > len(tr_files):
            warnings.warn(f"Requested {tr_num_samples} training samples, but only {len(tr_files)} are available.")
            tr_num_samples = len(tr_files)
        tr_files = tr_files[0:tr_num_samples]
        tr_labels = tr_labels[0:tr_num_samples]

    if vl_num_samples is not None:
        if vl_num_samples > len(vl_files):
            warnings.warn(f"Requested {vl_num_samples} validation samples, but only {len(vl_files)} are available.")
            vl_num_samples = len(vl_files)
        vl_files = vl_files[0:vl_num_samples]
        vl_labels = vl_labels[0:vl_num_samples]

    print(f"{len(tr_files)=} {Counter(tr_labels)=}")
    print(f"{len(vl_files)=} {Counter(vl_labels)=}")

    return tr_files, vl_files, tr_labels, vl_labels


def count_parameters(model: Module, requires_grad: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


# FIXME: account for multiple GPUs.
def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    lief.logging.set_level(lief.logging.LEVEL.OFF)
    org_num_threads = torch.get_num_threads()
    new_num_threads = get_optimal_num_worker_threads(info.num_workers)
    torch.set_num_threads(new_num_threads)
    print(f"Worker {worker_id} of {info.num_workers} using {org_num_threads} --> {torch.get_num_threads()} threads.")


def main() -> None:

    lief.logging.set_level(lief.logging.LEVEL.OFF)

    parser = create_argument_parser_from_dataclass(MainArgs, TrainerArgs)
    namespace = parser.parse_args()
    margs = MainArgs.from_namespace(namespace)
    targs = TrainerArgs.from_namespace(namespace)
    args: _MTArgs = flatten_dataclasses(margs, targs)  # type: ignore[assignment]
    print(f"{args=}")

    seed_everything(args.seed)

    if args.ddp or args.fsdp:
        # A CPU (GLOO) backend is needed to support certain features with CPU offloading.
        backend = "cpu:gloo,cuda:nccl" if (args.fsdp and args.fsdp_offload) else "cuda:nccl"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank())
        args.device = torch.device(local_rank())
        print(f"Distrubted worker {dist.get_rank()} of {dist.get_world_size()} with local rank {local_rank()}.")

    if dist.get_rank() > 0:
        args.disable_tqdm = True

    preprocessor = Preprocessor(
        args.do_parser,
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )

    # Split the files between distributed workers.
    tr_files, vl_files, tr_labels, vl_labels = get_materials(args.tr_num_samples, args.vl_num_samples)
    if dist.get_world_size() > 1:
        tr_files = tr_files[dist.get_rank()::dist.get_world_size()]
        vl_files = vl_files[dist.get_rank()::dist.get_world_size()]
        tr_labels = tr_labels[dist.get_rank()::dist.get_world_size()]
        vl_labels = vl_labels[dist.get_rank()::dist.get_world_size()]

    tr_dataset = BinaryDataset(tr_files, tr_labels, preprocessor)
    vl_dataset = BinaryDataset(vl_files, vl_labels, preprocessor)

    model = get_model(args.arch, args.size, args.do_characteristics, args.level)
    model = model.to("cpu")
    print(f"{model=}")
    print(f"num_parameters={count_parameters(model, requires_grad=True)}")

    min_length = math.ceil(get_model_input_lengths(model)[0] / 8) * 8
    min_lengths = getattr(model, "min_lengths", [min_length])
    min_lengths = [max(m, min_length) for m in min_lengths]
    print(f"{min_length=}")
    print(f"{min_lengths=}")

    if args.ddp:
        model = model.to(args.device)
        model = DistributedDataParallel(model, find_unused_parameters=True, static_graph=True)
    elif args.fsdp:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        mp_policy = MixedPrecisionPolicy(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, torch.float32)
        offload_policy = CPUOffloadPolicy() if args.fsdp_offload else OffloadPolicy()
        model.fully_shard(mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        # model = model.to(args.device)
    else:
        model = model.to(args.device)

    collate_fn: CollateFn | CollateFnHierarchical
    if args.level == HierarchicalLevel.NONE:
        collate_fn = CollateFn(pin_memory=False, bitpack=False, min_length=min_length)
    else:
        collate_fn = CollateFnHierarchical(pin_memory=False, bitpack=False, num_structures=len(LEVEL_STRUCTURE_MAP[args.level]), min_lengths=min_lengths)
    print(f"{collate_fn=}")

    def get_loader(dataset: Dataset, sampler: Sampler) -> DataLoader | CUDAPrefetcher:

        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory and args.device.type == "cuda",
            worker_init_fn=worker_init_fn,
            multiprocessing_context=None if args.num_workers == 0 else "forkserver",
            prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
            persistent_workers=None if args.num_workers == 0 else True,
        )
        print(f"dataloader=DataLoader(pin_memory={dataloader.pin_memory}, num_workers={dataloader.num_workers}, prefetch_factor={dataloader.prefetch_factor})")
        dataloader = CUDAPrefetcher(dataloader, args.device, args.num_streams)
        if dataloader.loader.persistent_workers:
            dataloader.warmup(0)
            time.sleep(4)  # Wait for all workers to be ready.
        return dataloader

    tr_sampler = GroupedLengthBatchSampler.from_lengths(args.tr_batch_size, list(map(os.path.getsize, tr_dataset.files)), first=True, shuffle=True)
    vl_sampler = GroupedLengthBatchSampler.from_lengths(args.vl_batch_size, list(map(os.path.getsize, vl_dataset.files)), first=True, shuffle=False)
    tr_loader = get_loader(tr_dataset, tr_sampler)
    vl_loader = get_loader(vl_dataset, vl_sampler)
    print(f"{tr_loader=}")
    print(f"{vl_loader=}")

    loss_fn = CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler: Optional[LRScheduler] = None

    stopper: Optional[EarlyStopper] = None

    trainer = Trainer(TrainerArgs.from_dict(args.to_dict()), model, tr_loader, vl_loader, loss_fn, optimizer, scheduler, stopper, args.device)
    print(f"{trainer=}")

    trainer = trainer()

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
