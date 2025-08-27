"""
Train and validate models.
"""

from argparse import ArgumentParser
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import sys
from typing import Any
from typing import Optional
from typing import Self
import warnings

import lief
import numpy as np
import torch
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss
from torch.nn import Module
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
from src.trainer import TrainerArgumentParser
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.utils import seed_everything
from src.utils import get_optimal_num_worker_threads
from src.utils import str_to_bool


class Architecture(Enum):
    MALCONV = "malconv"
    VIT     = "vit"


class ModelSize(Enum):
    SM = "sm"
    MD = "md"
    LG = "lg"


# TODO: figure out how to integrate this into the main loop properly (inheritence hacks didn't work).
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
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 1
    tr_batch_size: int = 1
    vl_batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        if self.tr_batch_size == 1 or self.vl_batch_size == 1:
            raise NotImplementedError("Batch size of 1 is not supported right now. See https://docs.pytorch.org/docs/stable/data#working-with-collate-fn.")

    @classmethod
    def from_namespace(cls, namespace: Namespace) -> Self:
        return cls(**{k: v for k, v in vars(namespace).items() if k in cls.__dataclass_fields__})


class MainArgumentParser(ArgumentParser):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--arch", type=Architecture, default=MainArgs.arch)
        self.add_argument("--size", type=ModelSize, default=MainArgs.size)
        self.add_argument("--seed", type=int, default=MainArgs.seed)
        self.add_argument("--do_parser", type=str_to_bool, default=MainArgs.do_parser)
        self.add_argument("--do_entropy", type=str_to_bool, default=MainArgs.do_entropy)
        self.add_argument("--do_characteristics", type=str_to_bool, default=MainArgs.do_characteristics)
        self.add_argument("--level", type=HierarchicalLevel, default=MainArgs.level)
        self.add_argument("--max_length", type=int, default=MainArgs.max_length)
        self.add_argument("--tr_num_samples", type=int, default=MainArgs.tr_num_samples)
        self.add_argument("--vl_num_samples", type=int, default=MainArgs.vl_num_samples)
        self.add_argument("--num_workers", type=int, default=MainArgs.num_workers)
        self.add_argument("--pin_memory", type=str_to_bool, default=MainArgs.pin_memory)
        self.add_argument("--prefetch_factor", type=int, default=MainArgs.prefetch_factor)
        self.add_argument("--tr_batch_size", type=int, default=MainArgs.tr_batch_size)
        self.add_argument("--vl_batch_size", type=int, default=MainArgs.vl_batch_size)
        self.add_argument("--learning_rate", type=float, default=MainArgs.learning_rate)
        self.add_argument("--weight_decay", type=float, default=MainArgs.weight_decay)


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


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    lief.logging.set_level(lief.logging.LEVEL.OFF)
    org_num_threads = torch.get_num_threads()
    new_num_threads = get_optimal_num_worker_threads(info.num_workers)
    torch.set_num_threads(new_num_threads)
    print(f"Worker {worker_id} of {info.num_workers} using {org_num_threads} --> {torch.get_num_threads()} threads.")


def main() -> None:

    lief.logging.set_level(lief.logging.LEVEL.OFF)

    Parser = type("Parser", (MainArgumentParser, TrainerArgumentParser), {})
    parser = Parser(description="Train and validate models.")
    args = parser.parse_args()

    if args.device.type != "cuda":
        raise NotImplementedError("Training on CPU is not supported right now because the CUDAPrefetcher is responsible for decompressing inputs.")

    seed_everything(args.seed)

    preprocessor = Preprocessor(
        args.do_parser,
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )

    tr_files, vl_files, tr_labels, vl_labels = get_materials(args.tr_num_samples, args.vl_num_samples)

    tr_dataset = BinaryDataset(tr_files, tr_labels, preprocessor)
    vl_dataset = BinaryDataset(vl_files, vl_labels, preprocessor)

    model = get_model(args.arch, args.size, args.do_characteristics, args.level)
    print(f"{model=}")
    print(f"num_parameters={count_parameters(model, requires_grad=True)}")

    min_length = math.ceil(get_model_input_lengths(model)[0] / 8) * 8
    min_lengths = getattr(model, "min_lengths", [min_length])
    min_lengths = [max(m, min_length) for m in min_lengths]
    print(f"{min_length=}")
    print(f"{min_lengths=}")

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
        print(f"dataloader=DataLoader(pin_memory={dataloader.pin_memory})")
        if args.device.type == "cuda":
            dataloader = CUDAPrefetcher(dataloader, args.device)
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

    trainer = Trainer(TrainerArgs.from_namespace(args), model, tr_loader, vl_loader, loss_fn, optimizer, scheduler, stopper)

    trainer = trainer()


if __name__ == "__main__":
    main()
