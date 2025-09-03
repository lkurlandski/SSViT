"""
Train and validate models.
"""

from collections.abc import Sequence
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable
from typing import Optional
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
from src.data import FSample
from src.data import FSamples
from src.data import HSamples
from src.helpers import create_argument_parser_from_dataclass
from src.helpers import flatten_dataclasses
from src.helpers import _MTArgs
from src.helpers import Architecture
from src.helpers import ModelSize
from src.helpers import MainArgs
from src.trainer import Trainer
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.trainer import local_rank
from src.trainer import rank
from src.trainer import local_world_size
from src.trainer import world_size
from src.trainer import mp_dtype
from src.utils import seed_everything
from src.utils import get_optimal_num_worker_threads
from src.utils import count_parameters


def get_model(
    arch: Architecture,
    size: ModelSize,
    do_characteristics: bool,
    level: HierarchicalLevel,
) -> Classifier | HierarchicalMalConvClassifier | HierarchicalViTClassifier:

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


def get_materials(
    tr_num_samples: Optional[int] = None,
    vl_num_samples: Optional[int] = None,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
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


def get_collate_fn(level: HierarchicalLevel, min_lengths: list[int]) -> CollateFn | CollateFnHierarchical:
    if level == HierarchicalLevel.NONE:
        return CollateFn(False, False, min_lengths[0])
    return CollateFnHierarchical(False, False, len(LEVEL_STRUCTURE_MAP[level]), min_lengths)


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    lief.logging.set_level(lief.logging.LEVEL.OFF)
    org_num_threads = torch.get_num_threads()
    new_num_threads = get_optimal_num_worker_threads(info.num_workers, ngpu=local_world_size())
    torch.set_num_threads(new_num_threads)
    print(f"Worker {worker_id} of {info.num_workers} using {org_num_threads} --> {torch.get_num_threads()} threads.")


def get_loader(
    dataset: Dataset,
    sampler: Sampler,
    num_workers: int,
    collate_fn: Callable[[Sequence[FSample]], FSamples | HSamples],
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        multiprocessing_context=None if num_workers == 0 else "forkserver",
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=None if num_workers == 0 else True,
    )


def get_streamer(loader: DataLoader, device: torch.device, num_streams: int) -> CUDAPrefetcher:
    streamer = CUDAPrefetcher(loader, device, num_streams)
    if loader.persistent_workers:
        streamer.warmup(0)
        time.sleep(4)
    return streamer


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
        print(f"Distrubted worker {rank()} of {world_size()} with local rank {local_rank()}.")

    if args.tf32:
        torch.set_float32_matmul_precision("medium")

    if rank() > 0:
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
    if world_size() > 1:
        tr_files = tr_files[rank()::world_size()]
        vl_files = vl_files[rank()::world_size()]
        tr_labels = tr_labels[rank()::world_size()]
        vl_labels = vl_labels[rank()::world_size()]

    tr_dataset = BinaryDataset(tr_files, tr_labels, preprocessor)
    vl_dataset = BinaryDataset(vl_files, vl_labels, preprocessor)

    model = get_model(args.arch, args.size, args.do_characteristics, args.level)
    model = model.to("cpu")
    print(f"{model=}")

    num_parameters = count_parameters(model, requires_grad=False)
    print(f"{num_parameters=}")

    min_length = math.ceil(get_model_input_lengths(model)[0] / 8) * 8
    min_lengths = getattr(model, "min_lengths", [min_length])
    min_lengths = [max(m, min_length) for m in min_lengths]
    print(f"{min_length=}")
    print(f"{min_lengths=}")

    if args.ddp:
        model = model.to(args.device)
        model = DistributedDataParallel(model, static_graph=True)
    elif args.fsdp:
        mesh = init_device_mesh("cuda", (world_size(),))
        mp_policy = MixedPrecisionPolicy(mp_dtype(args.mp16, args.device))
        offload_policy = CPUOffloadPolicy() if args.fsdp_offload else OffloadPolicy()
        model.fully_shard(mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    else:
        model = model.to(args.device)

    collate_fn = get_collate_fn(args.level, min_lengths)
    print(f"{collate_fn=}")

    tr_sampler = GroupedLengthBatchSampler.from_lengths(args.tr_batch_size, list(map(os.path.getsize, tr_dataset.files)), first=True, shuffle=True)
    vl_sampler = GroupedLengthBatchSampler.from_lengths(args.vl_batch_size, list(map(os.path.getsize, vl_dataset.files)), first=True, shuffle=False)
    print(f"{tr_sampler=}")
    print(f"{vl_sampler=}")

    tr_loader = get_loader(tr_dataset, tr_sampler, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    vl_loader = get_loader(tr_dataset, tr_sampler, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    print(f"tr_loader=DataLoader(pin_memory={tr_loader.pin_memory}, num_workers={tr_loader.num_workers}, prefetch_factor={tr_loader.prefetch_factor})")
    print(f"vl_loader=DataLoader(pin_memory={vl_loader.pin_memory}, num_workers={vl_loader.num_workers}, prefetch_factor={vl_loader.prefetch_factor})")

    tr_streamer = get_streamer(tr_loader, args.device, args.num_streams)
    vl_streamer = get_streamer(vl_loader, args.device, args.num_streams)
    print(f"{tr_streamer=}")
    print(f"{vl_streamer=}")

    loss_fn = CrossEntropyLoss()
    print(f"{loss_fn=}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"{optimizer=}")

    scheduler: Optional[LRScheduler] = None
    print(f"{scheduler=}")

    stopper: Optional[EarlyStopper] = None
    print(f"{stopper=}")

    trainer = Trainer(TrainerArgs.from_dict(args.to_dict()), model, tr_streamer, vl_streamer, loss_fn, optimizer, scheduler, stopper, args.device)
    print(f"{trainer=}")

    trainer = trainer()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
