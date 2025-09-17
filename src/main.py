"""
Train and validate models.
"""

from collections.abc import Sequence
from collections import Counter
from copy import deepcopy
from hashlib import md5
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
import warnings

import lief
import numpy as np
from numpy import typing as npt
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
from src.architectures import MalConvBase
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import ViT
from src.architectures import PatchEncoder
from src.architectures import Classifier
from src.architectures import MalConvClassifier
from src.architectures import ViTClassifier
from src.architectures import HierarchicalClassifier
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier
from src.binanal import HierarchicalLevel
from src.binanal import CharacteristicGuider
from src.binanal import LEVEL_STRUCTURE_MAP
from src.binanal import HierarchicalStructure
from src.binanal import CharacteristicGuider
from src.data import SemanticGuides
from src.data import StructureMaps
from src.data import BinaryDataset
from src.data import BaseSimpleDBDataset
from src.data import MappingSimpleDBDataset
from src.data import IterableSimpleDBDataset
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.data import CUDAPrefetcher
from src.data import Preprocessor
from src.data import GroupedLengthBatchSampler
from src.data import ShardAwareBatchSampler
from src.data import FSample
from src.data import FSamples
from src.data import HSamples
from src.helpers import create_argument_parser_from_dataclass
from src.helpers import flatten_dataclasses
from src.helpers import _MTArgs
from src.helpers import Architecture
from src.helpers import ModelSize
from src.helpers import DBType
from src.helpers import MainArgs
from src.simpledb import SimpleDB
from src.simpledb import RandomMappingSimpleDB
from src.simpledb import ChunkedMappingSimpleDB
from src.simpledb import IterableSimpleDB
from src.simpledb import SimpleDBSample
from src.simpledb import split_simple_db
from src.split import tr_vl_ts_split
from src.trainer import Trainer
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.trainer import local_rank
from src.trainer import rank
from src.trainer import local_world_size
from src.trainer import world_size
from src.trainer import mp_dtype
from src.trainer import init_metrics_group
from src.utils import seed_everything
from src.utils import get_optimal_num_worker_threads
from src.utils import count_parameters


def get_model(
    arch: Architecture,
    size: ModelSize,
    do_characteristics: bool,
    level: HierarchicalLevel,
) -> Classifier | HierarchicalClassifier:

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
    if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
        clf_input_size = mcnv_channels
    elif arch in (Architecture.VIT,):
        clf_input_size = vit_d_model
    else:
        raise NotImplementedError(f"{arch}")
    head = ClassifificationHead(clf_input_size, num_classes, clf_hidden, clf_layers)

    arch_to_malconv_backbone: dict[Architecture, type[MalConvBase]] = {
        Architecture.MCV: MalConv,
        Architecture.MC2: MalConvLowMem,
        Architecture.MCG: MalConvGCG,
    }

    FiLMCls = FiLM if do_characteristics else FiLMNoP

    if level == HierarchicalLevel.NONE:
        embedding = Embedding(num_embeddings, embedding_dim, padding_idx)
        filmer = FiLMCls(guide_dim, embedding_dim, guide_hidden)
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            MalConvCls = arch_to_malconv_backbone[arch]
            backbone = MalConvCls(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride)
            return MalConvClassifier(embedding, filmer, backbone, head)
        if arch in (Architecture.VIT,):
            patcher = PatchEncoder(embedding_dim, vit_d_model, num_patches=num_patches, patch_size=patch_size)
            backbone = ViT(vit_d_model, vit_d_model, vit_nhead, vit_feedfrwd, num_layers=vit_layers)
            return ViTClassifier(embedding, filmer, patcher, backbone, head)

    if level == HierarchicalLevel.COARSE:
        # TODO: configure hyperparameters.
        embeddings = [
            Embedding(num_embeddings, embedding_dim, padding_idx)
            for _ in range(num_structures)
        ]
        filmers = [
            FiLMCls(guide_dim, embedding_dim, guide_hidden)
            for _ in range(num_structures)
        ]
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            MalConvCls = arch_to_malconv_backbone[arch]
            backbones = [
                MalConvCls(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride)
                for _ in range(num_structures)
            ]
            return HierarchicalMalConvClassifier(embeddings, filmers, backbones, head)
        if arch in (Architecture.VIT,):
            patchers = [
                PatchEncoder(embedding_dim, vit_d_model, num_patches=num_patches, patch_size=patch_size)
                for _ in range(num_structures)
            ]
            backbone = ViT(vit_d_model, vit_d_model, vit_nhead, vit_feedfrwd, num_layers=vit_layers)
            return HierarchicalViTClassifier(embeddings, filmers, patchers, backbone, head)

    if level == HierarchicalLevel.MIDDLE:
        # TODO: configure hyperparameters.
        embeddings = [
            Embedding(num_embeddings, embedding_dim, padding_idx)
            for _ in range(num_structures)
        ]
        filmers = [
            FiLMCls(guide_dim, embedding_dim, guide_hidden)
            for _ in range(num_structures)
        ]
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            MalConvCls = arch_to_malconv_backbone[arch]
            backbones = [
                MalConvCls(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride)
                for _ in range(num_structures)
            ]
            return HierarchicalMalConvClassifier(embeddings, filmers, backbones, head)
        if arch in (Architecture.VIT,):
            patchers = [
                PatchEncoder(embedding_dim, vit_d_model, num_patches=num_patches, patch_size=patch_size)
                for _ in range(num_structures)
            ]
            backbone = ViT(vit_d_model, vit_d_model, vit_nhead, vit_feedfrwd, num_layers=vit_layers)
            return HierarchicalViTClassifier(embeddings, filmers, patchers, backbone, head)

    if level == HierarchicalLevel.FINE:
        # TODO: configure hyperparameters.
        embeddings = [
            Embedding(num_embeddings, embedding_dim, padding_idx)
            for _ in range(num_structures)
        ]
        filmers = [
            FiLMCls(guide_dim, embedding_dim, guide_hidden)
            for _ in range(num_structures)
        ]
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            MalConvCls = arch_to_malconv_backbone[arch]
            backbones = [
                MalConvCls(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride)
                for _ in range(num_structures)
            ]
            return HierarchicalMalConvClassifier(embeddings, filmers, backbones, head)
        if arch in (Architecture.VIT,):
            patchers = [
                PatchEncoder(embedding_dim, vit_d_model, num_patches=num_patches, patch_size=patch_size)
                for _ in range(num_structures)
            ]
            backbone = ViT(vit_d_model, vit_d_model, vit_nhead, vit_feedfrwd, num_layers=vit_layers)
            return HierarchicalViTClassifier(embeddings, filmers, patchers, backbone, head)

    raise NotImplementedError(f"{level} {arch}")


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
    if isinstance(info.dataset, BaseSimpleDBDataset):
        info.dataset.worker_open_and_register_finalizer()
    print(f"Worker {worker_id} of {info.num_workers} using {org_num_threads} --> {torch.get_num_threads()} threads.")


def get_loader(
    dataset: Dataset,
    sampler: Sampler,
    shuffle: Optional[bool],
    num_workers: int,
    collate_fn: Callable[[Sequence[FSample]], FSamples | HSamples],
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        multiprocessing_context=None if num_workers == 0 else "forkserver",
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=None if num_workers == 0 else True,
    )


def get_streamer(loader: DataLoader, device: torch.device, num_streams: int) -> CUDAPrefetcher:
    """Wrap a DataLoader with a CUDAPrefetcher and warmup its workers."""
    streamer = CUDAPrefetcher(loader, device, num_streams)
    if loader.persistent_workers:
        streamer.warmup(0)
        time.sleep(4)
    if loader.num_workers == 0 and isinstance(loader.dataset, BaseSimpleDBDataset):
        loader.dataset.worker_open_and_register_finalizer()
    return streamer


def get_padbatch(level: HierarchicalLevel, do_parse: bool,do_entropy: bool, do_characteristics: bool, min_lengths: list[int], batch_size: int) -> FSamples | HSamples:
    """Return a short batch of purely padding samples."""
    from src.data import Name
    file = ["./" + "0" * 64] * batch_size
    name = [Name("0" * 64)] * batch_size
    label = torch.zeros(batch_size, dtype=torch.int64)

    def get_inputs(length: int) -> torch.Tensor:
        return torch.zeros((batch_size, length), dtype=torch.int32)

    def get_guides(length: int) -> SemanticGuides:
        parse = torch.zeros((batch_size, length), dtype=torch.float32) if do_parse else None
        entropy = torch.zeros((batch_size, length), dtype=torch.float32) if do_entropy else None
        characteristics = torch.zeros((batch_size, length, CharacteristicGuider.CHARACTERISTICS), dtype=torch.float32) if do_characteristics else None
        return SemanticGuides(parse, entropy, characteristics)

    def get_structure() -> StructureMaps:
        index: list[list[tuple[int, int]]] = [[] for _ in LEVEL_STRUCTURE_MAP[level]]
        lexicon = {}
        for i, s in enumerate(LEVEL_STRUCTURE_MAP[level]):
            index[i] = [(0, min_lengths[i])]
            lexicon[i] = s
        index = [deepcopy(index) for _ in range(batch_size)]
        return StructureMaps(index, lexicon)

    structure = get_structure()

    if level == HierarchicalLevel.NONE:
        inputs = get_inputs(min_lengths[0])
        guides = get_guides(min_lengths[0])
        return FSamples(file, name, label, inputs, guides, structure)  # type: ignore[arg-type]

    inputs_ = [get_inputs(l) for l in min_lengths]
    guides_ = [get_guides(l) for l in min_lengths]
    return HSamples(file, name, label, inputs_, guides_, structure)  # type: ignore[arg-type]


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
        init_metrics_group()
        print(f"Distrubted worker {rank()} of {world_size()} with local rank {local_rank()}.")

    if args.tf32:
        torch.set_float32_matmul_precision("medium")

    if rank() > 0:
        args.disable_tqdm = True

    mpdtype = mp_dtype(args.mp16, args.device)
    print(f"{mpdtype=}")

    model = get_model(args.arch, args.size, args.do_characteristics, args.level).to("cpu")
    num_parameters = count_parameters(model, requires_grad=False)
    min_length = math.ceil(get_model_input_lengths(model)[0] / 8) * 8
    min_lengths = [max(m, min_length) for m in getattr(model, "min_lengths", [min_length])]
    print(f"{model=}")
    print(f"{num_parameters=}")
    print(f"{min_length=}")
    print(f"{min_lengths=}")
    if args.ddp:
        model = model.to(args.device)
        model = DistributedDataParallel(model, static_graph=True)
    elif args.fsdp:
        mesh = init_device_mesh("cuda", (world_size(),))
        mp_policy = MixedPrecisionPolicy(mpdtype)
        offload_policy = CPUOffloadPolicy() if args.fsdp_offload else OffloadPolicy()
        model.fully_shard(mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    else:
        model = model.to(args.device)

    preprocessor = Preprocessor(
        args.do_parser,
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )
    print(f"{preprocessor=}")

    tr_db_root = Path("./datadb_tr")
    vl_db_root = Path("./datadb_vl")
    ts_db_root = Path("./datadb_ts")
    print(f"{tr_db_root=}")
    print(f"{vl_db_root=}")
    print(f"{ts_db_root=}")

    tr_db: SimpleDB
    vl_db: SimpleDB
    ts_db: SimpleDB
    if args.db_type == DBType.ITR:
        tr_db = IterableSimpleDB(tr_db_root, pread_block_bytes=8 * 2**20, merge_slack_bytes=1 * 2**20)
        vl_db = IterableSimpleDB(vl_db_root, pread_block_bytes=8 * 2**20, merge_slack_bytes=1 * 2**20)
        ts_db = IterableSimpleDB(ts_db_root, pread_block_bytes=8 * 2**20, merge_slack_bytes=1 * 2**20)
    if args.db_type == DBType.RND:
        ulimit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        tr_db = RandomMappingSimpleDB(tr_db_root, num_open=ulimit // 2, backend="mmap")
        vl_db = RandomMappingSimpleDB(vl_db_root, num_open=ulimit // 8, backend="mmap")
        ts_db = RandomMappingSimpleDB(ts_db_root, num_open=ulimit // 8, backend="mmap")
    if args.db_type == DBType.CHK:
        tr_db = ChunkedMappingSimpleDB(tr_db_root)
        vl_db = ChunkedMappingSimpleDB(vl_db_root)
        ts_db = ChunkedMappingSimpleDB(ts_db_root)
    print(f"{tr_db=}")
    print(f"{vl_db=}")
    print(f"{ts_db=}")

    tr_max_shard = len(tr_db.files_data) - 1 if args.tr_num_samples is None else tr_db.meta_df["shard"].iloc[args.tr_num_samples]
    vl_max_shard = len(vl_db.files_data) - 1 if args.vl_num_samples is None else vl_db.meta_df["shard"].iloc[args.vl_num_samples]
    ts_max_shard = len(ts_db.files_data) - 1 if args.ts_num_samples is None else ts_db.meta_df["shard"].iloc[args.ts_num_samples]
    print(f"{tr_max_shard=}")
    print(f"{vl_max_shard=}")
    print(f"{ts_max_shard=}")

    tr_shards = list(range(len(tr_db.files_data)))[0: tr_max_shard + 1]
    vl_shards = list(range(len(vl_db.files_data)))[0: vl_max_shard + 1]
    ts_shards = list(range(len(ts_db.files_data)))[0: ts_max_shard + 1]
    print(f"tr_shards=[0, ..., {tr_max_shard}]")
    print(f"vl_shards=[0, ..., {vl_max_shard}]")
    print(f"ts_shards=[0, ..., {ts_max_shard}]")

    tr_idx = tr_db.meta_df[tr_db.meta_df["shard"] <= tr_max_shard]["idx"].to_numpy(np.int64)
    vl_idx = vl_db.meta_df[vl_db.meta_df["shard"] <= vl_max_shard]["idx"].to_numpy(np.int64)
    ts_idx = ts_db.meta_df[ts_db.meta_df["shard"] <= ts_max_shard]["idx"].to_numpy(np.int64)
    tr_labels = tr_db.meta_df["malware"].to_numpy(np.int64)[tr_idx]
    vl_labels = vl_db.meta_df["malware"].to_numpy(np.int64)[vl_idx]
    ts_labels = ts_db.meta_df["malware"].to_numpy(np.int64)[ts_idx]
    print(f"tr_idx ({len(tr_idx)}): distribution={np.unique(tr_labels, return_counts=True)}")  # type: ignore[no-untyped-call]
    print(f"vl_idx ({len(vl_idx)}): distribution={np.unique(vl_labels, return_counts=True)}")  # type: ignore[no-untyped-call]
    print(f"ts_idx ({len(ts_idx)}): distribution={np.unique(ts_labels, return_counts=True)}")  # type: ignore[no-untyped-call]

    # Split the shards or idx between distributed workers.
    if world_size() > 1:
        if args.db_type == DBType.ITR:
            tr_shards = tr_shards[rank()::world_size()]
            vl_shards = vl_shards[rank()::world_size()]
            ts_shards = ts_shards[rank()::world_size()]
        if args.db_type == DBType.RND:
            tr_idx = tr_idx[rank()::world_size()]
            vl_idx = vl_idx[rank()::world_size()]
            ts_idx = ts_idx[rank()::world_size()]
        if args.db_type == DBType.CHK:
            tr_idx = tr_idx[rank()::world_size()]
            vl_idx = vl_idx[rank()::world_size()]
            ts_idx = ts_idx[rank()::world_size()]

    if args.db_type == DBType.ITR:
        tr_dataset = IterableSimpleDBDataset(tr_db, preprocessor, shuffle=True,  shards=tr_shards)  # type: ignore[arg-type]
        vl_dataset = IterableSimpleDBDataset(vl_db, preprocessor, shuffle=False, shards=vl_shards)  # type: ignore[arg-type]
        ts_dataset = IterableSimpleDBDataset(ts_db, preprocessor, shuffle=False, shards=ts_shards)  # type: ignore[arg-type]
    if args.db_type == DBType.RND:
        tr_dataset = MappingSimpleDBDataset(tr_db, preprocessor)  # type: ignore[arg-type]
        vl_dataset = MappingSimpleDBDataset(vl_db, preprocessor)  # type: ignore[arg-type]
        ts_dataset = MappingSimpleDBDataset(ts_db, preprocessor)  # type: ignore[arg-type]
    if args.db_type == DBType.CHK:
        tr_dataset = MappingSimpleDBDataset(tr_db, preprocessor)  # type: ignore[arg-type]
        vl_dataset = MappingSimpleDBDataset(vl_db, preprocessor)  # type: ignore[arg-type]
        ts_dataset = MappingSimpleDBDataset(ts_db, preprocessor)  # type: ignore[arg-type]
    print(f"{tr_dataset=}")
    print(f"{vl_dataset=}")
    print(f"{ts_dataset=}")

    collate_fn = get_collate_fn(args.level, min_lengths)
    print(f"{collate_fn=}")

    if args.db_type == DBType.ITR:
        tr_sampler = None
        vl_sampler = None
        ts_sampler = None
    if args.db_type == DBType.RND:
        # TODO: design a sampler that only selects from the subset of idx, not the entire db; adjust the __len__ of the dataset accordingly.
        warnings.warn(f"Properly sample selection for {args.db_type} is not implemented yet.")
        tr_sampler = None
        vl_sampler = None
        ts_sampler = None
    if args.db_type == DBType.CHK:
        # TODO: design a sampler that only selects from the subset of idx, not the entire db; adjust the __len__ of the dataset accordingly.
        warnings.warn(f"Properly sample selection for {args.db_type} is not implemented yet.")
        tr_sampler = ShardAwareBatchSampler(args.tr_batch_size, tr_db.size_df["shard"], tr_db.size_df["size"], tr_db.size_df["offset"], shuffle=True, seed=args.seed)
        vl_sampler = ShardAwareBatchSampler(args.vl_batch_size, vl_db.size_df["shard"], vl_db.size_df["size"], vl_db.size_df["offset"], shuffle=False, seed=args.seed)
        ts_sampler = ShardAwareBatchSampler(args.ts_batch_size, ts_db.size_df["shard"], ts_db.size_df["size"], ts_db.size_df["offset"], shuffle=False, seed=args.seed)
    print(f"{tr_sampler=}")
    print(f"{vl_sampler=}")
    print(f"{ts_sampler=}")

    if args.db_type == DBType.ITR:
        tr_shuffle = None
        vl_shuffle = None
        ts_shuffle = None
    if args.db_type == DBType.RND:
        tr_shuffle = True
        vl_shuffle = False
        ts_shuffle = False
    if args.db_type == DBType.CHK:
        tr_shuffle = None
        vl_shuffle = None
        ts_shuffle = None
    print(f"{tr_shuffle=}")
    print(f"{vl_shuffle=}")
    print(f"{ts_shuffle=}")

    tr_loader = get_loader(tr_dataset, tr_sampler, tr_shuffle, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    vl_loader = get_loader(vl_dataset, vl_sampler, vl_shuffle, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    ts_loader = get_loader(ts_dataset, ts_sampler, ts_shuffle, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    print(f"tr_loader=DataLoader(pin_memory={tr_loader.pin_memory}, num_workers={tr_loader.num_workers}, prefetch_factor={tr_loader.prefetch_factor})")
    print(f"vl_loader=DataLoader(pin_memory={vl_loader.pin_memory}, num_workers={vl_loader.num_workers}, prefetch_factor={vl_loader.prefetch_factor})")
    print(f"ts_loader=DataLoader(pin_memory={ts_loader.pin_memory}, num_workers={ts_loader.num_workers}, prefetch_factor={ts_loader.prefetch_factor})")

    tr_streamer = get_streamer(tr_loader, args.device, args.num_streams)
    vl_streamer = get_streamer(vl_loader, args.device, args.num_streams)
    ts_streamer = get_streamer(ts_loader, args.device, args.num_streams)
    print(f"{tr_streamer=}")
    print(f"{vl_streamer=}")
    print(f"{ts_streamer=}")

    loss_fn = CrossEntropyLoss()
    print(f"{loss_fn=}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"{optimizer=}")

    scheduler: Optional[LRScheduler] = None
    print(f"{scheduler=}")

    stopper: Optional[EarlyStopper] = None
    print(f"{stopper=}")

    padbatch = get_padbatch(args.level, args.do_parser, args.do_entropy, args.do_characteristics, min_lengths, args.vl_batch_size)
    print(f"{padbatch=}")

    trainer = Trainer(TrainerArgs.from_dict(args.to_dict()), model, tr_streamer, vl_streamer, loss_fn, optimizer, scheduler, stopper, args.device, padbatch)
    print(f"{trainer=}")

    trainer()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            print(f"[rank {rank()}] Destroying process group")
            dist.destroy_process_group()
            print(f"[rank {rank()}] Process group destroyed")
