"""
Train and validate models.
"""

from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
import math
import os
from pathlib import Path
import sys
from typing import cast
from typing import Any
from typing import Callable
from typing import Optional
from typing import Protocol
from typing import TypeVar
import warnings

import lief
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch import distributed as dist
from torch.utils.data import IterableDataset
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import OffloadPolicy
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
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
from src.data import IterableSimpleDBDataset
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.data import CUDAPrefetcher
from src.data import StreamlessCUDAPrefetcher
from src.data import Name
from src.data import Preprocessor
from src.data import FSample
from src.data import FSamples
from src.data import HSamples
from src.data import FOrHSamples
from src.helpers import create_argument_parser_from_dataclass
from src.helpers import flatten_dataclasses
from src.helpers import _MTArgs
from src.helpers import Architecture
from src.helpers import ModelSize
from src.helpers import MainArgs
from src.simpledb import SimpleDB
from src.simpledb import MetadataDB
from src.trainer import Trainer
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.trainer import local_rank
from src.trainer import rank
from src.trainer import local_world_size
from src.trainer import world_size
from src.trainer import mp_dtype
from src.trainer import Batch
from src.utils import num_sort_files
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
            backbone_ = MalConvCls(embedding_dim, mcnv_channels, mcnv_kernel, mcnv_stride)
            return MalConvClassifier(embedding, filmer, backbone_, head)
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


class SupportsFullyShard(Protocol):

    def fully_shard(self, *args: Any, **kwds: Any) -> None:
        ...

M = TypeVar("M", bound=Module)

def wrap_model_base(model: M, device: torch.device) -> M:
    model = model.to(device)
    return model

def wrap_model_ddp(model: M, device: torch.device, static_graph: bool) -> DistributedDataParallel:
    model = model.to(device)
    model = DistributedDataParallel(model, static_graph=static_graph)
    return model

def wrap_model_fsdp(model: M, mpdtype: torch.dtype, fsdp_offload: bool) -> M:
    mesh = init_device_mesh("cuda", (world_size(),))
    mp_policy = MixedPrecisionPolicy(mpdtype)
    offload_policy = CPUOffloadPolicy() if fsdp_offload else OffloadPolicy()
    cast(SupportsFullyShard, model).fully_shard(mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    return model


def get_collate_fn(level: HierarchicalLevel, min_lengths: list[int]) -> CollateFn | CollateFnHierarchical:
    if level == HierarchicalLevel.NONE:
        return CollateFn(False, False, min_lengths[0])
    return CollateFnHierarchical(False, False, len(LEVEL_STRUCTURE_MAP[level]), min_lengths)


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        raise RuntimeError("worker_init_fn called outside of worker process.")
    lief.logging.set_level(lief.logging.LEVEL.OFF)
    org_num_threads = torch.get_num_threads()
    new_num_threads = get_optimal_num_worker_threads(info.num_workers, ngpu=local_world_size())
    torch.set_num_threads(new_num_threads)
    print(f"Worker {worker_id} of {info.num_workers} using {org_num_threads} --> {torch.get_num_threads()} threads.")


def get_loader(
    dataset: Dataset[Any],
    batch_size: Optional[int],
    shuffle: Optional[bool],
    sampler: Optional[Sampler[Any]],
    batch_sampler: Optional[Sampler[Any]],
    num_workers: int,
    collate_fn: Callable[[Sequence[FSample]], FSamples | HSamples],
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader[FOrHSamples]:
    """
    Return a DataLoader with proper settings for multiprocessing.

    Note that providing `sampler` or `batch_sampler` will automatically
    override `shuffle`, `batch_size`, and `drop_last`. Similarly, providing
    and IterableDataset will override `shuffle` and `sampler`.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size if batch_sampler is None else None,
        shuffle=shuffle if sampler is None and batch_sampler is None and not isinstance(dataset, IterableDataset) else None,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        multiprocessing_context=None if num_workers == 0 else "forkserver",
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=False if num_workers == 0 else True,
    )


def get_streamer(loader: DataLoader[Any], device: torch.device, num_streams: int) -> DataLoader[FOrHSamples] | StreamlessCUDAPrefetcher | CUDAPrefetcher:
    """If multiple streams in use, wrap a DataLoader with a CUDAPrefetcher and warmup its workers."""
    if num_streams < 0:
        raise ValueError(f"`num_streams` must be non-negative, got {num_streams}.")
    if num_streams == 0:
        return loader
    if num_streams == 1:
        return StreamlessCUDAPrefetcher(loader, device)
    streamer = CUDAPrefetcher(loader, device, num_streams)
    if loader.persistent_workers:
        streamer.warmup(0)
    return streamer


def get_padbatch(level: HierarchicalLevel, do_parse: bool,do_entropy: bool, do_characteristics: bool, min_lengths: list[int], batch_size: int) -> FSamples | HSamples:
    """Return a short batch of purely padding samples."""
    file = ["./" + "0" * 64] * batch_size
    name = [Name("0" * 64)] * batch_size
    label = torch.zeros(batch_size, dtype=torch.int64)

    def get_inputs(length: int) -> torch.Tensor:
        return torch.zeros((batch_size, length), dtype=torch.int32)

    def get_guides(length: int) -> SemanticGuides:
        parse = torch.zeros((batch_size, length), dtype=torch.bool) if do_parse else None
        entropy = torch.zeros((batch_size, length), dtype=torch.float32) if do_entropy else None
        characteristics = torch.zeros((batch_size, length, len(CharacteristicGuider.CHARACTERISTICS)), dtype=torch.bool) if do_characteristics else None
        return SemanticGuides(parse, entropy, characteristics).decompress()

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
        return FSamples(file, name, label, inputs, guides, structure)

    inputs_ = [get_inputs(l) for l in min_lengths]
    guides_ = [get_guides(l) for l in min_lengths]
    return HSamples(file, name, label, inputs_, guides_, structure)


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

    wrap_model: Callable[[M], M | DistributedDataParallel]
    if args.ddp:
        wrap_model = partial(wrap_model_ddp, device=args.device, static_graph=True)
    elif args.fsdp:
        wrap_model = partial(wrap_model_fsdp, mpdtype=mpdtype, fsdp_offload=args.fsdp_offload)
    else:
        wrap_model = partial(wrap_model_base, device=args.device)
    print(f"{wrap_model=}")

    preprocessor = Preprocessor(
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )
    print(f"{preprocessor=}")

    root = Path("./data")
    print(f"{root=}")

    tr_datadb = SimpleDB(root / "data" / "tr")
    ts_datadb = SimpleDB(root / "data" / "ts")
    print(f"{tr_datadb=}")
    print(f"{ts_datadb=}")

    tr_metadb = MetadataDB(root / "meta" / "tr")
    ts_metadb = MetadataDB(root / "meta" / "ts")
    print(f"{tr_metadb=}")
    print(f"{ts_metadb=}")

    def get_last_shard(datadb: SimpleDB, num_samples: Optional[int]) -> int:
        if num_samples is None:
            return datadb.num_shards - 1
        s = 0
        for i in range(datadb.num_shards):
            s += sum(datadb.num_samples(i))
            if s >= num_samples:
                return i
        raise RuntimeError(f"There are only {s} samples in the database, so {num_samples} is too large.")

    tr_last_shard = get_last_shard(tr_datadb, args.tr_num_samples)
    ts_last_shard = get_last_shard(ts_datadb, args.ts_num_samples)
    print(f"{tr_last_shard=}")
    print(f"{ts_last_shard=}")

    tr_shards = list(range(len(tr_datadb.files_data)))[0: tr_last_shard + 1]
    ts_shards = list(range(len(ts_datadb.files_data)))[0: ts_last_shard + 1]
    print(f"tr_shards=[0, ..., {tr_last_shard}]")
    print(f"ts_shards=[0, ..., {ts_last_shard}]")

    def get_shardwise_stats(datadb: SimpleDB, last_shard: int) -> pd.DataFrame:
        dfs = []
        for shard, (size_df, meta_df) in enumerate(zip(datadb.get_size_dfs(), datadb.get_meta_dfs())):
            s = size_df["size"].to_numpy(np.int64)
            l = meta_df["malware"].to_numpy(np.int64)
            d = np.unique(l, return_counts=True)[1]
            df = pd.DataFrame({
                "shard-idx": shard,
                "size-max": np.max(s),
                "size-min": np.min(s),
                "size-avg": np.mean(s),
                "ratio-malware": d[1] / (d[0] + d[1]),
            }, index=[0])
            dfs.append(df)
            if shard >= last_shard:
                break
        return pd.concat(dfs, ignore_index=True)

    tr_stats_df = get_shardwise_stats(tr_datadb, tr_last_shard)
    ts_stats_df = get_shardwise_stats(ts_datadb, ts_last_shard)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(f"tr_stats_df=\n{tr_stats_df}")
        print(f"ts_stats_df=\n{ts_stats_df}")

    # Split the shards between distributed workers.
    if world_size() > 1:
        tr_shards = tr_shards[rank()::world_size()]
        ts_shards = ts_shards[rank()::world_size()]

    tr_dataset = IterableSimpleDBDataset(tr_datadb, tr_metadb, preprocessor, tr_shards, shuffle=True)
    ts_dataset = IterableSimpleDBDataset(ts_datadb, ts_metadb, preprocessor, ts_shards, shuffle=False)
    print(f"{tr_dataset=}")
    print(f"{ts_dataset=}")

    collate_fn = get_collate_fn(args.level, min_lengths)
    print(f"{collate_fn=}")

    tr_loader = get_loader(tr_dataset, args.tr_batch_size, True,  None, None, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    ts_loader = get_loader(ts_dataset, args.ts_batch_size, False, None, None, args.num_workers, collate_fn, args.pin_memory, args.prefetch_factor)
    print(f"tr_loader=DataLoader(pin_memory={tr_loader.pin_memory}, num_workers={tr_loader.num_workers}, prefetch_factor={tr_loader.prefetch_factor})")
    print(f"ts_loader=DataLoader(pin_memory={ts_loader.pin_memory}, num_workers={ts_loader.num_workers}, prefetch_factor={ts_loader.prefetch_factor})")

    tr_streamer = get_streamer(tr_loader, args.device, args.num_streams)
    ts_streamer = get_streamer(ts_loader, args.device, args.num_streams)
    print(f"{tr_streamer=}")
    print(f"{ts_streamer=}")

    rargs = TrainerArgs.from_dict(args.to_dict())
    print(f"{rargs=}")

    loss_fn = CrossEntropyLoss()
    print(f"{loss_fn=}")

    optimizer_init: Callable[[Iterable[torch.nn.Parameter]], Optimizer] = lambda params: AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"{optimizer_init=}")

    scheduler_init: Callable[[Optimizer], LRScheduler] = lambda optim: LambdaLR(optim, lambda _: 1.0)
    print(f"{scheduler_init=}")

    padbatch = get_padbatch(args.level, args.do_parser, args.do_entropy, args.do_characteristics, min_lengths, args.vl_batch_size)
    print(f"{padbatch=}")

    stopper: EarlyStopper = EarlyStopper(patience=float("inf"))
    print(f"{stopper=}")

    checkpoint = None
    if args.resume and args.outdir.exists():
        checkpoints = num_sort_files(args.outdir.glob("checkpoint-*"), lstrip="checkpoint-", rstrip="")
        if len(checkpoints) > 0:
            checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {checkpoint}")

    if checkpoint:
        trainer = Trainer.from_checkpoint(
            checkpoint,
            model=model,
            wrap_model=wrap_model,
            tr_loader=tr_streamer,
            vl_loader=ts_streamer,
            loss_fn=loss_fn,
            optimizer_init=optimizer_init,
            scheduler_init=scheduler_init,
            device=args.device,
        )
    else:
        wmodel = wrap_model(model)
        optimizer = optimizer_init(wmodel.parameters())
        scheduler = scheduler_init(optimizer)
        trainer = Trainer(
            args=rargs,
            model=wmodel,
            tr_loader=tr_streamer,
            vl_loader=ts_streamer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            stopper=stopper,
            device=args.device,
            padbatch=padbatch,
        )
    print(f"{trainer=}")

    trainer = trainer()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            print(f"[rank {rank()}] Destroying process group")
            dist.destroy_process_group()
            print(f"[rank {rank()}] Process group destroyed")
