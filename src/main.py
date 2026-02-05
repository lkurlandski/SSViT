"""
Train and validate models.
"""

from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from inspect import signature
import math
import os
from pathlib import Path
from pprint import pformat
from pprint import pprint
import sys
import time
from typing import cast
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import TypedDict
from typing import TypeVar
import warnings

import lief
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch import nn
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
from torch.optim.optimizer import ParamsT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.profiler import schedule
from torch.profiler import tensorboard_trace_handler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures import get_model_input_lengths
from src.architectures import Identity
from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import MalConvBase
from src.architectures import MalConv
from src.architectures import MalConvLowMem
from src.architectures import MalConvGCG
from src.architectures import ViT
from src.architectures import PatchPositionalityEncoder
from src.architectures import PatchEncoderBase
from src.architectures import PatchEncoder
from src.architectures import ConvPatchEncoder
from src.architectures import HierarchicalConvPatchEncoder
from src.architectures import PatchEncoderLowMem
from src.architectures import PatchEncoderLowMemSwitchMoE
from src.architectures import DWCPatchEncoder
from src.architectures import Classifier
from src.architectures import MalConvClassifier
from src.architectures import ViTClassifier
from src.architectures import HierarchicalClassifier
from src.architectures import HierarchicalMalConvClassifier
from src.architectures import HierarchicalViTClassifier
from src.architectures import StructuralClassifier
from src.architectures import StructuralMalConvClassifier
from src.architectures import StructuralViTClassifier
from src.binanal import HierarchicalLevel
from src.binanal import LEVEL_STRUCTURE_MAP
from src.binanal import DIRECTORY_STRUCTURES
from src.binanal import HierarchicalStructure
from src.binanal import HierarchicalStructureNone
from src.binanal import HierarchicalStructureCoarse
from src.binanal import HierarchicalStructureMiddle
from src.binanal import HierarchicalStructureFine
from src.data import SemanticGuides
from src.data import StructureMaps
from src.data import Inputs
from src.data import IterableSimpleDBDataset
from src.data import CollateFn
from src.data import CollateFnHierarchical
from src.data import CollateFnStructural
from src.data import CUDAPrefetcher
from src.data import StreamlessCUDAPrefetcher
from src.data import Name
from src.data import Preprocessor
from src.data import FSample
from src.data import FSamples
from src.data import HSamples
from src.data import SSamples
from src.data import FOrHSamples
from src.data import PAD_TO_MULTIPLE_OF
from src.helpers import create_argument_parser_from_dataclass
from src.helpers import flatten_dataclasses
from src.helpers import _MTArgs
from src.helpers import Scheduler
from src.helpers import Design
from src.helpers import Architecture
from src.helpers import PatcherArchitecture
from src.helpers import PositionalEncodingArchitecture
from src.helpers import PatchPositionalEncodingArchitecture
from src.helpers import MainArgs
from src.simpledb import SimpleDB
from src.simpledb import MetadataDB
from src.simpledb import roundup
from src.trainer import Trainer
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper
from src.trainer import local_rank
from src.trainer import rank
from src.trainer import local_world_size
from src.trainer import world_size
from src.trainer import mp_dtype
from src.trainer import is_dist
from src.trainer import largest_possible_dataloader_length
from src.utils import num_sort_files
from src.utils import seed_everything
from src.utils import get_optimal_num_worker_threads
from src.utils import count_parameters


STRICT_RAFF_MATCH = os.environ.get("STRICT_RAFF_MATCH", "0") == "1"
if STRICT_RAFF_MATCH:
    warnings.warn("STRICT_RAFF_MATCH is enabled. MalConv models will match the original Raff et al. implementation more closely.")


STRUCTURES_AS_GUIDES = os.environ.get("STRUCTURES_AS_GUIDES", "0") == "1"
if STRUCTURES_AS_GUIDES:
    warnings.warn("STRUCTURES_AS_GUIDES is enabled. Hierarchical structures will be used as semantic guides.")


MAIN_RUN_PROFILE = os.environ.get("MAIN_RUN_PROFILE", "0") == "1"
if MAIN_RUN_PROFILE:
    warnings.warn("MAIN_RUN_PROFILE is enabled. Certain command line arguments will be overridden.")


def get_model(
    design: Design,
    arch: Architecture,
    parch: PatcherArchitecture = PatcherArchitecture.BAS,
    posenc: PositionalEncodingArchitecture = PositionalEncodingArchitecture.FIXED,
    patchposenc: PatchPositionalEncodingArchitecture = PatchPositionalEncodingArchitecture.NONE,
    num_guides: int = 0,
    structures: Sequence[HierarchicalStructure] = tuple(),
    max_structures: Optional[int] = None,
    max_length: Optional[int] = None,
    share_embeddings: bool = False,
    share_patchers: bool = False,
    batch_size: Optional[int] = None,
    enable_checkpoint: bool = False,
    enable_compile: bool = False,
    do_ddp_keepalive: bool = False,
    *,
    embedding_dim: int = 8,
    film_hidden_size: int = 16,
    patcher_channels: int = 64,
    patcher_kernel_size: int = 64,
    patcher_stride: int = 64,
    patcher_pooling: Literal["max", "avg", "atn"] = "max",
    patcher_depth: int = 4,
    num_patches: Optional[int] = 256,
    patch_size: Optional[int] = 4096,
    patchposencoder_hidden_size: int = 64,
    vit_d_model: int = 512,
    vit_nhead: int = 8,
    vit_dim_feedforward: int = 2048,
    vit_num_layers: int = 8,
    clf_hidden_size: int = 256,
    clf_num_layers: int = 2,
    moe_num_experts: int = 1,
    moe_probe_dim: int = 16,
    moe_probe_kernel_size: int = 256,
    moe_probe_stride: int = 256,
    moe_router_hidden: int = 256,
    moe_router_temperature: float = 1.0,
    moe_router_noise_std: float = 0.0,
    moe_router_top_k: int = 1,
    patcher_batch_sizes: Optional[Sequence[int]] = None,
    backbone_seq_lengths: Optional[Sequence[int]] = None,
    backbone_batch_sizes: Optional[Sequence[int]] = None,
) -> Classifier | HierarchicalClassifier | StructuralClassifier:
    """
    Get a pre-configured classification model for the given settings.

    The model's hyperparameters have been chosen based on prior work and
        were selected to facilitate good tensor core utilization on modern GPUs
        (i.e., using multiples of 8 and ideally 64, where possible).

    Args:
        design: The model design type.
        arch: The backbone architecture type.
        parch: The patcher architecture type (`arch==Architecture.VIT`).
        posenc: The positional encoding type (`arch==Architecture.VIT`).
        patchposenc: The patch positional encoding type (`arch==Architecture.VIT`).
        max_length: The maximum input length.
        num_guides: The number of semantic guides.
        structures: The hierarchical structures.
        max_structures: The maximum number of structures to use (design == Design.STRUCTURAL).
        share_embeddings: For hierarchical and structural designs, whether to share embeddings across structures.
        share_patchers: For hierarchical and structural designs, whether to share patchers across structures.
        batch_size: Compiling models can be done more effectively if the maximum batch size is known ahead of time.
        enable_checkpoint: Whether to enable activation checkpointing in certain model components to reduce memory usage.
        enable_compile: Whether to enable compilation for certain model components to improve performance.
        do_ddp_keepalive: Whether to enable DDP keepalive (to avoid timeout issues on long iterations).

    TODO: do_ddp_keepalive only applies to StructuralViTClassifier right now. Consider generalizing to other models.
    TODO: do_ddp_keepalive is compute intensive and stupid and should be removed at some point in the future.

    Returns:
        The model.
    """
    design = Design(design)
    arch = Architecture(arch)
    parch = PatcherArchitecture(parch)
    posenc = PositionalEncodingArchitecture(posenc)
    patchposenc = PatchPositionalEncodingArchitecture(patchposenc)

    num_structures = len(structures) if structures is not None else 0

    if enable_compile:
        if batch_size is None:
            raise ValueError("When enable_compile is True, batch_size must be specified for StructuralViTClassifier.")

    # The original idea was to use different architectures for different structures, for example,
    # CNNs with small strides for shorter structures, like the PE Header. Unfortunately, this does not work
    # that well in practice because the stuctures that should be short can actually be quite long due to
    # natural and unnatural variation in PE files. As a result, the architectures specialized for shorter
    # inputs end up having to handle very long inputs, which can result in lower throughput and increased
    # memory usage. Most severely, this can even lead to segmentation faults in the underlying CUDA kernels.
    # All told, we need to use architectures that can handle long inputs elegantly for all structures.
    TINY_STRUCTURES = (
        HierarchicalStructureCoarse.HEADERS,
        HierarchicalStructureCoarse.DIRECTORY,
        HierarchicalStructureMiddle.HEADERS,
        HierarchicalStructureMiddle.DIRECTORY,
        HierarchicalStructureFine.DOS_HEADER,
        HierarchicalStructureFine.DOS_STUB,
        HierarchicalStructureFine.COFF_HEADER,
        HierarchicalStructureFine.OPTN_HEADER,
        HierarchicalStructureFine.SECTN_TABLE,
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

    # (Embedding) Build the embedding
    def build_embedding() -> Embedding:
        return Embedding(num_embeddings=384, embedding_dim=embedding_dim, padding_idx=0)

    # (FiLM | FiLMNop) Build the filmer
    def build_filmer() -> FiLM | FiLMNoP:
        if num_guides == 0:
            return FiLMNoP(num_guides, embedding_dim, film_hidden_size)
        return FiLM(num_guides, embedding_dim, film_hidden_size)

    # (MalConvBase) Build the malconv
    malconv_channels = 256 if arch == Architecture.MCG else 128
    def build_malconv() -> MalConvBase:
        kernel_size = 256 if arch == Architecture.MCG else 512
        stride = 64 if arch == Architecture.MCG else 512
        if arch == Architecture.MCV:
            return MalConv(embedding_dim, malconv_channels, kernel_size, stride)
        if arch == Architecture.MC2:
            return MalConvLowMem(embedding_dim, malconv_channels, kernel_size, stride)
        if arch == Architecture.MCG:
            return MalConvGCG(embedding_dim, malconv_channels, kernel_size, stride)
        raise NotImplementedError(f"{arch}")

    # (PatchEncoderBase) Build the patcher
    if parch in (PatcherArchitecture.CNV, PatcherArchitecture.HCV):
        num_patches = None
    else:
        patch_size = None
    def build_patcher(num_patches: Optional[int], patch_size: Optional[int]) -> PatchEncoderBase:
        if parch == PatcherArchitecture.BAS:
            return PatchEncoder(embedding_dim, patcher_channels, num_patches, patch_size, kernel_size=patcher_kernel_size, stride=patcher_stride, pooling=patcher_pooling)
        if parch == PatcherArchitecture.CNV:
            return ConvPatchEncoder(embedding_dim, patcher_channels, num_patches, patch_size)
        if parch == PatcherArchitecture.HCV:
            return HierarchicalConvPatchEncoder(embedding_dim, patcher_channels, num_patches, patch_size)
        if parch == PatcherArchitecture.MEM:
            return PatchEncoderLowMem(embedding_dim, patcher_channels, num_patches, patch_size, kernel_size=patcher_kernel_size, stride=patcher_stride)
        if parch == PatcherArchitecture.EXP:
            return PatchEncoderLowMemSwitchMoE(
                embedding_dim,
                patcher_channels,
                num_patches,
                patch_size,
                kernel_size=patcher_kernel_size,
                stride=patcher_stride,
                num_experts=moe_num_experts,
                probe_dim=moe_probe_dim,
                probe_kernel_size=moe_probe_kernel_size,
                probe_stride=moe_probe_stride,
                router_hidden=moe_router_hidden,
                router_temperature=moe_router_temperature,
                router_noise_std=moe_router_noise_std,
                router_top_k=moe_router_top_k,
            )
        if parch == PatcherArchitecture.DWC:
            model = DWCPatchEncoder(
                embedding_dim, patcher_channels, num_patches, patch_size,
                depth=patcher_depth,
                kernel_size=patcher_kernel_size,
                stride=patcher_stride,
                pooling=patcher_pooling,
                checkpoint_segments=-1 if enable_checkpoint else 0,
            )
            if enable_compile:
                model.conv = torch.compile(  # type: ignore[assignment]
                    model.conv,
                    fullgraph=True,
                    dynamic=False,
                    backend="inductor",
                    mode="default",
                )
            return model
        raise NotImplementedError(f"{parch}")

    # (LayerNorm) Build the norm
    def build_norm() -> nn.LayerNorm:
        return nn.LayerNorm(patcher_channels)

    # (PatchPositionalityEncoder | Identity) Build the patchposencoder
    def build_patchposencoder() -> PatchPositionalityEncoder | Identity:
        if patchposenc == PatchPositionalEncodingArchitecture.NONE:
            return Identity()
        if patchposenc == PatchPositionalEncodingArchitecture.REL:
            return PatchPositionalityEncoder(in_channels=patcher_channels, max_length=None, hidden_size=patchposencoder_hidden_size)
        if patchposenc == PatchPositionalEncodingArchitecture.BTH:
            if max_length is None:
                raise ValueError("max_length must be specified when using BTH patch positional encodings.")
            return PatchPositionalityEncoder(in_channels=patcher_channels, max_length=max_length, hidden_size=patchposencoder_hidden_size)
        if patchposenc == PatchPositionalEncodingArchitecture.ABS:
            raise NotImplementedError(f"{patchposenc}")
        raise NotImplementedError(f"{patchposenc}")

    # (ViT) Build the transformer
    def build_transformer() -> ViT:
        posencname: Literal["none", "fixed", "learned"] = posenc.name.lower()  # type: ignore[assignment]
        if posencname not in ("none", "fixed", "learned"):
            raise TypeError(f"Unknown positional encoding: {posencname}")

        pooling: Literal["cls", "mean"] = "cls"

        if design in (Design.FLAT, Design.HIERARCHICAL):
            max_len = num_patches
            if posenc == PositionalEncodingArchitecture.LEARNED and num_patches is None:
                assert patch_size is not None
                if max_length is None:
                    raise ValueError("max_length must be specified when using a dynamic number of patches and learned positional encodings.")
                max_len = math.ceil(max_length / patch_size)
        if design == Design.STRUCTURAL:
            max_len = max_structures

        if max_len is None:
            if posenc == PositionalEncodingArchitecture.LEARNED:
                raise ValueError("max_len must be specified when using learned positional encodings.")
            if enable_compile:
                raise ValueError("max_len must be specified when enable_compile is True.")

        model = ViT(
            embedding_dim=patcher_channels,
            d_model=vit_d_model,
            nhead=vit_nhead,
            dim_feedforward=vit_dim_feedforward,
            num_layers=vit_num_layers,
            posencoder=posencname,
            max_len=max_len,
            activation="gelu",
            pooling=pooling,
            seq_lengths=backbone_seq_lengths,
            batch_sizes=backbone_batch_sizes,
        )

        if enable_compile:
            model.transformer = torch.compile(  # type: ignore[assignment]
                model.transformer,
                fullgraph=True,
                dynamic=False,
                backend="inductor",
                mode="default",
            )

        return model

    # (ClassifificationHead) Build the head
    def build_head() -> ClassifificationHead:
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            input_size = malconv_channels
        elif arch == Architecture.VIT:
            input_size = vit_d_model
        else:
            raise NotImplementedError(f"{arch}")
        return ClassifificationHead(input_size, num_classes=2, hidden_size=clf_hidden_size, num_layers=clf_num_layers, dropout=0.1)

    # Helpers
    T = TypeVar("T", bound=Module)
    def _build_modules(caller: Callable[[], T], num: int, shared: bool) -> list[T]:
        if not shared:
            return [caller() for _ in range(num)]
        instance = caller()
        return [instance for _ in range(num)]

    # (Classifier) Construct a flat model
    def construct_flat_model() -> Classifier:
        embedding = build_embedding()
        filmer = build_filmer()
        head = build_head()
        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            malconv = build_malconv()
            return MalConvClassifier(embedding, filmer, malconv, head)
        if arch in (Architecture.VIT,):
            patcher = build_patcher(num_patches, patch_size)
            norm = build_norm()
            patchposencoder = build_patchposencoder()
            transformer = build_transformer()
            return ViTClassifier(embedding, filmer, patcher, norm, patchposencoder, transformer, head)
        raise NotImplementedError(f"{arch}")

    # (HierarchicalClassifier) Construct a hierarchical model
    def construct_hierarchical_model() -> HierarchicalClassifier:
        if structures is None or len(structures) == 0 or num_structures == 0:
            raise ValueError(f"Hierarchical design requires non-empty structures {structures=}.")

        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            embeddings = _build_modules(build_embedding, num_structures, share_embeddings)
            filmers = _build_modules(build_filmer, num_structures, share_embeddings)
            malconvs = _build_modules(build_malconv, num_structures, False)
            head = build_head()
            return HierarchicalMalConvClassifier(embeddings, filmers, malconvs, head)

        if arch in (Architecture.VIT,):
            if share_patchers:
                raise NotImplementedError("Hierarchical ViT with shared patchers is not implemented.")
            # If working with a fixed-patch budget, then we allocate patches intelligently.
            # We provide one patch for each tiny structure then divide the remaining patches among the full structures.
            num_patches_tiny = None
            num_patches_full = None
            actual_num_patches = None
            if num_patches is not None:
                num_tiny = sum(1 for s in structures if s in TINY_STRUCTURES)
                num_full = sum(1 for s in structures if s not in TINY_STRUCTURES)
                num_patches_tiny = 1
                num_patches_full = (num_patches - num_tiny * num_patches_tiny) // max(1, num_full)
                actual_num_patches = num_patches_full * num_full + num_patches_tiny * num_tiny
                print(f"[rank {rank()}] [get_model] [INFO] {num_patches} were requested in total, but using {actual_num_patches} patches.")
            embeddings = _build_modules(build_embedding, num_structures, share_embeddings)
            filmers = _build_modules(build_filmer, num_structures, share_embeddings)
            patchers = [build_patcher(num_patches_tiny if s in TINY_STRUCTURES else num_patches_full, patch_size) for s in structures]
            patchposencoders = _build_modules(build_patchposencoder, num_structures, False)
            norms = _build_modules(build_norm, num_structures, False)
            transformer = build_transformer()
            head = build_head()
            return HierarchicalViTClassifier(embeddings, filmers, patchers, norms, patchposencoders, transformer, head)

        raise NotImplementedError(f"{arch}")

    # (StructuralClassifier) Construct a structural model
    def construct_structural_model() -> StructuralClassifier:
        if structures is None or len(structures) == 0 or num_structures == 0:
            raise ValueError(f"Hierarchical design requires non-empty structures {structures=}.")

        if arch in (Architecture.MCV, Architecture.MC2, Architecture.MCG):
            embeddings = _build_modules(build_embedding, num_structures, share_embeddings)
            filmers = _build_modules(build_filmer, num_structures, share_embeddings)
            malconvs = _build_modules(build_malconv, num_structures, False)
            head = build_head()
            return StructuralMalConvClassifier(embeddings, filmers, malconvs, head)

        if arch in (Architecture.VIT,):
            # If working with a fixed-patch budget, then we just allocate one patch per structure.
            # Otherwise, multiple patches may be allocated to each structure.
            num_patches_ = 1 if num_patches is not None else None
            patch_size_ = patch_size if patch_size is not None else None
            # If using the MoE-based patch encoders, then we want all structures to have shared components.
            # The specialization (i.e., the non-shared components) will be handled internally by the MoE.
            # In this case, the user should specify both `share_embeddings=True` and `share_patchers=True`.
            embeddings = _build_modules(build_embedding, num_structures, share_embeddings)
            filmers = _build_modules(build_filmer, num_structures, share_embeddings)
            patchers = _build_modules(partial(build_patcher, num_patches=num_patches_, patch_size=patch_size_), num_structures, share_patchers)
            norms = _build_modules(build_norm, num_structures, share_patchers)
            patchposencoders = _build_modules(build_patchposencoder, num_structures, share_patchers)
            transformer = build_transformer()
            head = build_head()
            return StructuralViTClassifier(
                embeddings,
                filmers,
                patchers,
                norms,
                patchposencoders,
                transformer,
                head,
                batch_sizes=patcher_batch_sizes,
                do_ddp_keepalive=do_ddp_keepalive,
            )

        raise NotImplementedError(f"{arch}")

    if design == Design.FLAT:
        return construct_flat_model()

    if share_patchers and not share_embeddings:
        warnings.warn("share_patchers is True but share_embeddings is False. While not an error, this is an unusual configuration.")
    if parch == PatcherArchitecture.EXP and not share_patchers:
        warnings.warn("parch == PatcherArchitecture.EXP but share_patchers is False. While not an error, this is an unusual configuration.")
    if share_patchers and parch != PatcherArchitecture.EXP:
        warnings.warn("share_patchers is True but parch != PatcherArchitecture.EXP. While not an error, this is an unusual configuration.")

    if design == Design.HIERARCHICAL:
        return construct_hierarchical_model()

    if design == Design.STRUCTURAL:
        return construct_structural_model()

    raise NotImplementedError(f"{design}")


def get_lr_scheduler(optimizer: Optimizer, lr_beg: float, lr_max: float, lr_end: float, total_steps: int, warmup_steps: int) -> LRScheduler:
    if optimizer.state_dict()["param_groups"][0]["lr"] != lr_max:
        raise ValueError("Optimizer learning rate does not match base_lr.")

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, total_steps, lr_end)

    # Linearly increase from `lr_beg` to `lr_max` over `warmup_steps`.
    warmup_scheduler = LinearLR(optimizer, lr_beg / lr_max, 1.0, warmup_steps)
    # Cosine decay from `lr_max` to `lr_end` over the remaining steps.
    cosine_scheduler = CosineAnnealingLR(optimizer, total_steps - warmup_steps, lr_end)
    # Chain the two schedulers together.
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps])

    return scheduler


class SupportsFullyShard(Protocol):

    def fully_shard(self, *args: Any, **kwds: Any) -> None:
        ...

M = TypeVar("M", bound=Module)

def wrap_model_base(model: M, device: torch.device) -> M:
    model = model.to(device)
    return model

def wrap_model_ddp(model: M, device: torch.device, **kwds: Any) -> DistributedDataParallel:
    model = model.to(device)
    model = DistributedDataParallel(model, **kwds)
    return model

def wrap_model_fsdp(model: M, mpdtype: torch.dtype, fsdp_offload: bool) -> M:
    mesh = init_device_mesh("cuda", (world_size(),))
    mp_policy = MixedPrecisionPolicy(mpdtype)
    offload_policy = CPUOffloadPolicy() if fsdp_offload else OffloadPolicy()
    cast(SupportsFullyShard, model).fully_shard(mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
    return model


def get_collate_fn(design: Design, min_lengths: list[int], length_bins: Optional[Sequence[int]], structures: list[HierarchicalStructure], muddy_padded: bool) -> CollateFn | CollateFnHierarchical | CollateFnStructural:
    if design == Design.FLAT:
        return CollateFn(False, False, min_lengths[0], length_bins, muddy_padded)
    if design == Design.STRUCTURAL:
        return CollateFnStructural(False, False, len(structures), min_lengths, length_bins, muddy_padded)
    if design == Design.HIERARCHICAL:
        return CollateFnHierarchical(False, False, len(structures), min_lengths, length_bins, muddy_padded)
    raise NotImplementedError(f"{design}")


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        raise RuntimeError("worker_init_fn called outside of worker process.")
    lief.logging.set_level(lief.logging.LEVEL.OFF)
    org_num_threads = torch.get_num_threads()
    if (new_num_threads := int(os.environ.get("PTW_NUM_THREADS", "-1"))) < 1:
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
    collate_fn: Callable[[Sequence[FSample]], FSamples | HSamples | SSamples],
    pin_memory: bool,
    drop_last: bool,
    prefetch_factor: int,
) -> DataLoader[FOrHSamples]:
    """
    Return a DataLoader with proper settings for multiprocessing.

    Note that providing `sampler` or `batch_sampler` will automatically
    override `shuffle`, `batch_size`, and `drop_last`. Similarly, providing
    and IterableDataset will override `shuffle` and `sampler`.
    """
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size if batch_sampler is None else None,
        shuffle=shuffle if sampler is None and batch_sampler is None and not isinstance(dataset, IterableDataset) else None,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=None if num_workers == 0 else worker_init_fn,
        multiprocessing_context=None if num_workers == 0 else "forkserver",
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=False if num_workers == 0 else True,
    )

    if loader.persistent_workers:
        iter(loader)
        time.sleep(loader.num_workers * 1.0)
        print("", end="", flush=True)

    return loader


def get_padbatch(design: Design, structures: list[HierarchicalStructure], do_parse: bool, do_entropy: bool, which_characteristics: Sequence[lief.PE.Section.CHARACTERISTICS], min_lengths: list[int], batch_size: int) -> FSamples | HSamples | SSamples:
    """Return a short batch of purely padding samples."""
    file = ["./" + "0" * 64] * batch_size
    name = [Name("0" * 64)] * batch_size
    label = torch.zeros(batch_size, dtype=torch.int64)

    def get_inputs(length: int) -> Inputs:
        inputsids = torch.zeros((batch_size, length), dtype=torch.int32)
        lengths = torch.tensor([length] * batch_size, dtype=torch.int32)
        return Inputs(inputsids, lengths, False)

    def get_guides(length: int) -> SemanticGuides:
        parse = torch.zeros((batch_size, length), dtype=torch.bool) if do_parse else None
        entropy = torch.zeros((batch_size, length), dtype=torch.float32) if do_entropy else None
        characteristics = torch.zeros((batch_size, length, len(which_characteristics)), dtype=torch.bool) if which_characteristics else None
        return SemanticGuides(parse, entropy, characteristics).decompress()

    def get_structure_as_guides(length: int) -> SemanticGuides:
        parse = None
        entropy = None
        characteristics = torch.zeros((batch_size, length, len(structures)), dtype=torch.bool)
        return SemanticGuides(parse, entropy, characteristics).decompress()

    def get_structure() -> StructureMaps:
        index: list[list[tuple[int, int]]] = [[] for _ in structures]
        lexicon = {}
        for i, s in enumerate(structures):
            index[i] = [(0, min_lengths[i])]
            lexicon[i] = s
        return StructureMaps([deepcopy(index) for _ in range(batch_size)], lexicon)

    if design == Design.FLAT and not STRUCTURES_AS_GUIDES:
        inputs = get_inputs(min_lengths[0])
        guides = get_guides(min_lengths[0])
        structure = get_structure()
        return FSamples(file, name, label, inputs, guides, structure)

    if design == Design.FLAT and STRUCTURES_AS_GUIDES:
        inputs = get_inputs(min_lengths[0])
        guides = get_structure_as_guides(min_lengths[0])
        index = [[(0, min_lengths[0])]]
        lexicon = {0: HierarchicalStructureNone.ANY}
        structure = StructureMaps([deepcopy(index) for _ in range(batch_size)], lexicon)
        return FSamples(file, name, label, inputs, guides, structure)

    if design == Design.STRUCTURAL:
        inputs_ = [get_inputs(l) for l in min_lengths]
        guides_ = [get_guides(l) for l in min_lengths]
        structure = get_structure()
        order   = [[(structure_index, local_index) for structure_index in range(len(structures))] for local_index in range(batch_size)]
        return SSamples(file, name, label, inputs_, guides_, structure, order)

    if design == Design.HIERARCHICAL:
        inputs_ = [get_inputs(l) for l in min_lengths]
        guides_ = [get_guides(l) for l in min_lengths]
        structure = get_structure()
        return HSamples(file, name, label, inputs_, guides_, structure)

    raise NotImplementedError(f"{design}")


def decay_aware_param_groups(model: Module, weight_decay: float, allow_overlapping: bool = False, verbose: bool = False) -> list[dict[str, Any]]:
    """
    Return decay aware parameter groups. Decay will not be applied to biases, normalization, and embeddings.

    Args:
        model: The model.
        weight_decay: The weight decay rate.
        allow_overlapping: If overlapping parameters are found between decay
            and non-decay groups, moves them to the non-decay group instead of raising.
        verbose: If True, prints each param's name and whether it will be decayed.

    Returns:
        A list of parameter groups, each with a `weight_decay` key.
    """

    # Parameters in these modules will be ignored.
    layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LazyInstanceNorm1d,
        nn.LazyInstanceNorm2d,
        nn.LazyInstanceNorm3d,
        nn.LayerNorm,
        nn.RMSNorm,
        nn.Embedding,
        nn.EmbeddingBag,
    )

    # Parameters in these modules and their submodules will be ignored recursively.
    layersr = (
        FiLM,
    )

    # Collect the ids of all params into two bins: one for no decay, one for decay.
    no_decay_ids: set[int] = set()
    ys_decay_ids: set[int] = set()
    for module in model.modules():
        # Don't decay these layers.
        if isinstance(module, layers):
            for p in module.parameters(recurse=False):
                if not p.requires_grad:
                    continue
                no_decay_ids.add(id(p))
            continue
        if isinstance(module, layersr):
            for p in module.parameters(recurse=True):
                if not p.requires_grad:
                    continue
                no_decay_ids.add(id(p))
            continue
        # Otherwise, decay them, ignoring bias terms (one-dimensional).
        for n, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if id(p) in no_decay_ids:
                continue
            if n == "bias":
                no_decay_ids.add(id(p))
            elif "bias" in n.lower():
                warnings.warn(f"Bias-like parameter {n} will be excluded from decay.")
                no_decay_ids.add(id(p))
            elif p.ndim == 1:
                warnings.warn(f"Bias-like parameter {n} will not be excluded from decay.")
                ys_decay_ids.add(id(p))
            else:
                ys_decay_ids.add(id(p))

    if verbose:
        print("Param Groups:")
        for n, p in model.named_parameters(recurse=True):
            s = " " * (31 - len(n))
            if not p.requires_grad:
                print(f"  {n} {s} --> no grad")
            elif id(p) in no_decay_ids:
                print(f"  {n} {s} --> no decay")
            elif id(p) in ys_decay_ids:
                print(f"  {n} {s} --> decay")
            else:
                raise RuntimeError("Unreachable.")

    # Ensure every id param appears exactly once.
    if (intersection := set.intersection(no_decay_ids, ys_decay_ids)):
        if not allow_overlapping:
            raise ValueError(f"Found overlapping param ids: {intersection}.")
        for i in intersection:
            ys_decay_ids.remove(i)

    # Collect the params into two lists: one for no decay, one for decay.
    no_decay_params: list[nn.Parameter] = []
    ys_decay_params: list[nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in no_decay_ids:
            no_decay_params.append(p)
            continue
        if id(p) in ys_decay_ids:
            ys_decay_params.append(p)
            continue
        raise ValueError(f"Parameter {id(p)} is not covered by a param group.")

    # Ensure every trainable param appears exactly once.
    total = sum(1 for p in model.parameters() if p.requires_grad)
    if len(no_decay_params) + len(ys_decay_params) != total:
        raise ValueError(f"Found {total} trainable parameters, but {len(no_decay_params) + len(ys_decay_params)} are in param groups.")

    groups = [
        {"params": ys_decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return groups


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


def main_run_profile(trainer: Trainer, tr_batch_size: int, num_samples: int = 128) -> None:
    if world_size() != 1:
        raise RuntimeError("Profiling only implemented for single worker.")

    trainer.args.eval_steps = None
    trainer.args.eval_epochs = None
    trainer.args.chpt_steps = None
    trainer.args.chpt_epochs = None
    trainer.args.logg_steps = None
    trainer.args.logg_epochs = None

    # If saving stacks, profile a small number of active windows.
    with_stack = os.environ.get("WITH_STACK", "0") == "1"
    print(f"Profiling with{'' if with_stack else 'out'} stack analysis.")

    with_groups = int(os.environ.get("WITH_GROUPS", "0"))
    print(f"Profiling with {with_groups} grouped averages.")

    skip_first = 0
    wait = 0
    warmup = 16
    active = 16
    repeat = 1
    sched = schedule(skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=repeat)
    total_mini_steps = skip_first + (wait + warmup + active) * repeat

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    i = 0
    while (outdir := trainer.args.outdir / f"tb_trace-{i}").exists():
        i += 1
    file_cpu = outdir / "cpu.txt"
    file_gpu = outdir / "gpu.txt"
    file_stk = outdir / "stacks.txt"
    file_grp = outdir / "groups.txt"

    handler = tensorboard_trace_handler(outdir.as_posix())

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        trainer.train(prof, end_mini_step=total_mini_steps)
        print("Profiling complete. Computing averages...")
        averages = prof.key_averages()
        tab_cpu = averages.table(sort_by="cpu_time_total", row_limit=20)
        print(f"Top Usage (CPU):\n{tab_cpu}")
        file_cpu.write_text(tab_cpu)
        tab_gpu = averages.table(sort_by="cuda_time_total", row_limit=20)
        print(f"Top Usage (GPU):\n{tab_gpu}")
        file_gpu.write_text(tab_gpu)
        if with_stack:
            print("Exporting stacks...")
            prof.export_stacks(file_stk.as_posix(), "self_cpu_time_total")

        if with_groups > 0:
            print("Computing grouped averages...")
            averages = prof.key_averages(group_by_stack_n=with_groups)
            averages = [e for e in averages if e.key in ("aten::copy_", "aten::to", "aten::_to_copy", "aten::to", "aten::add_")]
            for e in sorted(averages, key=lambda x: x.cpu_time_total, reverse=True)[:with_groups * 2]:
                print(e.key, e.cpu_time_total, e.count)
                with open(file_grp, "a") as fp:
                    print(e.key, e.cpu_time_total, e.count, file=fp)
                if e.stack:
                    for s in e.stack[:12]:
                        print(f"  {s}")
                        with open(file_grp, "a") as fp:
                            print(f"  {s}", file=fp)
                print("-" * 80)
                with open(file_grp, "a") as fp:
                    print("-" * 80, file=fp)


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

    structures = list(LEVEL_STRUCTURE_MAP[args.level])
    if args.ignore_directory_structures:
        structures = [s for s in structures if s not in DIRECTORY_STRUCTURES]
    num_guides = len(args.which_characteristics) + (1 if args.do_entropy else 0)
    if STRUCTURES_AS_GUIDES:
        num_guides = len(structures)
    print(f"structures={pformat(structures)}")
    print(f"{num_guides=}")

    ################################################################################
    # Static Shape Logic (BEG).
    ################################################################################

    patcher_seq_lengths = None   # Lengths that inputs (bytes) may take
    patcher_batch_sizes = None   # Batch sizes that inputs (bytes) may take
    backbone_seq_lengths = None  # Lengths that backbone (transformer) may take
    backbone_batch_sizes = None  # Batch sizes that backbone (transformer) may take

    # patcher_seq_lengths
    if args.enable_compile or args.static_shapes_bin_patcher_seq_lengths:
        if args.design == Design.FLAT:
            if args.max_length is None:
                raise RuntimeError(f"When using static shapes with the {args.design} design, max_length must be specified.")
            longest = args.max_length
        else:
            if args.max_length_per_structure is None:
                raise RuntimeError(f"When using static shapes with the {args.design} design, max_length must be specified.")
            longest = args.max_length_per_structure
        patcher_seq_lengths = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, longest]
        patcher_seq_lengths = sorted(set(filter(lambda x: x <= longest, patcher_seq_lengths)))
    # patcher_batch_sizes
    if args.enable_compile or args.static_shapes_bin_patcher_batch_sizes:
        patcher_batch_sizes = [args.tr_batch_size, args.tr_batch_size // 2, args.ts_batch_size, args.ts_batch_size // 2, 1]
        patcher_batch_sizes = sorted(set(filter(lambda x: x > 0, patcher_batch_sizes)))
    # backbone_seq_lengths
    if args.enable_compile or args.static_shapes_bin_backbone_seq_lengths:
        if args.design == Design.STRUCTURAL:
            if args.max_structures is None:
                raise RuntimeError(f"When using static shapes with the {args.design} design, max_structures must be specified.")
            max_len = args.max_structures
        else:
            max_len = args.model_config.get("num_patches") or signature(get_model).parameters["num_patches"].default
        backbone_seq_lengths = [max_len, max_len // 2, max_len // 4]
        backbone_seq_lengths = sorted(set(filter(lambda x: x > 0, backbone_seq_lengths)))
    # backbone_batch_sizes
    if args.enable_compile or args.static_shapes_bin_backbone_batch_sizes:
        backbone_batch_sizes = [args.tr_batch_size, args.ts_batch_size]
        backbone_batch_sizes = sorted(set(filter(lambda x: x > 0, backbone_batch_sizes)))

    # Allow overwriting from CLI.
    patcher_seq_lengths = args.model_config.get("patcher_seq_lengths", patcher_seq_lengths)
    patcher_batch_sizes = args.model_config.get("patcher_batch_sizes", patcher_batch_sizes)
    backbone_seq_lengths = args.model_config.get("backbone_seq_lengths", backbone_seq_lengths)
    backbone_batch_sizes = args.model_config.get("backbone_batch_sizes", backbone_batch_sizes)

    print(f"{patcher_seq_lengths=}")
    print(f"{patcher_batch_sizes=}")
    print(f"{backbone_seq_lengths=}")
    print(f"{backbone_batch_sizes=}")

    if patcher_seq_lengths is not None and len(patcher_seq_lengths) == 0:
        raise RuntimeError("No valid patcher sequence lengths could be determined.")
    if patcher_batch_sizes is not None and len(patcher_batch_sizes) == 0:
        raise RuntimeError("No valid patcher batch sizes could be determined.")
    if backbone_seq_lengths is not None and len(backbone_seq_lengths) == 0:
        raise RuntimeError("No valid backbone sequence lengths could be determined.")
    if backbone_batch_sizes is not None and len(backbone_batch_sizes) == 0:
        raise RuntimeError("No valid backbone batch sizes could be determined.")

    # Need one compiled kernel per unique shape times two (with/without grad).
    if args.enable_compile:
        assert patcher_seq_lengths is not None
        assert patcher_batch_sizes is not None
        assert backbone_seq_lengths is not None
        assert backbone_batch_sizes is not None
        cache_size_for_patchers = len(patcher_seq_lengths)  * len(patcher_batch_sizes)
        cache_size_for_backbone = len(backbone_seq_lengths) * len(backbone_batch_sizes)
        torch._dynamo.config.cache_size_limit = 2 * (cache_size_for_patchers + cache_size_for_backbone)
        print(f"{cache_size_for_patchers=}")
        print(f"{cache_size_for_backbone=}")
        print(f"{torch._dynamo.config.cache_size_limit=}")

    ################################################################################
    # Static Shape Logic (END).
    ################################################################################

    model = get_model(
        args.design,
        args.arch,
        args.parch,
        args.posenc,
        args.patchposenc,
        num_guides,
        structures,
        args.max_structures,
        args.max_length,
        args.share_embeddings,
        args.share_patchers,
        args.tr_batch_size,
        args.enable_checkpoint,
        args.enable_compile,
        do_ddp_keepalive=False,
        patcher_batch_sizes=patcher_batch_sizes,
        backbone_seq_lengths=backbone_seq_lengths,
        backbone_batch_sizes=backbone_batch_sizes,
        **dict(args.model_config),
    ).to("cpu")
    num_parameters = count_parameters(model, requires_grad=False)
    min_length = roundup(get_model_input_lengths(model)[0], PAD_TO_MULTIPLE_OF)
    min_lengths = [roundup(l, PAD_TO_MULTIPLE_OF) for l in getattr(model, "min_lengths", [min_length])]
    print(f"{model=}")
    print(f"{num_parameters=}")
    print(f"{min_length=}")
    print(f"{min_lengths=}")

    wrap_model: Callable[[M], M | DistributedDataParallel]
    if args.ddp:
        wrap_model = partial(wrap_model_ddp, device=args.device, static_graph=False, find_unused_parameters=False)
    elif args.fsdp:
        wrap_model = partial(wrap_model_fsdp, mpdtype=mpdtype, fsdp_offload=args.fsdp_offload)
    else:
        wrap_model = partial(wrap_model_base, device=args.device)
    print(f"{wrap_model=}")

    preprocessor = Preprocessor(
        args.do_entropy,
        args.which_characteristics,
        args.level,
        structures=structures,
        max_length=args.max_length,
        unsafe=False,
        structures_as_guides=STRUCTURES_AS_GUIDES,
        max_structures=args.max_structures if args.design == Design.STRUCTURAL else None,
        max_length_per_structure=args.max_length_per_structure,
    )
    print(f"{preprocessor=}")

    root = Path("./data")
    print(f"{root=}")

    tr_datadb = SimpleDB(root / "data" / "tr", check=False)
    ts_datadb = SimpleDB(root / "data" / "ts", check=False)
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

    # tr_stats_df = get_shardwise_stats(tr_datadb, tr_last_shard)
    # ts_stats_df = get_shardwise_stats(ts_datadb, ts_last_shard)
    # with pd.option_context("display.max_rows", 4, "display.max_columns", None):
    #     print(f"tr_stats_df=\n{tr_stats_df}")
    #     print(f"ts_stats_df=\n{ts_stats_df}")

    # Split the shards between distributed workers.
    if world_size() > 1:
        tr_shards = tr_shards[rank()::world_size()]
        ts_shards = ts_shards[rank()::world_size()]
        print(f"[rank {rank()}] tr_shards={tr_shards}")
        print(f"[rank {rank()}] ts_shards={ts_shards}")

    if args.num_workers > len(tr_shards):
        warnings.warn(f"More workers requested ({args.num_workers}) than tr shards ({len(tr_shards)}). Number of workers will be reduced to {len(tr_shards)}.")
    if args.num_workers > len(ts_shards):
        warnings.warn(f"More workers requested ({args.num_workers}) than ts shards ({len(ts_shards)}). Number of workers will be reduced to {len(ts_shards)}.")

    tr_dataset = IterableSimpleDBDataset(tr_datadb, tr_metadb, preprocessor, tr_shards, shuffle=True)
    ts_dataset = IterableSimpleDBDataset(ts_datadb, ts_metadb, preprocessor, ts_shards, shuffle=False)
    print(f"{tr_dataset=}")
    print(f"{ts_dataset=}")
    print(f"{len(tr_dataset)=}")
    print(f"{len(ts_dataset)=}")

    collate_fn = get_collate_fn(args.design, min_lengths, patcher_seq_lengths, structures, args.muddy_padded)
    print(f"{collate_fn=}")

    tr_loader = get_loader(tr_dataset, args.tr_batch_size, True,  None, None, min(args.num_workers, len(tr_shards)), collate_fn, args.pin_memory, False, args.prefetch_factor)
    ts_loader = get_loader(ts_dataset, args.ts_batch_size, False, None, None, min(args.num_workers, len(ts_shards)), collate_fn, args.pin_memory, False, args.prefetch_factor)
    print(f"tr_loader=DataLoader(pin_memory={tr_loader.pin_memory}, num_workers={tr_loader.num_workers}, prefetch_factor={tr_loader.prefetch_factor}, drop_last={tr_loader.drop_last})")
    print(f"ts_loader=DataLoader(pin_memory={ts_loader.pin_memory}, num_workers={ts_loader.num_workers}, prefetch_factor={ts_loader.prefetch_factor}, drop_last={ts_loader.drop_last})")
    print(f"{len(tr_loader)=} {largest_possible_dataloader_length(tr_loader)=}")
    print(f"{len(ts_loader)=} {largest_possible_dataloader_length(ts_loader)=}")

    rargs = TrainerArgs.from_dict(args.to_dict())
    print(f"{rargs=}")

    loss_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"{loss_fn=}")

    optimizer_init: Callable[[ParamsT], Optimizer] = partial(AdamW, lr=args.lr_max, weight_decay=args.weight_decay)
    print(f"{optimizer_init=}")

    if args.max_steps is not None:
        total_steps = args.max_steps
    elif args.max_epochs is not None:
        # Compute the total number of steps based on the longest epoch across all workers.
        # This is a copy-paste from the Trainer class. Maybe we can refactor later.
        length = torch.tensor(largest_possible_dataloader_length(tr_loader), device=args.device)
        if is_dist():
            lengths = [torch.zeros(1, dtype=length.dtype, device=args.device) for _ in range(world_size())]
            dist.all_gather(lengths, length, group=None)
            lengths = [int(l.item()) for l in lengths]
        else:
            lengths = [int(length.item())]
        longest = max(lengths)
        steps_per_epoch = math.ceil(longest / args.gradient_accumulation_steps)
        total_steps = max(1, math.ceil(steps_per_epoch * args.max_epochs))
    else:
        raise ValueError("Either `max_steps` or `max_epochs` must be specified.")
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[rank {rank()}] {total_steps=}, {warmup_steps=}")

    scheduler_init: Callable[[Optimizer], LRScheduler]
    if args.sched == Scheduler.NONE:
        scheduler_init = partial(LambdaLR,
            lr_lambda=lambda _: 1.0,
        )
        print("scheduler=LambdaLR(")
    elif args.sched == Scheduler.OCLR:
        scheduler_init = partial(OneCycleLR,
            max_lr=args.lr_max,
            total_steps=total_steps,
            pct_start=args.warmup_ratio,
            div_factor=args.lr_max / args.lr_beg,
            final_div_factor=args.lr_beg / args.lr_end,
        )
        print("scheduler=OneCycleLR(")
    elif args.sched == Scheduler.CUST:
        scheduler_init = partial(get_lr_scheduler,
            lr_beg=args.lr_beg,
            lr_max=args.lr_max,
            lr_end=args.lr_end,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        print("scheduler=CustomLR(")
    else:
        raise NotImplementedError(f"{args.sched} scheduler not implemented.")
    print(f"  [{0, warmup_steps}] := {args.lr_beg} --> {args.lr_max}")
    print(f"  [{warmup_steps, total_steps - warmup_steps}] := {args.lr_max} --> {args.lr_end}")
    print(")")
    print(f"{scheduler_init=}")

    padbatch = get_padbatch(args.design, structures, args.do_parser, args.do_entropy, args.which_characteristics, min_lengths, args.vl_batch_size)
    print(f"{padbatch=}")

    stopper = EarlyStopper(args.stopper_patience, args.stopper_threshold, args.stopper_mode)  # type: ignore[arg-type]
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
            tr_loader=tr_loader,
            vl_loader=ts_loader,
            loss_fn=loss_fn,
            optimizer_init=optimizer_init,
            get_param_groups=partial(decay_aware_param_groups, weight_decay=args.weight_decay),
            scheduler_init=scheduler_init,
            device=args.device,
        )
        if rank() == 0:
            print("Retrieved a trainer from checkpoint. Current log: ")
            keys = {"epoch", "step", "tr_loss", "vl_loss", "vl_roc", "vl_prc"}
            for d in trainer.log:
                print({k: v for k, v in Trainer.round_and_filter_results_dict(d).items() if k in keys})
        # Updates to the saved training args should be applied with care.
        attrs = [
            "outdir",
            "gradient_accumulation_steps",
            "eval_epochs", "eval_steps",
            "chpt_epochs", "chpt_steps",
            "logg_epochs", "logg_steps",
        ]
        for attr in attrs:
            if getattr(trainer.args, attr) != getattr(args, attr):
                print(f"[NOTE] Loaded trainer mismatch (`{attr}`). Updating ({getattr(trainer.args, attr)} --> {getattr(args, attr)}).")
                setattr(trainer.args, attr, getattr(args, attr))
        # Patch the padbatch to account for changes to the data module.
        # TODO: we should probably try to improve this aspect of the system. Its a little unreliable.
        trainer.padbatch = padbatch

    else:
        wmodel = wrap_model(model)
        params = decay_aware_param_groups(wmodel, args.weight_decay)
        optimizer = optimizer_init(params)
        scheduler = scheduler_init(optimizer)
        trainer = Trainer(
            args=rargs,
            model=wmodel,
            tr_loader=tr_loader,
            vl_loader=ts_loader,
            padbatch=padbatch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            stopper=stopper,
            device=args.device,
        )
    print(f"{trainer=}")
    print("", end="", flush=True)

    if MAIN_RUN_PROFILE:
        main_run_profile(trainer, args.tr_batch_size)
        return

    trainer = trainer()
    return


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            print(f"[rank {rank()}] Destroying process group")
            dist.destroy_process_group()
            print(f"[rank {rank()}] Process group destroyed")
