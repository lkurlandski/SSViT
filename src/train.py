"""
Train and validate models.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any
from typing import Optional
import warnings

import lief
import numpy as np
from torch.nn import Embedding
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures import ClassifificationHead
from src.architectures import FiLM
from src.architectures import FiLMNoP
from src.architectures import MalConv
from src.architectures import MalConvClassifier
from src.binanal import HierarchicalLevel
from src.data import BinaryDataset
from src.data import CollateFn
from src.data import CUDAPrefetcher
from src.data import Preprocessor
from src.data import GroupedLengthBatchSampler
from src.trainer import Trainer
from src.trainer import TrainerArgumentParser
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper


@dataclass
class MainArgs:
    do_parser: bool = False
    do_entropy: bool = False
    do_characteristics: bool = False
    level: HierarchicalLevel = HierarchicalLevel.NONE
    tr_num_samples: Optional[int] = None
    vl_num_samples: Optional[int] = None
    max_length: Optional[int] = None
    bitpack: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 1
    tr_batch_size: int = 1
    vl_batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_workers > 0:
            if os.environ.get("OMP_NUM_THREADS") is None:
                warnings.warn(f"Parallel data loading is enabled with num_workers={self.num_workers}, but OMP_NUM_THREADS is not set, which could result in CPU oversubscription.")
            if os.environ.get("MKL_NUM_THREADS") is None:
                warnings.warn(f"Parallel data loading is enabled with num_workers={self.num_workers}, but MKL_NUM_THREADS is not set, which could result in CPU oversubscription.")
        if self.tr_batch_size == 1 or self.vl_batch_size == 1:
            raise NotImplementedError("Batch size of 1 is not supported right now. See https://docs.pytorch.org/docs/stable/data#working-with-collate-fn.")


class MainArgumentParser(ArgumentParser):

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.add_argument("--do_parser", action="store_true", default=MainArgs.do_parser)
        self.add_argument("--do_entropy", action="store_true", default=MainArgs.do_entropy)
        self.add_argument("--do_characteristics", action="store_true", default=MainArgs.do_characteristics)
        self.add_argument("--level", type=HierarchicalLevel, default=MainArgs.level)
        self.add_argument("--max_length", type=int, default=MainArgs.max_length)
        self.add_argument("--tr_num_samples", type=int, default=MainArgs.tr_num_samples)
        self.add_argument("--vl_num_samples", type=int, default=MainArgs.vl_num_samples)
        self.add_argument("--bitpack", action="store_true", default=MainArgs.bitpack)
        self.add_argument("--num_workers", type=int, default=MainArgs.num_workers)
        self.add_argument("--pin_memory", action="store_true", default=MainArgs.pin_memory)
        self.add_argument("--prefetch_factor", type=int, default=MainArgs.prefetch_factor)
        self.add_argument("--tr_batch_size", type=int, default=MainArgs.tr_batch_size)
        self.add_argument("--vl_batch_size", type=int, default=MainArgs.vl_batch_size)
        self.add_argument("--learning_rate", type=float, default=MainArgs.learning_rate)
        self.add_argument("--weight_decay", type=float, default=MainArgs.weight_decay)


def get_materials() -> tuple[list[Path], list[Path], list[int], list[int]]:
    # TODO: define a temporal (not random) train/test split.
    # FIXME: this doesn't actually randomize things properly.
    benfiles = list(filter(lambda f: f.is_file(), Path("./data/ass").rglob("*")))
    benlabels = [0] * len(benfiles)
    malfiles = list(filter(lambda f: f.is_file(), Path("./data/sor").rglob("*")))
    mallabel = [1] * len(malfiles)
    files = benfiles + malfiles
    labels = benlabels + mallabel
    idx = np.arange(len(files))
    tr_idx = np.random.choice(idx, size=int(0.8 * len(files)), replace=False)
    vl_idx = np.setdiff1d(idx, tr_idx)  # type: ignore[no-untyped-call]
    tr_files = [files[i] for i in tr_idx]
    vl_files = [files[i] for i in vl_idx]
    tr_labels = [labels[i] for i in tr_idx]
    vl_labels = [labels[i] for i in vl_idx]
    return tr_files, vl_files, tr_labels, vl_labels


def main() -> None:

    lief.logging.set_level(lief.logging.LEVEL.OFF)

    Parser = type("Parser", (MainArgumentParser, TrainerArgumentParser), {})
    parser = Parser(description="Train and validate models.")
    args = parser.parse_args()

    preprocessor = Preprocessor(
        args.do_parser,
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )

    tr_files, vl_files, tr_labels, vl_labels = get_materials()

    tr_dataset = BinaryDataset(tr_files[0:args.tr_num_samples], tr_labels[0:args.tr_num_samples], preprocessor)
    vl_dataset = BinaryDataset(vl_files[0:args.vl_num_samples], vl_labels[0:args.vl_num_samples], preprocessor)

    collate_fn = CollateFn(pin_memory=False, bitpack=args.bitpack)
    print(f"{collate_fn=}")

    def get_loader(dataset: Dataset, sampler: Sampler) -> DataLoader | CUDAPrefetcher:
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory and args.device.type == "cuda",
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

    model = MalConvClassifier(
        Embedding(256 + 8, 8, padding_idx=0),
        FiLM(12, 8, 16) if args.do_characteristics else FiLMNoP(),
        MalConv(8, 128, 512, 512),
        ClassifificationHead(128, 2, 32, 2),
    )

    loss_fn = CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = LinearLR(optimizer)

    stopper: Optional[EarlyStopper] = None

    trainer = Trainer(TrainerArgs.from_namespace(args), model, tr_loader, vl_loader, loss_fn, optimizer, scheduler, stopper)

    trainer = trainer()


if __name__ == "__main__":
    main()
