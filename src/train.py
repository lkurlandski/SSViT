"""
Train and validate models.
"""

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from functools import partial
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import Tensor

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures import MultiChannelMalConv
from src.architectures import MultiChannelDiscreteSequenceVisionTransformer
from src.binanal import HierarchicalLevel
from src.data import BinaryDataset
from src.data import CollateFn
from src.data import CUDAPrefetcher
from src.data import Preprocessor
from src.data import Samples
from src.data import GroupedLengthBatchSampler
from src.trainer import Trainer
from src.trainer import TrainerArgumentParser
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper


def get_materials() -> tuple[list[Path], list[Path], list[int], list[int]]:
    # TODO: define a temporal (not random) train/test split.
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

    parser = TrainerArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--do_parser", action="store_true", default=False)
    parser.add_argument("--do_entropy", action="store_true", default=False)
    parser.add_argument("--do_characteristics", action="store_true", default=False)
    parser.add_argument("--level", type=HierarchicalLevel, default=HierarchicalLevel.NONE)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    targs = TrainerArgs.from_namespace(args)

    preprocessor = Preprocessor(
        args.do_parser,
        args.do_entropy,
        args.do_characteristics,
        args.level,
        max_length=args.max_length,
    )

    tr_files, vl_files, tr_labels, vl_labels = get_materials()

    tr_dataset = BinaryDataset(tr_files[0:args.num_samples], tr_labels[0:args.num_samples], preprocessor)
    vl_dataset = BinaryDataset(vl_files[0:args.num_samples], vl_labels[0:args.num_samples], preprocessor)

    collate_fn = CollateFn(pin_memory=True if targs.device.type == "cuda" else False)

    def get_loader(dataset: Dataset, sampler: Sampler) -> Collection[Samples]:
        iterable: Collection[Samples] = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=targs.num_workers,
            collate_fn=collate_fn,
            multiprocessing_context=None if args.num_workers == 0 else "forkserver",
            prefetch_factor=None if args.num_workers == 0 else 1,
            persistent_workers=None if args.num_workers == 0 else True,
        )
        if targs.device.type == "cuda":
            iterable = CUDAPrefetcher(iterable, targs.device)
        return iterable

    tr_sampler = GroupedLengthBatchSampler.from_lengths(targs.tr_batch_size, list(map(os.path.getsize, tr_dataset.files)), first=True, shuffle=True)
    vl_sampler = GroupedLengthBatchSampler.from_lengths(targs.vl_batch_size, list(map(os.path.getsize, vl_dataset.files)), first=True, shuffle=False)
    tr_loader = get_loader(tr_dataset, tr_sampler)
    vl_loader = get_loader(vl_dataset, vl_sampler)


    model = MultiChannelMalConv(
        [256 + 8],
        [8],
    )

    loss_fn = CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = LinearLR(optimizer)

    stopper: Optional[EarlyStopper] = None

    trainer = Trainer(targs, model, tr_loader, vl_loader, loss_fn, optimizer, scheduler, stopper)

    trainer = trainer()


if __name__ == "__main__":
    main()
