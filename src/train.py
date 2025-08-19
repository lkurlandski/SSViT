"""
Train and validate models.
"""

from pathlib import Path
import sys
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.architectures import MultiChannelMalConv
from src.architectures import MultiChannelDiscreteSequenceVisionTransformer
from src.binanal import HierarchicalLevel
from src.data import BinaryDataset
from src.data import CollateFn
from src.data import CUDAPrefetcher
from src.data import Preprocessor
from src.data import Samples
from src.trainer import Trainer
from src.trainer import TrainerArgumentParser
from src.trainer import TrainerArgs
from src.trainer import EarlyStopper


parser = TrainerArgumentParser()
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--do_parser", action="store_true", default=False)
parser.add_argument("--do_entropy", action="store_true", default=False)
parser.add_argument("--do_characteristics", action="store_true", default=False)
parser.add_argument("--level", type=HierarchicalLevel, default=HierarchicalLevel.NONE)
args = parser.parse_args()

targs = TrainerArgs.from_namespace(args)

model = MultiChannelMalConv(
    [256 + 8],
    [8],
)

benfiles = list(filter(lambda f: f.is_file(), Path("./data/ass").rglob("*")))
benlabels = [0] * len(benfiles)
malfiles = list(filter(lambda f: f.is_file(), Path("./data/sor").rglob("*")))
mallabel = [1] * len(malfiles)
files = benfiles + malfiles
labels = benlabels + mallabel

preprocessor = Preprocessor(args.do_parser, args.do_entropy, args.do_characteristics, args.level)

dataset = BinaryDataset(files, labels, preprocessor)
tr_dataset, vl_dataset = random_split(dataset, [0.8, 0.2])

collate_fn = CollateFn(True)

loss_fn = CrossEntropyLoss()

optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = LinearLR(optimizer)

stopper: Optional[EarlyStopper] = None

trainer = Trainer(targs, model, tr_dataset, vl_dataset, collate_fn, loss_fn, optimizer, scheduler, stopper)

trainer.train()
