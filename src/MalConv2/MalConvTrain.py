import os
from collections import deque

import random
import numpy as np

#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils import data

from torch.utils.data import Dataset, DataLoader, Subset

from MalConv import MalConv
#from MalConv2A import MalConv2A

from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score

import argparse

import json
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score as prc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data import IterableSimpleDBDataset
from src.data import CollateFn
from src.data import Preprocessor
from src.data import MetadataDB
from src.main import get_collate_fn
from src.main import get_loader
from src.main import get_streamer
from src.trainer import print_parameter_summary
from src.utils import seed_everything
from src.simpledb import SimpleDB

#Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Train a MalConv model')

parser.add_argument('--filter_size', type=int, default=512, help='How wide should the filter be')
parser.add_argument('--filter_stride', type=int, default=512, help='Filter Stride')
parser.add_argument('--embd_size', type=int, default=8, help='Size of embedding layer')
parser.add_argument('--num_channels', type=int, default=128, help='Total number of channels in output')
parser.add_argument('--epochs', type=int, default=10, help='How many training epochs to perform')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)

args = parser.parse_args()

#GPUS = args.gpus
GPUS = None

NON_NEG = args.non_neg
EMBD_SIZE = args.embd_size
FILTER_SIZE = args.filter_size
FILTER_STRIDE = args.filter_stride
NUM_CHANNELS= args.num_channels
EPOCHS = args.epochs
MAX_FILE_LEN = args.max_len

BATCH_SIZE = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(0)
preprocessor = Preprocessor(max_length=args.max_len)
root = Path("./data")
tr_datadb = SimpleDB(root / "data" / "tr", check=False)
ts_datadb = SimpleDB(root / "data" / "ts", check=False)
tr_metadb = MetadataDB(root / "meta" / "tr")
ts_metadb = MetadataDB(root / "meta" / "ts")
tr_shards = list(range(len(tr_datadb.files_data)))
ts_shards = list(range(len(ts_datadb.files_data)))
tr_dataset = IterableSimpleDBDataset(tr_datadb, tr_metadb, preprocessor, tr_shards, shuffle=True)
ts_dataset = IterableSimpleDBDataset(ts_datadb, ts_metadb, preprocessor, ts_shards, shuffle=False)
collate_fn = CollateFn(False, False)
tr_loader = get_loader(tr_dataset, BATCH_SIZE, True,  None, None, 0, collate_fn, True, 1)
ts_loader = get_loader(ts_dataset, BATCH_SIZE, False, None, None, 0, collate_fn, True, 1)
tr_streamer = get_streamer(tr_loader, device, num_streams=0)
ts_streamer = get_streamer(ts_loader, device, num_streams=0)
train_loader = tr_streamer
test_loader = ts_streamer


model = MalConv(channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE).to(device)

logfile = Path("./tmp/raff/mc2/results.jsonl")
logfile.parent.mkdir(parents=True, exist_ok=True)
if logfile.exists():
    raise FileExistsError(logfile)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

for epoch in tqdm(range(1, EPOCHS + 1)):
    
    preds = []
    truths = []
    running_loss = 0.0


    train_correct = 0
    train_total = 0
    
    epoch_stats = {'epoch':epoch}

    model.train()
    for batch in tqdm(train_loader):

        batch = batch.to(device, non_blocking=True)
        batch = batch.finalize(ftype=torch.float32, itype=torch.int64, ltype=torch.int64)
        inputs, labels = batch.get_inputs(), batch.get_label()

        #inputs, labels = inputs.to(device), labels.to(device)
        #Keep inputs on CPU, the model will load chunks of input onto device as needed
        labels = labels.to(device)

        optimizer.zero_grad()

    #     outputs, penultimate_activ, conv_active = model.forward_extra(inputs)
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss #+ decov_lambda*(decov_penalty(penultimate_activ) + decov_penalty(conv_active))
    #     loss = loss + decov_lambda*(decov_penalty(conv_active))
        loss.backward()
        # Log parameter summary after first batch of first two epochs.
        if (epoch == 1 or epoch == 2) and train_total == 0:
            print(f"{'-' * 20} Parameter Summary {'-' * 20}")
            print_parameter_summary(model, spaces=2)
            print(f"{'-' * 80}")

        optimizer.step()
        if NON_NEG:
            for p in model.parameters():
                p.data.clamp_(0)


        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        
        with torch.no_grad():
            preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
            truths.extend(labels.detach().cpu().numpy().ravel())
        
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    #print("Training Accuracy: {}".format(train_correct*100.0/train_total))
    
    epoch_stats['tr_loss'] = running_loss / train_total
    epoch_stats['tr_acc'] = train_correct*1.0/train_total
    epoch_stats['tr_auc'] = roc_auc_score(truths, preds)
    epoch_stats['tr_prc'] = prc_auc_score(truths, preds)
    #epoch_stats['train_loss'] = roc_auc_score(truths, preds)
    
    #Test Set Eval
    model.eval()
    eval_train_correct = 0
    eval_train_total = 0
    running_loss = 0.0
    
    preds = []
    truths = []
    with torch.no_grad():
        for batch in tqdm(test_loader):

            batch = batch.to(device, non_blocking=True)
            batch = batch.finalize(ftype=torch.float32, itype=torch.int64, ltype=torch.int64)
            inputs, labels = batch.get_inputs(), batch.get_label()

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
            truths.extend(labels.detach().cpu().numpy().ravel())
            
            eval_train_total += labels.size(0)
            eval_train_correct += (predicted == labels).sum().item()

    epoch_stats['vl_loss'] = running_loss / eval_train_total
    epoch_stats['vl_acc'] = eval_train_correct*1.0/eval_train_total
    epoch_stats['vl_auc'] = roc_auc_score(truths, preds)
    epoch_stats['vl_prc'] = prc_auc_score(truths, preds)

    with open(logfile, 'a') as fp:
        fp.write(f"{json.dumps(epoch_stats)}\n")
