"""
Sets random seed for reproducibility.
"""

import os
import random

import numpy as np
import torch


SEED = int(os.environ.get("SSViT_SEED", "0"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
