import sys
import os
sys.path.append(os.path.abspath('../'))
print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from utils import utils

DEVICE = utils.device()
