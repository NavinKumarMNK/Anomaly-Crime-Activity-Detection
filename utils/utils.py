import pytorch_lightning
import torch
import torch.nn as nn


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
