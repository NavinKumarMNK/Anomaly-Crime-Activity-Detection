import torch.nn as nn
import torch
import pytorch_lightning

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
