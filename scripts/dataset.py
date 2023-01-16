import os
from torch.utils.data import DataLoader, Dataset
import torch

class CrimeDataset(Dataset):
    def __init__(self, annotaiion_file, data_dir, transform=None):
        self.annotations = open(annotaiion_file, 'r').read().splitlines()
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
        