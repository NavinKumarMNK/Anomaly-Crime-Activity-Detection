'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import utils.utils as utils

# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

import cv2
import PIL
import numpy as np
from utils.preprocessing import ImagePreProcessing

class AutoEncoderDataset(Dataset):
    def __init__(self, batch_size:int,
                    data_path, annotation) -> None:
        super(AutoEncoderDataset, self).__init__()
        self.data_path = data_path
        self.annotation = open(annotation, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)

        self.preprocessing = ImagePreProcessing()

        self.index = 0

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index:int):
        
        i=0
        while True:
            i+=1
            if index+i >= len(self.annotation):
                index = 0
            video_path = self.annotation[index+i]
            video_path = os.path.join(self.data_path, video_path) 
            
            cap = cv2.VideoCapture(video_path.strip())     
            count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if not (cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0):
                continue
            
            frame = np.random.randint(0, count)
                
            # Get random frame indexes for batch size
            cap.set(1, frame)
            ret, frame = cap.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = self.preprocessing.transforms(torch.from_numpy(frame))
                frame = self.preprocessing.preprocess(frame)
                
            X = frame
            y = X.clone()

            if(torch.isnan(X).any() or torch.isnan(y).any()):
                print("reported")
                continue
            else:
                break

        return X, y
    
class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation) -> None:
        super(AutoEncoderDataModule, self).__init__()
        self.annotation = annotation
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path

    def setup(self, stage=None):
        full_dataset = AutoEncoderDataset(self.batch_size,
                                           self.data_path, self.annotation)
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=True, drop_last=True, pin_memory=True) 

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=True, pin_memory=True)

if __name__ == '__main__':
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    annotation = utils.dataset_image_autoencoder(
                            dataset_params['data_path'], "anomaly_train.txt")
    dataset = AutoEncoderDataModule(**dataset_params, 
                    annotation=annotation)
    dataset.setup()
    
    train_loader = dataset.train_dataloader()
    for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        break