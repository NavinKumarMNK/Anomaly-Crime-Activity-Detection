"@author: NavinKumarMNK"
import os
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import cv2

# write a code for custom pytroch lightning dataset, to retrieve the video data in batches of size n using the train_annotaion file and do transforms, augumentation, preprocess and take every5th frame of the video and make it as Tensor DataLoader that is passed to lrcn model

class CrimeActivityLRCNDataset(pl.LightningDataModule):
    def __init__(self, annotation_train:str, annotation_test:str, 
                        data_dir_path:str, num_workers:int,
                        sample_rate:int, batch_size:int
                        ):
        super(CrimeActivityLRCNDataset, self).__init__() 
        self.data_path = data_dir_path
        self.annotation_train = open(self.data_path + annotation_train, 'r').read().splitlines()
        self.annotation_test = open(self.data_path + annotation_test, 'r').read().splitlines()
        self.num_workers = num_workers
        self.transforms = self.set_transforms()
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def set_transforms(self):
        pass

    def __len__(self):
        return len(self.annotation_train), len(self.annotation_test)
    
    def __getitem__(self, index:int):
        return self.annotations_train[index]
    
    def setup(self, stage: str) -> None:
        # read the video file from the annotation_train
        # read the video file from the annotation_test
        pass

    def train_dataloader(self):
        return DataLoader(self.annotation_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.annotation_train, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.annotation_test, batch_size=self.batch_size, shuffle=False, num_workers=4)
