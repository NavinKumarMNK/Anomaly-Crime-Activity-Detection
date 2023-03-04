"@author: NavinKumarMNK"
import os
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import cv2
from utils import utils
import numpy as np
from models.preprocessing import ImagePreProcessing
preprocessing = ImagePreProcessing()

class LSTMDataset(pl.LightningDataModule):
    def __init__(self, annotation:str,
                        data_dir_path:str, num_workers:int,
                        sample_rate:int, batch_size:int,
                        num_classes:int
                        ):
        super(LSTMDataset, self).__init__() 
        self.data_path = data_dir_path
        self.annotation = open(self.data_path + annotation, 'r').read().splitlines()
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def __len__(self):
        return len(self.annotation_train), len(self.annotation_test)
    
    def __getitem__(self, index:int):
        annotation = self.annotation_train[index]
        video_path = self.data_path + annotation
        label = utils.label_parser(annotation.split('/')[0])
        video = cv2.VideoCapture(video_path)

        #sampling the frames
        frames = []
        j = 0
        for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate):
            video.set(1, i)
            if(j%self.sample_rate != 0):
                continue
            ret, frame = video.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = preprocessing.transforms(frame)
                frame = preprocessing.preprocess(frame)
                frame = preprocessing.augumentation(frame)
                frames.append(frame)
            j+=1
        video.release()
        video = cv2.vconcat(frames)
        video = torch.from_numpy(video)
        print(video.shape)
        print(label)
        
        return video, label
    
    def setup(self, stage: str) -> None:
        if(stage == 'fit'):
            self.annotation_train = self.annotation_train[:int(0.8*len(self.annotation_train))]
            self.annotation_val = self.annotation_train[int(0.8*len(self.annotation_train)):]
        else:
            self.annotation_test = self.annotation_test

    def train_dataloader(self):
        return DataLoader(self.annotation, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.annotation, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == "__main__":
    data = LSTMDataset(**utils.config_parse(
                'CRIME_ACTIVITY_DATASET'))
    data.setup()
    print(data)    
    train_loader = data.train_dataloader()
    for i, (video, label) in enumerate(train_loader):
        print(video.shape)
        print(label)