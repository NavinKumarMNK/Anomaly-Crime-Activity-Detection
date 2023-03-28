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
from models.EfficientNetv2.Encoder import EfficientNetv2Encoder

class AnomalyDataset(Dataset):
    def __init__(self, data_path, batch_size, annotation_train) -> None:
        super(AnomalyDataset, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.annotation = open(annotation_train,
                                        'r').read().splitlines()
        self.preprocessing = ImagePreProcessing()

    def __len__(self): 
        return len(self.annotation)
    
    def __getitem__(self, idx):
        while True:
            string = self.annotation[idx]
            lst = string.split('  ')

            label = float(utils.label_parser(lst[1]))
            start, end = int(lst[2]), int(lst[3])
            start2, end2 = int(lst[4]), int(lst[5])
            video_path = lst[0]
            video_path = os.path.join(self.data_path, lst[1], video_path) 
            cap = cv2.VideoCapture(video_path.strip())
            if not cap.isOpened():
                idx+=1
                continue

            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < self.batch_size:
                continue                

            frame_no = []
            frames = []
            original = []
            try:
                if label == 8:
                    frame_no = np.random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.batch_size)
                    frame_no = np.sort(frame_no)
                else:
                    frame_no = np.random.randint(start, end, self.batch_size)
                    frame_no = np.sort(frame_no)
            except Exception as e:
                print(lst, e)
                continue
                
            for frame in frame_no:
                cap.set(1, frame)
                ret, frame = cap.read()
                if ret:
                    frame = np.transpose(frame, (2, 0, 1))
                    frame = self.preprocessing.transforms(torch.from_numpy(frame))
                    frame = self.preprocessing.preprocess(frame)
                    framex = self.preprocessing.augumentation(frame)
                    frames.append(framex)
                    framey = self.preprocessing.improve(frame)
                    original.append(framey)

            if True:
            	# convert 5d [1, 4, 3, 256 ,256] to [4, 3, 256, 256] in torch
                X = torch.stack(frames, dim=0)
                y = X.clone()
                break
            '''
            else:
                if (idx + 1 == len(self.annotation)):
                    idx = 0
                else:
                    idx +=1
            '''
        return X, y
        

class AnomalyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation_train) -> None:
        super(AnomalyDataModule, self).__init__()
        self.annotation_train = annotation_train
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path

    def setup(self, stage=None):
        full_dataset = AnomalyDataset(batch_size=self.batch_size,
                                           data_path=self.data_path,
                                           annotation_train=self.annotation_train)
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers,
                           shuffle=True, drop_last=True, pin_memory=True) 

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers,
                           shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers,
                           shuffle=False, drop_last=True, pin_memory=True)


if __name__ == '__main__':
    dataset_params = utils.config_parse('ANOMALY_DATASET')
    annotation_train = utils.dataset_image_autoencoder(
                            dataset_params['data_path'])
    dataset = AnomalyDataModule(**dataset_params, 
                    annotation_train=annotation_train)
    dataset.setup()
    train_loader = dataset.train_dataloader()
    for i, (x, y) in enumerate(train_loader):
        x = x.view(x.size(1), x.size(2), x.size(3), x.size(4))
        break
