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

class CrimeDataset(Dataset):
    def __init__(self, batch_size:int,
                    data_path, annotation, sample_rate) -> None:
        super(CrimeDataset, self).__init__()
        self.data_path = data_path
        self.annotation = open(self.data_path + annotation, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)

        self.preprocessing = ImagePreProcessing()
        self.global_sample_rate = self.sample_rate = sample_rate
        self.index = 0
        
    def __len__(self): 
        return len(self.annotation)
    
    def __getitem__(self, idx):
        while True:
            self.sample_rate = self.global_sample_rate
            try:
                annotation = self.annotation[idx]
                lst = annotation.split('  ')
                video_path = self.data_path + lst[0]
                
                label = utils.label_parser(lst[1])
                video = cv2.VideoCapture(video_path)

                #sampling the frames
                frames = []

                if (video.get(cv2.CAP_PROP_FRAME_COUNT) < 127):
                    print("Video is too short")
                    continue
                elif (video.get(cv2.CAP_PROP_FRAME_COUNT) / self.sample_rate < self.batch_size):
                    self.sample_rate = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / self.batch_size)

                print(int(lst[2]), int(lst[3]), self.sample_rate)
                count = 0
                for i in range(int(lst[2]), int(lst[3]), self.sample_rate):
                    video.set(1, i)
                    ret, frame = video.read()
                    if ret:
                        frame = np.transpose(frame, (2, 0, 1))
                        frame = self.preprocessing.transforms(torch.from_numpy(frame))
                        frame = self.preprocessing.preprocess(frame)
                        frames.append(frame)
                        count += 1
                    if count > 127:
                        break

                video.release()
                video = torch.stack(frames)
                label = torch.tensor(label)
                print(video.shape)
                print(label)
                break

            except Exception as e:
                if (idx >= len(self.annotation)):
                    idx = 0
                print(e)
                idx += 1
                continue

        print(video.shape, label)
        return video, label         
        
class CrimeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation, sample_rate):
        super(CrimeDataModule, self).__init__()
        self.annotation = annotation
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path
        self.full_dataset = CrimeDataset(self.batch_size,
                                           self.data_path, self.annotation, sample_rate)
    
    def setup(self, stage=None):    
        train_size = int(0.8 * len(self.full_dataset))
        val_size = int(0.1 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False)
        
if __name__ == '__main__':
    dataset_params = utils.config_parse('CRIME_DATASET')    
    dataset = CrimeDataModule(**dataset_params)
    dataset.setup()
    print(len(dataset.full_dataset))