'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import utils.utils as utils
from utils.exceptions import *
# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from utils.preprocessing import ImagePreProcessing

class AnomalyCrimeDataset(Dataset):
    def __init__(self, batch_size:int,
                    data_path, annotation, sample_rate) -> None:
        super(AnomalyCrimeDataset, self).__init__()
        self.data_path = data_path
        self.annotation = open(self.data_path + annotation, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)

        self.preprocessing = ImagePreProcessing()
        if sample_rate == -1:
            self.sample_rate = np.random.randint(4, 7)
        self.index = 0
        
    def __len__(self): 
        return len(self.annotation)
    
    def __getitem__(self, idx):
        while True:
            self.sample_rate = np.random.randint(4, 7)
            try:
                annotation = self.annotation[idx]
                lst = annotation.split('  ')
                video_path = self.data_path + lst[1]  + '/' + lst[0]

                label = utils.label_parser(lst[1])
                video = cv2.VideoCapture(video_path)
                if not (video.isOpened() and video.get(cv2.CAP_PROP_FRAME_COUNT) > 0):
                    raise VideoNotOpened("Video is not opened or frame count is 0")

                #sampling the frames
                frames = []
                y = int(label)
                if (int(lst[2]) != -1):
                    if (int(lst[3]) - int(lst[2]) < self.batch_size and int(lst[2]) != -1):
                        raise VideoTooShort("Video is too short")
                    
                    elif (int(lst[3]) - int(lst[2]) // self.sample_rate < self.batch_size):
                        self.sample_rate = int((int(lst[3]) - int(lst[2])) // self.batch_size)
                    start = int(lst[2]) + np.random.randint(0, int(lst[3]) - int(lst[2]) - self.batch_size * self.sample_rate)
                    end = int(lst[3]) 
                else:
                    start = np.random.randint(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - self.batch_size * self.sample_rate)
                    end = start + self.batch_size * self.sample_rate
                    if (end > int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                        end = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        if(end - start < self.batch_size):
                            raise VideoTooShort("Video is too short")
                            
                        elif (end - start // self.sample_rate < self.batch_size):
                            self.sample_rate = int((end - start) // self.batch_size)
                
                end = start + self.batch_size * self.sample_rate
                count = 0             
                for i in range(start, end, self.sample_rate):
                    video.set(1, i)
                    ret, frame = video.read()
                    if ret:
                        frame = np.transpose(frame, (2, 0, 1))
                        frame = self.preprocessing.transforms(torch.from_numpy(frame))
                        frame = self.preprocessing.preprocess(frame)
                        frames.append(frame)
                        count += 1
                    if count >= self.batch_size:
                        break
                video.release()
                X = torch.stack(frames)
                if (X.shape[0] < self.batch_size):
                    X = torch.cat((X, X[0:self.batch_size - X.shape[0]]), dim=0)
                    y = torch.tensor([y] * self.batch_size)
                else:
                    X = X[0:self.batch_size]
                    y = torch.tensor([y] * X.shape[0])
                break       

            except Exception as e:
                if (idx >= len(self.annotation)):
                    idx = 0
                print(e)
                idx += 1
                continue

        return X, y
        
class AnomalyCrimeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation, sample_rate):
        super(AnomalyCrimeDataModule, self).__init__()
        self.annotation = annotation
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path
        self.full_dataset = AnomalyCrimeDataset(self.batch_size,
                                           self.data_path, self.annotation, sample_rate)
    
    def setup(self, stage=None):    
        train_size = int(0.8 * len(self.full_dataset))
        val_size = int(0.1 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, 
                          num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, 
                          num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, 
                          num_workers=self.num_workers, shuffle=False)


if __name__ == '__main__':
    dataset_params = utils.config_parse('CRIME_DATASET')    
    dataset = AnomalyCrimeDataModule(**dataset_params)
    dataset.setup()
    print(len(dataset.full_dataset))
    for x, y in iter(dataset.train_dataloader()):
        print(x.shape, "\n", y)
        break

