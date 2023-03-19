"@Author: NavinKumarMNK"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import torch
import cv2
from utils import utils
import numpy as np
from utils.preprocessing import ImagePreProcessing

class LSTMDataset(Dataset):
    def __init__(self, annotation:str,
                        data_dir_path:str, num_workers:int,
                        sample_rate:int
                        ):
        super(LSTMDataset, self).__init__() 
        self.data_path = utils.DATA_PATH + data_dir_path 
        self.annotation = open(self.data_path + annotation, 'r').read().splitlines()
        self.num_workers = num_workers
        self.preprocessing = ImagePreProcessing()
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index:int):
        annotation = self.annotation[index]
        lst = annotation.split('  ')
        video_path = self.data_path + lst[0]
        
        label = utils.label_parser(lst[1])
        video = cv2.VideoCapture(video_path)

        #sampling the frames
        frames = []

        print(int(lst[2]), int(lst[3]), self.sample_rate)
        count = 0
        for i in range(int(lst[2]), int(lst[3]), self.sample_rate):
            video.set(1, i)
            ret, frame = video.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = self.preprocessing.transforms(torch.from_numpy(frame))
                frame = self.preprocessing.preprocess(frame)
                frame = self.preprocessing.augumentation(frame)
                frames.append(frame)
                count += 1
            if count > 127:
                break

        video.release()
        video = torch.stack(frames)
        print(video.shape)
        print(label)
        
        return video, label

class LSTMDatasetModule(pl.LightningDataModule):
    def __init__(self, num_workers:int,
                    data_path, annotation, sample_rate) -> None:
        super(LSTMDatasetModule, self).__init__()
        self.annotation = annotation
        self.num_workers = int(num_workers)
        self.data_path = data_path
        self.sample_rate = sample_rate

    def setup(self, stage=None):
        self.dataset = LSTMDataset(num_workers=self.num_workers,
                                           data_dir_path=self.data_path, annotation=self.annotation,
                                           sample_rate=self.sample_rate)
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)


if __name__ == "__main__":
    data = LSTMDatasetModule(**utils.config_parse(
                'LSTM_DATASET'))
    data.setup()
    train_loader = data.train_dataloader()
    for i, (video, label) in enumerate(train_loader):
        print(video.shape)
        print(label)