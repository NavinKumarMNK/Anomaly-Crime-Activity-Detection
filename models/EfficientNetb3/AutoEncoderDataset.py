'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../../') not in sys.path:
    sys.path.append(os.path.abspath('../../'))
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
from models.preprocessing import ImagePreProcessing

class AutoEncoderDataset(Dataset):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path) -> None:
        super(AutoEncoderDataset, self).__init__()
        self.data_path = utils.ROOT_PATH + data_path
        annotation_train = utils.dataset_image_autoencoder()
        self.annotation_train = open(annotation_train, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.preprocessing = ImagePreProcessing()
        
    def __len__(self):
        return len(self.annotation_train)

    def __getitem__(self, index:int):
        video_path = self.annotation_train[index]
        video_path = os.path.join(self.data_path, video_path) 
        cap = cv2.VideoCapture(video_path.strip())     
        if not cap.isOpened():
            print("Error opening video stream or file")
        frames = []
        #random frames 
        count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get random frame indexes for batch size
        ret_frames= np.random.randint(0, count, self.batch_size)
        print(ret_frames)
        for frame in ret_frames:
            cap.set(1, frame)
            ret, frame = cap.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = self.preprocessing.transforms(torch.from_numpy(frame))
                frame = self.preprocessing.preprocess(frame)
                frame = self.preprocessing.augumentation(frame)
                frames.append(frame)

        X = torch.stack(frames, dim=0)
        y = X.clone()
        frames = []
        print(X.shape, y.shape)
        return X, y    
    
class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path) -> None:
        super(AutoEncoderDataModule, self).__init__()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path

    def setup(self, stage=None):
        full_dataset = AutoEncoderDataset(self.batch_size, self.num_workers, self.data_path)
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

if __name__ == '__main__':
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    dataset = AutoEncoderDataModule(**dataset_params)
    dataset.setup()
    
    train_loader = dataset.train_dataloader()
    for i, (X, y) in enumerate(train_loader):
        print(X.shape, y.shape)
        break
