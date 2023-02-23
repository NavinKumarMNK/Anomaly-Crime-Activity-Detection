'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../../../') not in sys.path:
    sys.path.append(os.path.abspath('../../../'))
import utils.utils as utils

# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
        
        self.i = 0
        self.index = 0

    def get_image_batch(self):
        video_path = self.annotation_train[self.index]
        video_path = os.path.join(self.data_path, video_path) 
        cap = cv2.VideoCapture(video_path.strip())     
        if not cap.isOpened():
            print("Error opening video stream or file")
        frames = []
        #random frames 
        count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get random frame indexes for batch size
        ret_frames= np.random.randint(0, count, self.batch_size)
        for frame in ret_frames:
            cap.set(1, frame)
            ret, frame = cap.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = self.preprocessing.transforms(torch.from_numpy(frame))
                frame = self.preprocessing.preprocess(frame)
                frame = self.preprocessing.augumentation(frame)
                frames.append(frame)

        X = torch.stack(frames)
        y = X.clone()
        frames = []
        self.X = X
        self.y = y 
        self.index += 1

    def __len__(self):
        return len(self.annotation_train)

    def __getitem__(self, index:int):
        if(self.i+1 == self.batch_size or self.i == 0):
            self.i=0
            self.get_image_batch()
        return self.X[index % self.batch_size], self.y[index % self.batch_size]
            
            
if __name__ == '__main__':
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    dataset = AutoEncoderDataset(**dataset_params)
    
    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
