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

class EncoderClassifierDataset(Dataset):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path) -> None:
        super(EncoderClassifierDataset, self).__init__()
        self.data_path = data_path
        self.annotation_train=utils.dataset_image_encoderclassifer()
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
    dataset = EncoderClassifierDataset(**dataset_params)
    dataset.setup('fit')

    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    test_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    from pytorch_lightning import Trainer
    from models.EfficientNetb3.AutoEncoder import AutoEncoder
    #from pytorch_lightning.loggers import WandbLogger
    #logger = WandbLogger(project='AutoEncoder', name='EfficientNetb3')

    #import wandb
    #wandb.init()
    from pytorch_lightning.callbacks import ModelSummary
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    device_monitor = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=utils.ROOT_PATH + '/weights/checkpoints/autoencoder/')
    model_summary = ModelSummary(max_depth=3)
    refresh_rate = TQDMProgressBar(refresh_rate=10)

    callbacks = [
        model_summary,
        refresh_rate,
        checkpoint_callback,
        early_stopping,
        device_monitor
    ]

    model = AutoEncoder()
    autoencoder_params = utils.config_parse('AUTOENCODER_TRAIN')
    print(autoencoder_params)
    trainer = Trainer(**autoencoder_params, callbacks=callbacks) #logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader,)

    