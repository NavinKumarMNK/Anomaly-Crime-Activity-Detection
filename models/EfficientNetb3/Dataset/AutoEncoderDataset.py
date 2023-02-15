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
    def __init__(self, annotation_train:str, annotation_test:str,
                    batch_size:int, num_workers:int,
                    data_path) -> None:
        super(AutoEncoderDataset, self).__init__()
        self.data_path = utils.ROOT_PATH + data_path
        self.annotation_train = open(self.data_path + annotation_train, 
                                        'r').read().splitlines()
        self.annotation_test = open(self.data_path + annotation_test, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.preprocessing = ImagePreProcessing()


    def __len__(self):
        return len(self.annotation_test), len(self.annotation_train)

    def __getitem__(self, index:int, is_train:bool=True):
        if is_train:
            annotation = self.annotation_train[index]
        else:
            annotation = self.annotation_test[index]
        
        video_path = self.data_path + annotation
        video = cv2.VideoCapture(video_path)
        
        frames = []
        for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            video.set(1, i)
            ret, frame = video.read()
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = self.preprocessing.transforms(torch.from_numpy(frame))
                frame = self.preprocessing.preprocess(frame)
                frame = self.preprocessing.augumentation(frame)
                frames.append(frame)

                if((i + 1) % self.batch_size == 0):
                    X = torch.stack(frames)
                    y = X.clone()
                    frames = []
                    return X, y 
            else:
                break

    def setup(self, stage: str) -> None:
        if(stage == 'fit'):
            self.annotation_train = self.annotation_train[:int(0.8*len(self.annotation_train))]
            self.annotation_val = self.annotation_train[int(0.8*len(self.annotation_train)):]
        else:
            self.annotation_test = self.annotation_test    

if __name__ == '__main__':
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    dataset = AutoEncoderDataset(**dataset_params)
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

    