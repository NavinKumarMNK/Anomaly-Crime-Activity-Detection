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
from torchvision import models
import wandb
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b3
from models.EfficientNetb3.Encoder import EfficientNetb3Encoder
from models.EfficientNetb3.Decoder import EfficientNetb3Decoder

class AutoEncoder(pl.LightningModule):
    def __init__(self, 
                    ) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = EfficientNetb3Encoder()
        self.decoder = EfficientNetb3Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def training_epoch_end(self, outputs):
        pass
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat
    
    def training_epoch_end(self, outputs) -> None:
        torch.save(self.encoder.state_dict(), utils.ROOT_PATH + '/weights/EfficientNetb3Encoder.pt')
        torch.save(self.decoder.state_dict(), utils.ROOT_PATH + '/weights/EfficientNetb3Decoder.pt')

if __name__ == '__main__':
    #from pytorch_lightning.loggers import WandbLogger
    #logger = WandbLogger(project='AutoEncoder', name='EfficientNetb3')

    #import wandb
    #wandb.init()
    from pytorch_lightning import Trainer
    from models.EfficientNetb3.Dataset.AutoEncoderDataset import AutoEncoderDataset
    from torch.utils.data import DataLoader

    dataset_params = utils.config_parse('AUTOENCODER_DATASET')
    dataset = AutoEncoderDataset(**dataset_params)

    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    
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

