'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
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
from models.EfficientNetv2.VarEncoder import Efficientnetv2VarEncoder
from models.LSTM.Decoder import LSTMDecoder
import ray_lightning as rl
from models.LSTM.LSTMDataset import LSTMDatasetModule

class EncoderDecoder(pl.LightningModule):
    def __init__(self, 
                    ) -> None:
        super(EncoderDecoder, self).__init__()
        self.example_input_array = torch.rand(1, 3, 256, 256)
        self.save_hyperparameters()
        self.encoder = Efficientnetv2VarEncoder()
        # encoder is freeze no change in weights
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
        self.decoder = LSTMDecoder()
  
    def forward(self, x):
        #with torch.no_grad():
        mu, var = self.encoder(x)
        x = self.encoder.reparameterize(mu, var)
        x = self.decoder(x.unsqueeze(0))
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.squeeze(0))        
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return {"loss" : loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/loss_epoch', avg_loss)
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.squeeze(0))
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        return {"val_loss": loss, "y_hat": y_hat, "y": y}
        
    
    def validation_epoch_end(self, outputs)-> None:
        loss, y_hat, y = outputs[0]['val_loss'], outputs[0]['y_hat'], outputs[0]['y']
        try:
            avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        except TypeError:
            avg_loss = loss
        self.log('val/loss_epoch', avg_loss)
        
        # validation loss is less than previous epoch then save the model
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = avg_loss
            #self.save_model()
        elif (avg_loss < self.best_val_loss):
            self.best_val_loss = avg_loss
            self.save_model()

    def save_model(self):
        self.decoder.save_model()
        artifact = wandb.Artifact('encoder-deocder.cpkt', type='model')
        artifact.add_file(utils.ROOT_PATH + '/weights/checkpoints/encoder-decoder/last.ckpt')

    def print_params(self): 
        print("Model Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.squeeze(0))
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)
        return {"test_loss": loss, "y_hat": y_hat, "y": y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat
    
if __name__ == '__main__' :
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='CrimeDetection', name='Encoder-Decoder')

    import wandb
    wandb.init()
    import ray
    
    #ray.init(runtime_env={"working_dir": utils.ROOT_PATH})
    
    dataset_params = utils.config_parse('LSTM_DATASET')    
    dataset = LSTMDatasetModule(**dataset_params)
    dataset.setup()
    print(len(dataset.full_dataset))

    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    device_monitor = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=utils.ROOT_PATH + 
                      '/weights/checkpoints/encoder-decoder/', monitor="val_loss", 
                      mode='min', every_n_train_steps=100, save_top_k=4, save_last=True)
    
    refresh_rate = TQDMProgressBar(refresh_rate=10)

    callbacks = [
        refresh_rate,
        checkpoint_callback,
        early_stopping,
        device_monitor
    ]

    model = EncoderDecoder()
    ed_params = utils.config_parse('ENCODER_DECODER_TRAIN')
    
    dist_env_params = utils.config_parse('DISTRIBUTED_ENV')
    strategy = None
    if int(dist_env_params['horovod']) == 1:
        strategy = rl.HorovodRayStrategy(num_workers=dist_env_params['num_workers'],
                                        use_gpu=dist_env_params['use_gpu'])
    elif int(dist_env_params['model_parallel']) == 1:
        strategy = rl.RayShardedStrategy(num_workers=dist_env_params['num_workers'],
                                        use_gpu=dist_env_params['use_gpu'])
    elif int(dist_env_params['data_parallel']) == 1:
        strategy = rl.RayStrategy(num_workers=dist_env_params['num_workers'],
                                        use_gpu=dist_env_params['use_gpu'])
    trainer = pl.Trainer(**ed_params, 
                    callbacks=callbacks, 
                    logger=logger,
                    #strategy=strategy
                    accelerator='gpu',
                    num_sanity_val_steps=0,
                    log_every_n_steps=5)

    trainer.fit(model, dataset)
    model.save_model()
    #model.decoder.finalize()

    
