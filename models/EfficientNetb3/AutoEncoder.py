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
import wandb
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from models.EfficientNetb3.Encoder import EfficientNetb3Encoder
from models.EfficientNetb3.Decoder import EfficientNetb3Decoder
import ray_lightning as rl
from models.EfficientNetb3.AutoEncoderDataset import AutoEncoderDataModule
    
class AutoEncoder(pl.LightningModule):
    def __init__(self, 
                    ) -> None:
        super(AutoEncoder, self).__init__()
        self.example_input_array = torch.zeros(1, 3, 256, 256)
        self.save_hyperparameters()
        self.encoder = EfficientNetb3Encoder()
        self.decoder = EfficientNetb3Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(1), x.size(2), x.size(3), x.size(4))
        y = y.view(y.size(1), y.size(2), y.size(3), y.size(4))
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return {"loss" : loss}

    def training_epoch_end(self, outputs):
        loss, y_hat, y = outputs["loss"], outputs["y_hat"], outputs["y"]
        avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        self.log('train/loss_epoch', avg_loss)
        self.log('train/acc_epoch', torchmetrics.functional.accuracy(y_hat, y))
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(1), x.size(2), x.size(3), x.size(4))
        y = y.view(y.size(1), y.size(2), y.size(3), y.size(4))
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
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
        self.encoder.save_model()
        self.decoder.save_model()
        artifact = wandb.Artifact('auto_encoder_model.cpkt', type='model')
        #wandb.run.log_artifact(artifact)

    def print_params(self): 
        print("Model Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(1), x.size(2), x.size(3), x.size(4))
        y = y.view(y.size(1), y.size(2), y.size(3), y.size(4))
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
        return {"test_loss": loss, "y_hat": y_hat, "y": y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
        
    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat
        
if __name__ == '__main__':
    #from pytorch_lightning.loggers import WandbLogger
    #logger = WandbLogger(project='CrimeDetection', name='AutoEncoder')

    #import wandb
    #wandb.init()

    print("hello")
    import ray
    
    ray.init(runtime_env={"working_dir": utils.ROOT_PATH})
    
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')

    annotation_train = utils.dataset_image_autoencoder(dataset_params['data_path'])
    dataset = AutoEncoderDataModule(**dataset_params, annotation_train=annotation_train)

    dataset.setup()
    
    from pytorch_lightning.callbacks import ModelSummary
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    device_monitor = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=utils.ROOT_PATH + '/weights/checkpoints/autoencoder/',
                                            monitor="val_loss", mode='min', save_last=True, every_n_train_steps=1000)
    model_summary = ModelSummary(max_depth=3)
    refresh_rate = TQDMProgressBar(refresh_rate=10)

    callbacks = [
        model_summary,
        refresh_rate,
        checkpoint_callback,
        early_stopping,
        device_monitor
    ]
    print("hello")
    model = AutoEncoder()
    autoencoder_params = utils.config_parse('AUTOENCODER_TRAIN')
    print(autoencoder_params)
    print(torch.cuda.device_count())
    dist_env_params = utils.config_parse('DISTRIBUTED_ENV')
    strategy = None
    if int(dist_env_params['horovod']) == 1:
        strategy = rl.HorovodRayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'])
    elif int(dist_env_params['model_parallel']) == 1:
        strategy = rl.RayShardedStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'])
    elif int(dist_env_params['data_parallel']) == 1:
        strategy = rl.RayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'])
    print("hello")

    trainer = pl.Trainer(**autoencoder_params, 
                    callbacks=callbacks, 
                    strategy=strategy,
                    #accelerator='gpu',
                    
                    )
                    
    '''
    trainer = Trainer(**autoencoder_params, 
                    callbacks=callbacks, 
                    strategy='deepspeed',
                    accelerator='gpu',
                    num_nodes=6,
                    )
    '''

    print("hello")
    trainer.fit(model, dataset)


    #model.encoder.finalize()    

    trainer.test(model, dataset.test_dataloader())
    #wandb.finish()

