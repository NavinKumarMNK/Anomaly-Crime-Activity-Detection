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
from pytorch_lightning import Callback
import time
from pytorch_lightning import Trainer
import ray
from pytorch_lightning.loggers import WandbLogger
import wandb

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
        loss = outputs[0]['loss']
        try:
            avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        except TypeError:
            avg_loss = loss
        self.log('train/loss_epoch', avg_loss)
        
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
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
        return {"test_loss": loss, "y_hat": y_hat, "y": y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        
    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(
            max_memory, dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(
            epoch_time, dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(
            max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()
        print(
            f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} "
            f"seconds")
        print(
            f"Average Peak memory {max_memory.item() / float(world_size):.2f}"
            f"MiB")

def train():
    ray.init(runtime_env={"working_dir": utils.ROOT_PATH})
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')

    annotation_train = utils.dataset_image_autoencoder(
                            dataset_params['data_path'])
    dataset = AutoEncoderDataModule(**dataset_params, 
                    annotation_train=annotation_train)

    dataset.setup()
    
    print("hello")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    device_monitor = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=utils.ROOT_PATH + '/weights/checkpoints/autoencoder/',
                                            monitor="val_loss", mode='min', every_n_train_steps=100, save_top_k=1, save_last=True)
    model_summary = ModelSummary(max_depth=3)
    refresh_rate = TQDMProgressBar(refresh_rate=10)

    callbacks = [
        model_summary,
        refresh_rate,
        checkpoint_callback,
        early_stopping,
        device_monitor,
        #CUDACallback()
    ]

    model = AutoEncoder()
    autoencoder_params = utils.config_parse('AUTOENCODER_TRAIN')
    
    dist_env_params = utils.config_parse('DISTRIBUTED_ENV')
    strategy = None
    print("hello")
    if int(dist_env_params['horovod']) == 1:
        strategy = rl.HorovodRayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    elif int(dist_env_params['model_parallel']) == 1:
        print("model")
        strategy = rl.RayShardedStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    elif int(dist_env_params['data_parallel']) == 1:
        print("data")
        strategy = rl.RayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    
    trainer = Trainer(**autoencoder_params, 
                callbacks=callbacks, 
                #strategy=strategy,
                #accelerator='gpu',
                #logger=logger,
                num_sanity_val_steps=0,
                weights_save_path=utils.ROOT_PATH + '/weights/checkpoints/autoencoder/',
                enable_checkpointing=True,
                )

    #trainer.fit(model, dataset)
    
    model.encoder.finalize()

if __name__ == '__main__':
    
    #logger = WandbLogger(project='CrimeDetection', name='AutoEncoder')
    #wandb.init()
    #torch.distributed.init_process_group(backend='nccl', world_size=2, rank=0)

    from pytorch_lightning.callbacks import ModelSummary
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor
    
    

    train()
    