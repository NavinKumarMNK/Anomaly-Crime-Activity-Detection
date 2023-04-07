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
from models.EfficientNetv2.VarEncoder import EfficientnetV2VarEncoder
from models.EfficientNetv2.Decoder import EfficientNetv2Decoder
import ray_lightning as rl
from models.EfficientNetv2.AutoEncoderDataset import AutoEncoderDataModule
from pytorch_lightning import Callback
import time
from pytorch_lightning import Trainer
import ray
from pytorch_lightning.loggers import WandbLogger
import wandb
import tensorrt as trt

pl.seed_everything(42)
    
class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, 
                    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.example_input_array = torch.zeros(1, 3, 256, 256).half()
        self.save_hyperparameters()
        self.encoder = EfficientnetV2VarEncoder()
        self.decoder = EfficientNetv2Decoder()
        self.encoder.train()
        self.decoder.train()
        self.latent_dim = 1024
        self.beta = 0
        self.lr = 1e-4
        # xaiver initialization
        for param in self.encoder.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
        for param in self.decoder.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        try:
            mu, var = self.encoder(x)
            z = self.encoder.reparameterize(mu, var)
            x = self.decoder(z)
        except Exception as e:
            print(e, "Error!")
            x = torch.rand(4, 3, 224, 224)
            mu, var = self.encoder(x)
            z = self.encoder.reparameterize(mu, var)
            x = self.decoder(z)
        
        return x, mu, var

    def loss_function(self, recon_x, x, mu, logvar):
        MAE = nn.functional.l1_loss(recon_x, x, reduction='none')
        MAE = MAE.view(MAE.size(0), -1).mean(dim=1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.beta > 1:
            self.beta = 1
        self.log("loss/MAE_loss", MAE.mean())
        self.log("loss/kld_loss", KLD.mean())
        loss = MAE + self.beta*KLD
        loss = loss.mean()
        self.log("loss/beta", self.beta)
        self.log("Total loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.beta += 0.0001
        #x = x.view(x.size(1), x.size(2), x.size(3), x.size(4)).half()
        #y = y.view(y.size(1), y.size(2), y.size(3), y.size(4)).half()
        x_hat, mu, log_var = self(x)
        loss = self.loss_function(x_hat, y, mu, log_var)
        self.log('train_loss', loss)
        if batch_idx % 1000 == 0:
            self.log_image(x, x_hat, y)
        return {"loss" : loss}

    def training_epoch_end(self, outputs)-> None:
        loss = outputs[0]['loss']
        try:
            avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        except TypeError:
            avg_loss = loss
        self.log('train/loss_epoch', avg_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(1), x.size(2), x.size(3), x.size(4)).half()
        #y = y.view(y.size(1), y.size(2), y.size(3), y.size(4)).half()
        x_hat, mu, log_var = self(x)
        loss = self.loss_function(x_hat, y, mu, log_var)
        self.log('val_loss', loss)
        if batch_idx % 100 == 0:
            self.log_image(x, x_hat, y) 
        return {"val_loss": loss, "y_hat": x_hat, "y": y}

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(1), x.size(2), x.size(3), x.size(4)).half()
        #y = y.view(y.size(1), y.size(2), y.size(3), y.size(4)).half()
        x_hat, mu, log_var = self(x)
        loss = self.loss_function(x_hat, y, mu, log_var)
        self.log('test_loss', loss)
        return {"test_loss": loss, "y_hat": x_hat, "y": y}

    def save_model(self):
        self.encoder.save_model()
        self.decoder.save_model()
        artifact = wandb.Artifact('auto_encoder_model.cpkt', type='model')
        wandb.run.log_artifact(artifact)

    def print_params(self): 
        print("Model Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return [optimizer]
        
    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def log_image(self, x, x_hat, y):
        x = (x + 1) / 2
        x_hat = (x_hat + 1) / 2
        y = (y + 1) / 2

        x = x.to('cuda')
        x_hat = x_hat.to('cuda')
        y = y.to('cuda')

        # Apply inverse normalization to convert back to RGB
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).to('cuda')
        x = (x * std + mean).clamp(0, 1)
        x_hat = (x_hat * std + mean).clamp(0, 1)
        y = (y * std + mean).clamp(0, 1)

        # Log input, output, and target images
        self.logger.experiment.log({
                'input_images': [wandb.Image(x)],
                'output_images': [wandb.Image(x_hat)],
                'target_images': [wandb.Image(y)],
        })


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
            f"Average Peak memory  {max_memory.item() / float(world_size):.2f}"
            f"MiB")

def train():
    #ray.init(runtime_env={"working_dir": utils.ROOT_PATH})
    dataset_params = utils.config_parse('AUTOENCODER_DATASET')


    annotation = utils.dataset_image_autoencoder(
                            dataset_params['data_path'], "anomaly_train.txt")
    dataset = AutoEncoderDataModule(**dataset_params, 
                    annotation=annotation)

    dataset.setup()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    device_monitor = DeviceStatsMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=utils.ROOT_PATH + '/weights/checkpoints/vae/',
                                            monitor="val_loss", mode='min', every_n_train_steps=100, save_top_k=1, save_last=True)
    refresh_rate = TQDMProgressBar(refresh_rate=10)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        refresh_rate,
        checkpoint_callback,
        early_stopping,
        device_monitor,
        lr_monitor,
        #CUDACallback()
    ]

    model = VariationalAutoEncoder()
    autoencoder_params = utils.config_parse('AUTOENCODER_TRAIN')
    dist_env_params = utils.config_parse('DISTRIBUTED_ENV')
    strategy = None
    if int(dist_env_params['horovod']) == 1:
        strategy = rl.HorovodRayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    elif int(dist_env_params['deep_speed']) == 1:
        strategy = 'deepspeed_stage_1'
    elif int(dist_env_params['model_parallel']) == 1:
        strategy = rl.RayShardedStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    elif int(dist_env_params['data_parallel']) == 1:
        strategy = rl.RayStrategy(use_gpu=dist_env_params['use_gpu'],
                                        num_workers=dist_env_params['num_workers'],
                                        num_cpus_per_worker=dist_env_params['num_cpus_per_worker'])
    
    trainer = Trainer(**autoencoder_params, 
                callbacks=callbacks, 
                strategy='deepspeed_stage_1',
                accelerator='gpu',
                logger=logger,
                num_sanity_val_steps=0,
                #resume_from_checkpoint=utils.ROOT_PATH + '/weights/checkpoints/vae/last.ckpt',
                log_every_n_steps=5
                )
    
    trainer.fit(model, dataset)
    model.save_model()
   
    model.encoder.finalize()

if __name__ == '__main__':
    
    logger = WandbLogger(project='CrimeDetection3', name='VariationalAutoEncoder')
    wandb.init()
    
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor
    from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

    train()

    wandb.finish()
