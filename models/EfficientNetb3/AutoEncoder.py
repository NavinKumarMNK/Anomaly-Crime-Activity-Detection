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
from models.EfficientNetb3.Encoder import EfficientNetb3Encoder
from models.EfficientNetb3.Decoder import EfficientNetb3Decoder
import ray_lightning as rl

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
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        loss, y_hat, y = outputs["loss"], outputs["y_hat"], outputs["y"]
        avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        self.log('train/loss_epoch', avg_loss)
        self.log('train/acc_epoch', torchmetrics.functional.accuracy(y_hat, y))
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
    
    def validation_epoch_end(self, outputs)-> None:
        loss, y_hat, y = outputs["loss"], outputs["y_hat"], outputs["y"]
        avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        self.log('val/loss_epoch', avg_loss)
        self.log('val/acc_epoch', torchmetrics.functional.accuracy(y_hat, y))
        # validation loss is less than previous epoch then save the model
        if (avg_loss < self.best_val_loss):
            self.best_val_loss = avg_loss
            self.save_model()

    def save_model(self):
        dummy_input = torch.randn(1, self.input_size, 256, 256)
        torch.onnx.export(self, dummy_input, self.weights_save_path+'.onnx', verbose=True, input_names=['input'], output_names=['output'])
        torch.save(self.state_dict(), self.weights_save_pat + '.pt')
        artifact = wandb.Artifact('lrcn_model.cpkt', type='model')
        artifact.add_file(self.weights_save_path+'.onnx')
        wandb.run.log_artifact(artifact)

    def print_params(self): 
        print("Model Parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
    
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
    
    def training_epoch_end(self, outputs) -> None:
        torch.save(self.encoder.state_dict(), utils.ROOT_PATH + '/weights/EfficientNetb3Encoder.pt')
        torch.save(self.decoder.state_dict(), utils.ROOT_PATH + '/weights/EfficientNetb3Decoder.pt')

class LRCNLogger(pl.Callback):
    def __init__(self, model:AutoEncoder, data:AutoEncoderDataset) -> None:
        super(LRCNLogger, self).__init__()
        self.model = model
        self.data = data

    def on_train_start(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_train_epoch_end(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_test_end(self, trainer, pl_module):
        wandb.watch(self.model, log="all")

    def on_validation_epoch_end(self, trainer, pl_module):
        logits = pl_module(self.data.val_dataloader())
        preds = torch.argmax(logits, dim=1)
        print("Logging validation metrics")
        trainer.logger.experiment.log({
            "val_acc": torchmetrics.functional.accuracy(preds, self.data.val_y),
            "val_auc": torchmetrics.functional.auc(preds, self.data.val_y),
            "examples" : [wandb.Video(video, fps=4, format="mp4", caption= f"Pred: {pred} - Label: {label}")
                            for video, pred, label in zip(self.data.val_x, preds, self.data.val_y)],
            "global_step": trainer.global_step,          
            }, step=trainer.global_step)


if __name__ == '__main__':
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='AutoEncoder', name='EfficientNetb3')

    import wandb
    wandb.init()
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
    trainer = Trainer(**autoencoder_params, 
                    callbacks=callbacks, 
                    logger=logger,
                    strategy=strategy)
    trainer.fit(model, train_dataloader, val_dataloader)

    

