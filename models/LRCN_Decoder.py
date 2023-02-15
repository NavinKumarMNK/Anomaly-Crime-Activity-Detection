"@Author: NavinKumarMNK"
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
import utils.utils as utils

# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import wandb
import torchmetrics
from models.Dataset import CrimeActivityLRCNDataset
import tensorrt as trt
from models.EfficientNetb3.Encoder import EfficientNetb3Encoder as Encoder
# LRCN model
class LRCN(pl.LightningModule):
    def __init__(self, input_size:int, encoder_output_size:int, hidden_size:int, 
                    num_layers:int, num_classes:int, weights_save_path,
                    pretrained:bool=True, learning_rate:float=0.0001
                    ) -> None:
        super(LRCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        self.learning_rate = learning_rate
        self.weights_save_path = weights_save_path

        # Resnet34 for CNN feature extraction
        self.base_model = Encoder()
        # Support the training of the base model : Fine Tuning
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.encoder_output_size, 
                    hidden_size=self.hidden_size, num_layers=num_layers, 
                    batch_first=True)
        self.td = nn.utils.rnn.PackedSequence(self.lstm, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.out = nn.Softmax(dim=1)

        self.best_val_loss = 1e5

    def forward(self, x:torch.Tensor):
        # video => (batch_size, frames, channels, height, width)
        # split the video into frames        
        windows = torch.split(x, dim=1)

        # initialize hidden state and cell state
        h = torch.zeros(1, x.size(0), self.hidden_size)
        c = torch.zeros(1, x.size(0), self.hidden_size)

        outputs = []
        for i, window in iter(windows):
            features = self.base_model(window)
            features = features.view(features.size(0), -1, self.encoder_output_size)

            output, (h, c) = self.td(features.unsqueeze(1), (h, c))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc(outputs)
        outputs = self.out(outputs)
        return outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_epoch_end(self, outputs):
        loss, y_hat, y = outputs["loss"], outputs["y_hat"], outputs["y"]
        avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        self.log('train/loss_epoch', avg_loss)
        self.log('train/acc_epoch', torchmetrics.functional.accuracy(y_hat, y))
        self.log('train/auc_epoch', torchmetrics.functional.auc(y_hat, y))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/acc', torchmetrics.functional.accuracy(y_hat, y))
        self.log('val/auc', torchmetrics.functional.auc(y_hat, y))
        return loss
    
    def validation_epoch_end(self, outputs) -> None:
        loss, y_hat, y = outputs["loss"], outputs["y_hat"], outputs["y"]
        avg_loss = torch.stack([x['loss'] for x in loss]).mean()
        self.log('val/loss_epoch', avg_loss)
        self.log('val/acc_epoch', torchmetrics.functional.accuracy(y_hat, y))
        self.log('val/auc_epoch', torchmetrics.functional.auc(y_hat, y))
        # validation loss is less than previous epoch then save the model
        if (avg_loss < self.best_val_loss):
            self.best_val_loss = avg_loss
            self.save_model()

    def save_model(self):
        #dummy input for a video of  batch size, n frames, each frame of input_size,
        dummy_input = torch.randn(1, self.input_size, 224, 224)
        torch.onnx.export(self, dummy_input, self.weights_save_path+'.onnx', verbose=True, input_names=['input'], output_names=['output'])
        torch.save(self.state_dict(), self.weights_save_pat + '.pt')
        artifact = wandb.Artifact('lrcn_model.cpkt', type='model')
        artifact.add_file(self.weights_save_path+'.onnx')
        wandb.run.log_artifact(artifact)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/acc', torchmetrics.functional.accuracy(y_hat, y))
        self.log('test/auc', torchmetrics.functional.auc(y_hat, y)) 
        return loss

    def print_params(self):
        return f'''LRCN(input_size={self.input_size}, 
                hidden_size={self.hidden_size}, 
                num_layers={self.num_layers}, 
                num_classes={self.num_classes}, 
                encoder_output_size={self.encoder_output_size}, 
                sample_rate={self.sample_rate}, 
                learning_rate={self.learning_rate})'''

    #inference function of pytorch lightning
    def on_predict_start(self) -> None:
        # convert the model into tensorrt and optimize the inferecne usign torch_tensorrt
        self.tensorrt = trt(self, max_batch_size=1, max_workspace_size=1<<25,
                                    precision_mode="FP16", use_onnx=True)  
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.tensorrt(x)
        return y_hat

    def predict_epoch_end(self, outputs) -> None:
        return outputs
    
class LRCNLogger(pl.Callback):
    def __init__(self, model:LRCN, data:CrimeActivityLRCNDataset) -> None:
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
    wandb.init()
    logger = pl.loggers.WandbLogger(project="CrimeDetection-LRCN", entity="mnk")
    trainer_params = utils.config_parse('LRCN_TRAIN')
    trainer = pl.Trainer(**trainer_params, logger=logger,    
                        )
    model_params = utils.config_parse('LRCN_MODEL')
    model = LRCN(**model_params)
    trainer.fit(model)
    data = CrimeActivityLRCNDataset()
    trainer.test(datamodule=data)
    wandb.finish()
    