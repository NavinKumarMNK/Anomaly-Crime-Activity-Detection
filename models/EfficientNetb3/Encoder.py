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
import wandb
import torch.nn as nn
import tensorrt as trt
import onnx

# Encoder
class EfficientNetb3Encoder(pl.LightningModule):
    def __init__(self):
        super(EfficientNetb3Encoder, self).__init__()
        self.file_path = utils.ROOT_PATH + '/weights/EfficientNetb3Encoder'
        self.example_input_array = torch.rand(1, 3, 256, 256)
        self.example_output_array = torch.rand(1, 1536)
        self.model = torch.load(utils.ROOT_PATH + '/weights/EfficientNetb3Encoder.pt')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def finalize(self):
        self.to_onnx(export_params=True)
        self.to_torchscript()
        self.to_tensorrt()

    def to_tensorrt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1
            with open(self.file_path+'.onnx', 'rb') as model:
                parser.parse(model.read())
            
            config = builder.create_builder_config()
            config.set_memory_pool_limit = 1 << 30
            config.set_flag(trt.BuilderFlag.FP16)

            network.get_input(0).shape = [1, 3, 256, 256]
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            with open(self.file_path+'.trt', 'wb') as f:
                f.write(engine.serialize())   

    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def to_onnx(self, **kwargs):
        return super().to_onnx(self.file_path+'.onnx', self.example_input_array, **kwargs)
    
    def to_torchscript(self, method='script', **kwargs):
        return super().to_torchscript(self.file_path+'_script.pt', method, self.example_input_array, **kwargs)


if __name__ == '__main__':
    model = EfficientNetb3Encoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    model.finalize()
        