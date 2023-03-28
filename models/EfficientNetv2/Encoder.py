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
import torch.nn as nn
import tensorrt as trt
import onnx
from torchvision import models
        

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b3(include_top=False, pretrained=False)
        self.model.classifier = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

# Encoder
class EfficientNetv2Encoder(pl.LightningModule):
    def __init__(self):
        super(EfficientNetv2Encoder, self).__init__()
        self.file_path = utils.ROOT_PATH + '/weights/EfficientNetv2Encoder'
        self.example_input_array = torch.rand(1, 3, 256, 256)
        self.example_output_array = torch.rand(1, 1280)
        self.save_hyperparameters()
        self.model = Encoder()
        try:
            self.model = torch.load(utils.ROOT_PATH + '/weights/EfficientNetv2Encoder.pt')
            print("Encoder Model Found")
        except Exception as e:
            self.model = models.efficientnet_v2_s(include_top=False, weights='EfficientNet_V2_S_Weights.DEFAULT')
            self.model.classifier = nn.Identity()
            torch.save(self.model, utils.ROOT_PATH + '/weights/EfficientNetv2Encoder.pt')

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

    def save_model(self):
        print("Saving Model")
        torch.save(self.model, self.file_path+'.pt')

    def finalize(self):
        self.save_model()
        self.to_torchscript(self.file_path+'_script.pt', method='script', example_inputs=self.example_input_array)
        self.to_onnx(self.file_path+'.onnx', self.example_input_array, export_params=True)
        self.to_tensorrt()

    def to_tensorrt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1
            with open(self.file_path+'.onnx', 'rb') as model:
                parser.parse(model.read())
            
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)

            network.get_input(0).shape = [1, 3, 256, 256]
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            with open(self.file_path+'.trt', 'wb') as f:
                f.write(engine.serialize())   
                
if __name__ == '__main__':
    model = EfficientNetv2Encoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    model.finalize()
        
