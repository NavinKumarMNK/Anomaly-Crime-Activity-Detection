"@Author: NavinKumarMNK"
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
import torchmetrics
import tensorrt as trt
from models.EfficientNetb3.Encoder import EfficientNetb3Encoder as Encoder
# LRCN model

class LSTMDecoder(pl.LightningModule):
    def __init__(self, encoder_output_size:int=1536, hidden_size:int=768, 
                    num_layers:int=3, num_classes:int=14) -> None:
        super().__init__()
        self.file_path = utils.ROOT_PATH + '/weights/LSTMDecoder'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_output_size = encoder_output_size
        self.example_input_array = torch.rand(1, self.encoder_output_size)
        self.example_output_array = torch.rand(1, num_classes)
        self.lstm = nn.LSTM(input_size=self.encoder_output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.save_hyperparameters()
        self.reset_hidden()
        try:
            self.lstm = torch.load(utils.ROOT_PATH + '/weights/LSTMDecoder.pt')
        except FileNotFoundError:
            torch.save(self.lstm, utils.ROOT_PATH + '/weights/LSTMDecoder.pt')

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h, c

    def reset_hidden(self):
        self.hidden=None

    def forward(self, x):
        # x.shape: (batch_size, seq_size=1, input_size)
        if self.hidden is None:
            self.hidden = self.init_hidden(x.shape[0])

        self.h, self.c = self.hidden
        out, (self.h, self.c) = self.lstm(x.unsqueeze(1), (self.h, self.c))
        out = self.fc(out[:, -1, :]) # only use the last timestep
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _= self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/acc', torchmetrics.functional.accuracy(y_hat, y))
        return loss

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
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024*1024*1024)
            config.set_flag(trt.BuilderFlag.FP16)

            network.get_input(0).shape = [1, 1536]
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            with open(self.file_path+'.trt', 'wb') as f:
                f.write(engine.serialize()) 
    
    def to_onnx(self, **kwargs):
        return super().to_onnx(self.file_path+'.onnx', self.example_input_array, **kwargs)
    
    def to_torchscript(self, method='script', **kwargs):
        return super().to_torchscript(self.file_path+'_script.pt', method, self.example_input_array, **kwargs)

if __name__ == '__main__': 
    model = LSTMDecoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    i = 0
    while i < 30:
        values = torch.randn(1, 1536)
        model(values)
        i+=1
        if(i==10):
            model.reset_hidden()

    print(model(values).shape)
    
    model.finalize()
