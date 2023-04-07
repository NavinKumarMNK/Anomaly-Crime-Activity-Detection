'@Author:NavinKumarMNK'
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import utils.utils as utils
import tensorrt as trt
import transformers as trf
from swin_transformer import SwinTransformer
from timm.models.layers import DropPath, to_2tuple
from timm.models.swin_transformer import SwinTransformerBlock, SwinTransformer
import timm

import torch.nn as nn
import timm

class SwinTransformerModel(nn.Module):
    def __init__(self, latent_dim=1024, img_size=224):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        self.fc = nn.Linear(21841, latent_dim)
        self.bc = nn.BatchNorm1d(21841)
        self.dropout = nn.Dropout(0.2)
        self.example_input_array = torch.randn(1, 3, 224, 224)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.bc(x)
        x = self.fc(x)
        return x

class EncoderSwimTransformer(pl.LightningModule):
    def __init__(self, weights_path="/weights/EncoderSwimTransformer"):
        super(EncoderSwimTransformer, self).__init__()
        self.model = SwinTransformerModel()
        self.weights_path = weights_path
        self.example_input_array = torch.randn(1, 3, 224, 224)
        self.example_output_array = torch.randn(1, 1024)
        self.save_hyperparameters()
        self.best_val_loss = None
        try:
            self.model = torch.load(utils.ROOT_PATH + self.weights_path + '.pt')
        except FileNotFoundError:
            torch.save(self.model, utils.ROOT_PATH + self.weights_path + '.pt')
        
    def forward(self, x):
        x = self.model(x)
        return x

    def feature_extractor(self, x):
        x = self.model(x)
        return x

    def predict_step(self, batch, batch_idx):
        x1 = self.model(batch)
        return x1

    def save_model(self):
        torch.save(self.model, utils.ROOT_PATH + '/weights/EncoderSwimTransformer.pt')
        '''self.to_onnx(utils.ROOT_PATH + self.weights_path +"_onnx.onnx", self.example_input_array, 
                        export_params=True,  opset_version=12, dtype=torch.float32, 
                        do_constant_folding=True, input_names=['input'], output_names=['output'], 
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        self.to_tensorrt()        
        '''
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def finalize(self):
        self.save_model()
        #self.to_torchscript(utils.ROOT_PATH + self.weights_path +"_script.pt", method="script", example_inputs=self.example_input_array)         
        self.to_onnx(utils.ROOT_PATH + self.weights_path +"_onnx.onnx", self.example_input_array.half(), export_params=True)
        self.to_tensorrt()        

    def to_tensorrt(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1
            with open(utils.ROOT_PATH + self.weights_path+'_onnx.onnx', 'rb') as model:
                parser.parse(model.read())
            
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024*1024*1024)
            config.set_flag(trt.BuilderFlag.FP16)

            network.get_input(0).shape = self.example_input_array.shape
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            with open(utils.ROOT_PATH + self.weights_path+'.trt', 'wb') as f:
                f.write(engine.serialize()) 
    
if __name__ == '__main__':
    model = EncoderSwimTransformer('/weights/EncoderSwimTransformer')
    print(model.count_parameters())
    #model.finalize()

    inp = torch.randn(2, 3, 224, 224)
    out = model(inp)
    print(out.shape)
