
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
from models.EfficientNetv2.Encoder import EfficientNetv2Encoder

class EfficientnetV2VarEncoder(pl.LightningModule):
    def __init__(self):
        super(EfficientnetV2VarEncoder, self).__init__()
        self.file_path = utils.ROOT_PATH + '/weights/EfficientNetv2VE'
        self.encoder = EfficientNetv2Encoder()
        self.latent_dim = 1280
        self.example_input_array = torch.rand(1, 3, 256, 256)
        self.example_output_array = torch.rand(1, 1280)
        self.save_hyperparameters()
        self.fc_mu = nn.Linear(1280, self.latent_dim)
        self.fc_var = nn.Linear(1280, self.latent_dim)

        try:
            torch.save(self, utils.ROOT_PATH + '/weights/' + 'VE.pt')
        except Exception as e:
            self.encoder = torch.load(utils.ROOT_PATH + '/weights/' + 'VE.pt').encoder
            self.fc_mu = torch.load(utils.ROOT_PATH + '/weights/' + 'VE.pt').fc_mu
            self.fc_var = torch.load(utils.ROOT_PATH + '/weights/' + 'VE.pt').fc_var

        try:
            self.fc_mu.load_state_dict(torch.load( utils.ROOT_PATH + '/weights/' + 'fc_mu.pth'))
        except Exception as e:
            torch.save(self.fc_mu.state_dict(),  utils.ROOT_PATH + '/weights/' + 'fc_mu.pth')

        try:
            self.fc_var.load_state_dict(torch.load(utils.ROOT_PATH + '/weights/' + 'fc_var.pth'))
        except Exception as e:
            torch.save(self.fc_var.state_dict(), utils.ROOT_PATH + '/weights/' + 'fc_var.pth')
        
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def save_model(self):
        torch.save(self, utils.ROOT_PATH + '/weights/' + 'VE.pt')
        torch.save(self.fc_mu.state_dict(), utils.ROOT_PATH + '/weights/' + 'fc_mu.pth')
        torch.save(self.fc_var.state_dict(), utils.ROOT_PATH + '/weights/' + 'fc_var.pth')
    
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
           
            config.max_workspace_size = 4*1024*1024*1024            

            network.get_input(0).shape = [1, 3, 256, 256]
            engine = builder.build_serialized_network(network, config)
            engine = builder.build_engine(network, config)
            with open(self.file_path+'.trt', 'wb') as f:
                f.write(engine.serialize())   

if __name__ == '__main__':
    model = EfficientnetV2VarEncoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    model.finalize()
        
