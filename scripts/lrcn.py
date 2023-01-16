"@Author: NavinKumarMNK"
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
import utils.utils as utils

# Congifuration file
import configparser
config = configparser.ConfigParser()
config.read('../model.cfg')
lrcn_config = dict(config.items(['LRCN']))

# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from scripts.encoder import CNNModel as encoder


# LRCN model
class LRCN(pl.LightningModule):
    def __init__(self, input_size, encoder_output_size, hidden_size, 
                    num_layers, num_classes, pretrained=True):
        super(LRCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        
        # Resnet34 for CNN feature extraction
        if (pretrained == True):
            self.base_model = models.resnet34(pretrained=True)
        else:
            self.base_model = encoder
        # Support the training of the base model
        for param in self.base_model.parameters():
            param.requires_grad = True

        # LSTM on top of Base Model
        self.lstm = nn.LSTM(input_size=self.encoder_output_size, 
                    hidden_size=self.hidden_size, num_layers=num_layers, 
                    batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        pass



if __name__ == '__main__':
    model = LRCN()
    print(model)