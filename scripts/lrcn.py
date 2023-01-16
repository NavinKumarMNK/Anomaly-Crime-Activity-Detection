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
from scripts.encoder import CNNModel as encoder    

# LRCN model
class LRCN(pl.LightningModule):
    def __init__(self, input_size:int, encoder_output_size:int, hidden_size:int, 
                    num_layers:int, num_classes:int, pretrained:bool=True, 
                    sample_rate:int =-1, learning_rate:float=0.0001) -> None:
        super(LRCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate

        # Resnet34 for CNN feature extraction
        if (pretrained == True):
            self.base_model = models.resnet34(pretrained=True)
        else:
            self.base_model = encoder
        # Support the training of the base model : Fine Tuning
        for param in self.base_model.parameters():
            param.requires_grad = True

        # LSTM on top of Base Model
        self.lstm = nn.LSTM(input_size=self.encoder_output_size, 
                    hidden_size=self.hidden_size, num_layers=num_layers, 
                    batch_first=True)
        
        # Time distributed LSTM
        self.td = nn.TimeDistributed(self.lstm)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # split the input into windows
        windows = torch.split(x, self.window_size, dim=1)

        # initialize hidden state and cell state
        h = torch.zeros(1, x.size(0), self.hidden_size)
        c = torch.zeros(1, x.size(0), self.hidden_size)

        # process each window
        outputs = []
        #iterate over the windows
        for i, window in iter(windows):
            # CNN feature extraction
            features = self.base_model(window)
            features = features.view(features.size(0), -1, self.encoder_output_size)

            # LSTM
            output, (h, c) = self.td(features.unsqueeze(1), (h, c))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.fc(outputs)
        return outputs
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    

if __name__ == '__main__': 
    model = LRCN(utils.ConfigParser('LRCN').parse())
    print(model)