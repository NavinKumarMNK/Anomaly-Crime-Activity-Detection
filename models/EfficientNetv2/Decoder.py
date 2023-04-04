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
from torchvision import models
import wandb
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Initial representation
        self.fc = nn.Linear(1024, 2*2*1024)
        self.bn1d = nn.BatchNorm1d(2*2*1024)
        self.gelu = nn.GELU()

        # Decoder layers
            
        self.conv1 = nn.ConvTranspose2d(1024, 768, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(768)
        self.relu1 = nn.GELU()

        self.conv9 = nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.GELU()

        self.conv12 = nn.ConvTranspose2d(512, 512, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.GELU()


        self.conv11 = nn.ConvTranspose2d(512, 384, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn11 = nn.BatchNorm2d(384)
        self.relu11 = nn.GELU()

        self.conv2 = nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.GELU()

        self.conv10 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.bn10 = nn.BatchNorm2d(256)
        self.relu10 = nn.GELU()

        self.conv3 = nn.ConvTranspose2d(256, 192, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(192)
        self.relu3 = nn.GELU()
        
        self.conv4 = nn.ConvTranspose2d(192, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.GELU()

        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.GELU()

        self.conv6 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.GELU()

        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.GELU()

        # Residual blocks with SE attention
        self.res1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            SEAttention(256),
            nn.ReLU()
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            SEAttention(64),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.25)
        
        self.conv8 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1d(x)
        x = self.gelu(x)

        x = x.view(x.size(0), 1024, 2, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        x = self.dropout(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.res1(x) + x
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.res2(x) + x
        x = self.dropout(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)

        x = self.tanh(x)

        return x

class EfficientNetv2Decoder(pl.LightningModule):
    def __init__(self):
        super(EfficientNetv2Decoder, self).__init__()
        self.model = Decoder()
        try:
            self.model = torch.load(utils.ROOT_PATH + '/weights/EfficientNetv2DecoderLarge.pt')
            print("Decoder Weights Found")
        except Exception as e:
            torch.save(self.model, utils.ROOT_PATH + '/weights/EfficientNetv2DecoderLarge.pt')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def save_model(self):
        torch.save(self.model, utils.ROOT_PATH + '/weights/EfficientNetv2DecoderLarge.pt')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

if __name__ == '__main__':
    model = EfficientNetv2Decoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    
    inp = torch.randn(2, 1024)
    out = model(inp)
    print(out.shape)