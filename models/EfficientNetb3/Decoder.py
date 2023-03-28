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
from torchvision.models import efficientnet_b3

import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
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
        self.fc = nn.Linear(1280, 4*4*768)
        self.bn1d = nn.BatchNorm1d(4*4*768)
        self.gelu = nn.GELU()

        # Decoder layers
        
        self.conv1 = nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(16)
        self.relu6 = nn.ReLU()


        # Residual blocks with SE attention
        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            SEAttention(32),
            nn.ReLU()
        )
        
        self.conv7 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1d(x)
        x = self.gelu(x)
        x = x.view(-1, 768, 4, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.res1(x) + x

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.sigmoid(x)

        return x
class DecoderLarge(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Initial representation
        self.fc = nn.Linear(1280, 4*4*1024)
        self.bn1d = nn.BatchNorm1d(4*4*1024)
        self.gelu = nn.GELU()

        # Decoder layers
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        # Residual blocks with SE attention
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

        
        self.conv7 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1d(x)
        x = self.gelu(x)
        x = x.view(-1, 1024, 4, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.res1(x) + x


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.res2(x) + x

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.sigmoid(x)

        return x
    


class EfficientNetb3Decoder(pl.LightningModule):
    def __init__(self):
        super(EfficientNetb3Decoder, self).__init__()
        self.model = DecoderLarge()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def save_model(self):
        torch.save(self.model.state_dict(), utils.ROOT_PATH + '/weights/EfficientNetb3DecoderLarge.pt')

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
    model = EfficientNetb3Decoder()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    