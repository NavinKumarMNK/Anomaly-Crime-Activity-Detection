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
import pytorch_lightning as pl

# classifier
class Classifier(nn.Module):
    def __init__(self, no_of_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, no_of_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)

class EfficientNetb3Classifier(pl.LightningModule):
    def __init__(self):
        super(EfficientNetb3Classifier, self).__init__()
        params = utils.config_parse('GENERAL')
        self.no_of_classes = int(params['no_of_classes'])
        self.model = Classifier(self.no_of_classes)
        self.model.load_state_dict(torch.load(utils.ROOT_PATH + '/weights/EfficientNetb3Classifier.pt'))

    def forward(self, x):
        x = x.view(-1, 1536)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat
    

if __name__ == '__main__':
    model = EfficientNetb3Classifier()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
        
