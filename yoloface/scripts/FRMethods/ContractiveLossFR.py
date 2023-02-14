'@Author: NavinKumarMNK' 
import sys 
if '../' not in sys.path:
    sys.path.append('../../')
    
from utils import utils
import os
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContractiveLossFR(LightningModule):
    def __init__(self, embedding_size:int, num_classes:int, 
                    margin:float, easy_margin:bool, pretrained:bool):
        super(ContractiveLossFR, self).__init__()
        self.embedding_size = int(embedding_size)
        self.num_classes = num_classes
        self.margin = float(margin)
        self.easy_margin = easy_margin
        self.embedding = nn.Linear(self.embedding_size, 
                                self.num_classes).double().to("cuda")
        
        try:
            if(pretrained == True):
                path = utils.ROOT_PATH + '/weights/contractive_loss_weights.pt'
                self.load_state_dict(torch.load(path))        
        except Exception as e:
            print(path)
            print(e)
            
    def forward(self, embeddings:torch.DoubleTensor):
        embeddings = embeddings.double()
        logits = self.embedding(embeddings.to("cuda"))
        return logits

    def contrastive_loss(self, logits, labels):
        # calculate the similarity between embeddings
        similarity = torch.norm(logits, dim=1, p=2)
        diagonal = similarity.diag()
        cost_s = torch.clamp(self.margin - diagonal + similarity, min=0)
        mask = torch.eye(similarity.size(0), device=DEVICE) > .5
        I = torch.eye(similarity.size(0), device=DEVICE, dtype=torch.bool)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost = cost_s.max()
        return cost

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self.forward(embeddings)
        loss = self.contrastive_loss(logits, labels)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def prediction_step(self, batch):
        embeddings = batch
        logits = self.forward(embeddings)
        predictions = logits.argmax(1)
        return {"predictions": predictions}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def on_train_end(self) -> None:
        torch.save(self.state_dict(), utils.ROOT_PATH + '/weights/contractive_loss_weights.pt')

class ContractiveLossFREmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class ContractiveLossFREmbeddingsDataModule(pl.LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 1):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        self.labels = df.iloc[:, 0].values
        self.embeddings = df.iloc[:, 1:].values
        # convert to numpy array
        self.labels =   torch.Tensor(self.labels).long()
        self.embeddings = torch.tensor(self.embeddings)
        self.train_embeddings, self.val_embeddings, self.train_labels, self.val_labels = train_test_split(self.embeddings, self.labels, test_size=0.2, random_state=42)
    
    def train_dataloader(self):
        train_dataset = ContractiveLossFREmbeddingsDataset(self.train_embeddings, self.train_labels)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset = ContractiveLossFREmbeddingsDataset(self.val_embeddings, self.val_labels)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4) 
    
if __name__ == "__main__":
    args = utils.config_parse('CONTRACTIVE_LOSS_FR')
    args['num_classes'] = len(os.listdir(utils.ROOT_PATH + '/database/faces')) 
    model = ContractiveLossFR(**args, pretrained=True)
    data = ContractiveLossFREmbeddingsDataModule(utils.ROOT_PATH + '/database/embeddings.csv', batch_size=1)
    data.prepare_data()
    trainer = pl.Trainer(
    accelerator='gpu', devices=1,
    min_epochs=10,
    max_epochs=25,
    )
    trainer.fit(model, data)
    trainer.save_checkpoint(utils.ROOT_PATH + '/weights/contractive_loss.cpkt')
