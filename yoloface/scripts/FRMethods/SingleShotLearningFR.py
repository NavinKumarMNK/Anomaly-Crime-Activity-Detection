'@Author: NavinKumarMNK'
import sys 
if '../' not in sys.path:
    sys.path.append('../../')

import numpy as np
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule
from scripts.FaceNet import FaceNet
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleShotLearningFR(LightningModule):
    def __init__(self, embedding_size=512, pretrained: bool = True):
        super(SingleShotLearningFR, self).__init__()
        self.facenet = FaceNet(model='resnet101', pretrained=True, im_size=64)
        self.embedding_size = embedding_size
        try:
            if(pretrained == True):
                path = utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt'
                self.load_state_dict(torch.load(path))        
        except Exception as e:
            print("Pretrained weights not found")
                        
    def forward(self, x):
        x = self.facenet(x)
        return x

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + margin)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        anchor = self.forward(anchor)
        positive = self.forward(positive)
        negative = self.forward(negative)
        #print(anchor.shape, positive.shape, negative.shape)
        loss = self.triplet_loss(anchor, positive, negative)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_end(self) -> None:
        torch.save(self.state_dict(), utils.ROOT_PATH + '/weights/ssl_facenet_weights.pt')

class TripletFaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels
 
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        anchor_image = image
        anchor_label = self.labels[idx]

        # find positive image (same label)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = torch.randint(0, len(self.image_paths), (1,)).item()
        positive_image = Image.open(self.image_paths[positive_idx])
        if self.transform:
            positive_image = self.transform(positive_image)
        positive_label = self.labels[positive_idx]
        
        # find negative image (different label)
        negative_idx = idx
        while negative_idx == idx or self.labels[negative_idx] == anchor_label:
            negative_idx = torch.randint(0, len(self.image_paths), (1,)).item()
        negative_image = Image.open(self.image_paths[negative_idx])
        if self.transform:
            negative_image = self.transform(negative_image)
        negative_label = self.labels[negative_idx]

        return anchor_image, positive_image, negative_image

class SSLFacentDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, label_map, batch_size=32):
        super(SSLFacentDataModule, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.label_map = label_map
        self.count = 0
        self.transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = []
        self.labels = []
        
    def prepare_data(self, stage=None):
        with open(self.label_map) as f:
            self.label_map = json.load(f)
        for label_name in os.listdir(self.root_dir):
            label = self.label_map[label_name]
            label_dir = os.path.join(self.root_dir, label_name)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def train_dataloader(self):
        self.train_dataset = TripletFaceDataset(self.image_paths, self.labels, self.transform_train)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        self.val_dataset = TripletFaceDataset(self.image_paths, self.labels, self.transform_val)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    model = SingleShotLearningFR(pretrained=True)
    data = SSLFacentDataModule(root_dir=utils.ROOT_PATH+ "/database/faces",
                            label_map=utils.ROOT_PATH+'/database/label_map.json',
                            batch_size=2)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
    trainer.fit(model, data)
    trainer.save_checkpoint(utils.ROOT_PATH + '/weights/ssl_facenet.ckpt')