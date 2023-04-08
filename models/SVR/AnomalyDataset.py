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
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import PIL
import numpy as np
from utils.preprocessing import ImagePreProcessing
from models.EfficientNetv2.VarEncoder import Efficientnetv2VarEncoder

class AnomalyDataset(Dataset):
    def __init__(self, batch_size:int,
                    data_path, annotation) -> None:
        super(AnomalyDataset, self).__init__()
        self.data_path = data_path
        self.annotation = open(self.data_path + annotation, 
                                        'r').read().splitlines()
        self.batch_size = int(batch_size)

        self.preprocessing = ImagePreProcessing()

        self.index = 0
        
    def __len__(self): 
        return len(self.annotation)
    
    def __getitem__(self, idx):
        i = 0
        while True:
            string = self.annotation[idx+i]
            lst = string.split('  ')

            label = float(utils.label_parser(lst[1]))
            start, end = int(lst[2]), int(lst[3])
            start2, end2 = int(lst[4]), int(lst[5])
            video_path = lst[0]
            video_path = os.path.join(self.data_path, lst[1], video_path) 

            cap = cv2.VideoCapture(video_path.strip())
            if not cap.isOpened():
                i+=1
                print("Error opening video stream or file")
                continue

            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < self.batch_size+1:
                i+=1
                continue

            frame_no = []
            labels = []
            frames = []
            if start == -1:
                frame_no = np.random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.batch_size)
                for i in range(self.batch_size):
                    labels.append(label)
            elif start2 != -1:
                frame_no = np.random.randint(start, end, self.batch_size // 4)
                for i in range(self.batch_size // 4):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(start2, end2, self.batch_size // 4))
                for i in range(self.batch_size // 4):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(0, start, 1))
                for i in range(self.batch_size // 4):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(end, start2, 32))
                for i in range(self.batch_size // 4):
                    labels.append(1)
            elif start2 == -1:
                frame_no = np.random.randint(start, end, self.batch_size // 2)
                for i in range(self.batch_size // 2):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(0, start, self.batch_size // 2))
                for i in range(self.batch_size // 2):
                    labels.append(0)

            for frame in frame_no:
                cap.set(1, frame)
                ret, frame = cap.read()
                if ret:
                    frame = np.transpose(frame, (2, 0, 1))
                    frame = self.preprocessing.transforms(torch.from_numpy(frame))
                    frame = self.preprocessing.preprocess(frame)
                    frame = self.preprocessing.augumentation(frame)
                    frames.append(frame)

            labels = np.array(labels, dtype=np.float32)
            # convert 5d [1, 4, 3, 256 ,256] to [4, 3, 256, 256] in torch
            X = torch.stack(frames, dim=0)

            if X.shape[0] == self.batch_size:
                break
            else:
                if X.shape[0] > self.batch_size:
                    X = X[:self.batch_size]
                    labels = labels[:self.batch_size]
                    break 
                else :
                    X = torch.cat((X, X[:self.batch_size - X.shape[0]]), dim=0)
                    labels = np.append(labels, labels[:self.batch_size - X.shape[0]])
                    break
        return X, labels         
        
class AnomalyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, num_workers:int,
                    data_path, annotation):
        super(AnomalyDataModule, self).__init__()
        self.annotation = annotation
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.data_path = data_path
        self.full_dataset = AnomalyDataset(self.batch_size,
                                           self.data_path, self.annotation)
    
    def setup(self, stage=None):    
        train_size = int(0.9 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True,
                           drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True,
                           drop_last=True)
        
if __name__ == '__main__':
    dataset_params = utils.config_parse('ANOMALY_DATASET')    
    dataset = AnomalyDataModule(**dataset_params)
    dataset.setup()
    print(len(dataset.full_dataset))
    store_path = utils.DATA_PATH + '/svr.npy'
    test_path = utils.DATA_PATH + '/svr_test.npy'
    feature_extractor = Efficientnetv2VarEncoder().to('cuda')
    with torch.no_grad():
        feature_extractor.eval()
        for i, (X, labels) in enumerate(dataset.train_dataloader()):
            X = X.to('cuda').squeeze(0)

            mu, var = feature_extractor(X)
            X = feature_extractor.reparameterize(mu, var)
            labels = labels.transpose(0, 1)
            X = X.detach().cpu().numpy()

            if os.path.exists(store_path):
                file = np.load(store_path)
                X = np.append(X, labels, axis=1)
                file = np.append(file, X, axis=0)
                np.save(store_path, file)
            else:
                X = np.append(X, labels, axis=1)
                np.save(store_path, X)

        for i ,(X, labels) in enumerate(dataset.test_dataloader()):
            X = X.to('cuda').squeeze(0)
            mu, var = feature_extractor(X)
            X = feature_extractor.reparameterize(mu, var)
            labels = labels.transpose(0, 1)
            X = X.detach().cpu().numpy()


            if os.path.exists(test_path):
                file = np.load(test_path)
                X = np.append(X, labels, axis=1)

                file = np.append(file, X, axis=0)
                np.save(test_path, file)
            else:
                X = np.append(X, labels, axis=1)

                np.save(test_path, X)