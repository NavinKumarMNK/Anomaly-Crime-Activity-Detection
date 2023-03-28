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
from torch.utils.data import Dataset, DataLoader, TensorDataset
import cv2
import PIL
import numpy as np
from utils.preprocessing import ImagePreProcessing
from models.EfficientNetv2.Encoder import EfficientNetv2Encoder

class AnomalyDataset(Dataset):
    def __init__(self, data_path) -> None:
        super(AnomalyDataset, self).__init__()
        self.data_path = utils.ROOT_PATH + data_path
        self.annotation = open(self.data_path+"anomaly_test.txt", 
                                        'r').read().splitlines()
        self.preprocessing = ImagePreProcessing()
        
    def __len__(self): 
        return len(self.annotation)
    
    def __getitem__(self, idx):
        try:
            string = self.annotation[idx]
            lst = string.split('  ')
            print(lst)
            label = float(utils.label_parser(lst[1]))
            start, end = int(lst[2]), int(lst[3])
            start2, end2 = int(lst[4]), int(lst[5])
            video_path = lst[0]
            video_path = os.path.join(self.data_path, lst[1], video_path) 
            print(video_path)
            cap = cv2.VideoCapture(video_path.strip())
            if not cap.isOpened():
                print("Error opening video stream or file")

            frame_no = []
            labels = []
            frames = []
            if start == -1:
                print(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                frame_no = np.random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 128)
                for i in range(128):
                    labels.append(0)
            elif start2 != -1:
                frame_no = np.random.randint(start, end, 32)
                for i in range(32):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(start2, end2, 32))
                for i in range(32):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(0, start, 1))
                for i in range(32):
                    labels.append(0)
                frame_no = np.append(frame_no, np.random.randint(end, start2, 32))
                for i in range(32):
                    labels.append(0)
            elif start2 == -1:
                frame_no = np.random.randint(start, end, 64)
                for i in range(64):
                    labels.append(1)
                frame_no = np.append(frame_no, np.random.randint(0, start, 64))
                for i in range(64):
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
            print(X.shape)
            return X, labels
        except Exception as e:
            print(e)
            return None, None        
        
if __name__ == '__main__':
    dataset = AnomalyDataset(data_path='/data/')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    store_path = utils.ROOT_PATH + '/data/svr.npy'
    feature_extractor = EfficientNetv2Encoder().to('cuda')
    
    print("hello")
    with torch.no_grad():
        feature_extractor.eval()
        for i, (X, labels) in enumerate(dataloader):
            X = X.to('cuda').squeeze(0)

            X = feature_extractor(X)
            labels = labels.transpose(0, 1)
            X = X.detach().cpu().numpy()
            print(labels.shape, X.shape)

            if os.path.exists(store_path):
                file = np.load(store_path)
                X = np.append(X, labels, axis=1)
                print(X.shape, file.shape)
                file = np.append(file, X, axis=0)
                np.save(store_path, file)
            else:
                X = np.append(X, labels, axis=1)
                print(X.shape)
                np.save(store_path, X)

            print(i, X.shape, X[-1])
            break
