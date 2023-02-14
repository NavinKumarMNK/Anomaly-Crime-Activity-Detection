"@author: NavinKumarMNK"
import os
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import cv2
from utils import utils
from models.preprocessing import ImagePreProcessing
preprocessing = ImagePreProcessing()

class CrimeActivityLRCNDataset(pl.LightningDataModule):
    def __init__(self, annotation_train:str, annotation_test:str, 
                        data_dir_path:str, num_workers:int,
                        sample_rate:int, batch_size:int
                        ):
        super(CrimeActivityLRCNDataset, self).__init__() 
        self.data_path = data_dir_path
        self.annotation_train = open(self.data_path + annotation_train, 'r').read().splitlines()
        self.annotation_test = open(self.data_path + annotation_test, 'r').read().splitlines()
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def __len__(self):
        return len(self.annotation_train), len(self.annotation_test)
    
    def __getitem__(self, index:int, is_train:bool=True):
        if is_train:
            annotation = self.annotation_train[index]
            video_path = self.data_path + annotation
            label = utils.label_parser(annotation.split('/')[0])
            video = cv2.VideoCapture(video_path)

            #sampling the frames
            frames = []
            for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate):
                video.set(1, i)
                ret, frame = video.read()
                if ret:
                    frames.append(frame)
            video.release()
            video = cv2.vconcat(frames)
            video = torch.from_numpy(video)

            video = preprocessing.transforms(video)
            video = preprocessing.preprocess(video)
            video = preprocessing.augumentation(video)

            return video, label


        else:
            annotation = self.annotation_test[index]
            video_path = self.data_path + annotation
            label = utils.label_parser(annotation.split('/')[0])
            video = cv2.VideoCapture(video_path)

            #sampling the frames
            frames = []
            for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate):
                video.set(1, i)
                ret, frame = video.read()
                if ret:
                    frames.append(frame)
            video.release()
            video = cv2.vconcat(frames)
            video = torch.from_numpy(video)

            video = preprocessing.transforms(video)
            video = preprocessing.preprocess(video)
            video = preprocessing.augumentation(video)

            return video, label
    
    def setup(self, stage: str) -> None:
        if(stage == 'fit'):
            self.annotation_train = self.annotation_train[:int(0.8*len(self.annotation_train))]
            self.annotation_val = self.annotation_train[int(0.8*len(self.annotation_train)):]
        else:
            self.annotation_test = self.annotation_test

    def train_dataloader(self):
        return DataLoader(self.annotation_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.annotation_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.annotation_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class TemporalAnomalyDataset():
    def __init__(self, annotation_train:str, annotation_test:str, 
                    data_dir_path:str, num_workers:int,
                    sample_rate:int, batch_size:int
                    ):
        super(CrimeActivityLRCNDataset, self).__init__() 
        self.data_path = data_dir_path
        self.annotation_train = open(self.data_path + annotation_train, 'r').read().splitlines()
        self.annotation_test = open(self.data_path + annotation_test, 'r').read().splitlines()
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def __len__(self):
        return len(self.annotation_train), len(self.annotation_test)
    
    def __getitem__(self, index:int, is_train:bool=True):
        if is_train:            
            annotation = self.annotation_test[index]
            video_path = self.data_path + annotation
            self.window_size = 20 # 20 frames
            video = cv2.VideoCapture(video_path)

            #sampling the frames
            frames = []
            for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate):
                video.set(1, i)
                ret, frame = video.read()
                if ret:
                    frames.append(frame)
            video.release()
            video = cv2.vconcat(frames)
            video = torch.from_numpy(video)

            video = preprocessing.transforms(video)
            video = preprocessing.preprocess(video)
            video = preprocessing.augumentation(video)

            return video

        else:
            annotation = self.annotation_train[index]
            video, label_dir , start1, end1, start2, end2 = annotation.split(' ')
            video_path = self.data_path + label_dir + "/" +video
            video = cv2.VideoCapture(video_path)

            #sampling the frames
            frames = []
            for i in range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.sample_rate):
                video.set(1, i)
                ret, frame = video.read()
                if ret:
                    frames.append(frame)
            video.release()
            video = cv2.vconcat(frames)
            video = torch.from_numpy(video)

            video = preprocessing.transforms(video)
            video = preprocessing.preprocess(video)
            video = preprocessing.augumentation(video)

            start1 = int(start1/self.sample_rate)
            end1 = int(end1/self.sample_rate)
            if start2 != -1:
                start2 = int(start2/self.sample_rate)
                end2 = int(end2/self.sample_rate)

            return video, start1, end1, start2, end2

    def setup(self, stage: str) -> None:
        if(stage == 'fit'):
            self.annotation_train = self.annotation_train[:int(0.8*len(self.annotation_train))]
            self.annotation_val = self.annotation_train[int(0.8*len(self.annotation_train)):]
        else:
            self.annotation_test = self.annotation_test    


    def train_dataloader(self):
        return DataLoader(self.annotation_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.annotation_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.annotation_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



if __name__ == "__main__":
    data = CrimeActivityLRCNDataset(**utils.config_parse(
                'CRIME_ACTIVITY_DATASET'))
    data.setup()
    print(data)    
    train_loader = data.train_dataloader()
    for i, (video, label) in enumerate(train_loader):
        print(video.shape)
        print(label)