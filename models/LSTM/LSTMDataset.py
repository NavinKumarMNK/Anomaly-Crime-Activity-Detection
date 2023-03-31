"@Author: NavinKumarMNK"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import torch
import cv2
from utils import utils
from utils.exceptions import VideoNotOpened, VideoTooShort
import numpy as np
from utils.preprocessing import ImagePreProcessing

class LSTMDataset(Dataset):
    def __init__(self, annotation:str,
                        data_dir_path:str, num_workers:int,
                        sample_rate:int
                        ):
        super(LSTMDataset, self).__init__() 
        self.data_path = utils.DATA_PATH + data_dir_path 
        self.annotation = open(self.data_path + annotation, 'r').read().splitlines()
        self.num_workers = num_workers
        self.preprocessing = ImagePreProcessing()
        self.global_sample_rate = self.sample_rate = sample_rate

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index:int):
        while True:
            self.sample_rate = self.global_sample_rate
            try:
                annotation = self.annotation[idx]
                lst = annotation.split('  ')
                video_path = self.data_path + lst[1]  + '/' + lst[0]
                
                print(annotation)

                label = utils.label_parser(lst[1])
                video = cv2.VideoCapture(video_path)
                print(video_path, label)
                if not (video.isOpened() and video.get(cv2.CAP_PROP_FRAME_COUNT) > 0):
                    raise VideoNotOpened("Video is not opened or frame count is 0")

                #sampling the frames
                frames = []

                if (int(lst[2]) != -1):
                    if (int(lst[3]) - int(lst[2]) < self.batch_size and int(lst[2]) != -1):
                        raise VideoTooShort("Video is too short")
                    
                    elif (int(lst[3]) - int(lst[2]) / self.sample_rate < self.batch_size):
                        self.sample_rate = int((int(lst[3]) - int(lst[2])) / self.batch_size)
                    start = int(lst[2])
                    end = int(lst[3])
                else:
                    start = 0
                    end = start + self.batch_size * self.sample_rate
                    if (end > int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                        end = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        if(end - start < self.batch_size):
                            raise VideoTooShort("Video is too short")
                            
                        elif (end - start / self.sample_rate < self.batch_size):
                            self.sample_rate = int((end - start) / self.batch_size)
                
                end = start + self.batch_size * self.sample_rate
                count = 0             
                
                for i in range(start, end, self.sample_rate):
                    video.set(1, i)
                    ret, frame = video.read()
                    if ret:
                        frame = np.transpose(frame, (2, 0, 1))
                        frame = self.preprocessing.transforms(torch.from_numpy(frame))
                        frame = self.preprocessing.preprocess(frame)
                        frames.append(frame)
                        count += 1
                    if count > 127:
                        break

                video.release()
                X = torch.stack(frames)
                y = torch.tensor(label)
                print(X.shape)
                print(y)
                break

            except Exception as e:
                if (idx >= len(self.annotation)):
                    idx = 0
                print(e)
                idx += 1
                continue

        print(X.shape, y)
        return X, y 
    
class LSTMDatasetModule(pl.LightningDataModule):
    def __init__(self, num_workers:int,
                    data_path, annotation, sample_rate) -> None:
        super(LSTMDatasetModule, self).__init__()
        self.annotation = annotation
        self.num_workers = int(num_workers)
        self.data_path = data_path
        self.sample_rate = sample_rate

    def setup(self, stage=None):
        self.dataset = LSTMDataset(num_workers=self.num_workers,
                                           data_dir_path=self.data_path, annotation=self.annotation,
                                           sample_rate=self.sample_rate)
        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)


if __name__ == "__main__":
    data = LSTMDatasetModule(**utils.config_parse(
                'LSTM_DATASET'))
    data.setup()
    train_loader = data.train_dataloader()
    for i, (video, label) in enumerate(train_loader):
        print(video.shape)
        print(label)