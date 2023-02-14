'@Author: NavinKumarMNK'
import sys 
if '../' not in sys.path:
    sys.path.append('../')
import numpy as np
from utils import utils
import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from scripts.FRMethods.ContractiveLossFR import ContractiveLossFR, ContractiveLossFREmbeddingsDataModule
from scripts.Upsample import Upsample
from scripts.FRMethods.SingleShotLearningFR import SSLFacentDataModule, SingleShotLearningFR
from scripts.FRMethods.ContractiveLossFR import ContractiveLossFR
from PIL import Image
import cv2
import json
from scripts.Upsample import Upsample
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor():
    def __init__(self, file=False,
                    label=True):
        self.label = label
        self.upsample = Upsample()
        self.file = file
        self.model = SingleShotLearningFR(pretrained=True)
        args = utils.config_parse('CONTRACTIVE_LOSS_FR')
        args['num_classes'] = len(os.listdir(utils.ROOT_PATH + '/database/faces')) 
        self.embedding = ContractiveLossFR(**args, pretrained=True) 
        self.model = self.model.to(DEVICE)

    def upsampler(self , image, file=False):
        try:
            self.upsample.set_image(image, 0.5, file)
            self.upsample.denoising()
            self.upsample.sharpening_mask()
            self.upsample.super_resolution_gan('espcn')        
            self.upsample.interpolation()
            image = self.upsample.get_image()
            return image
        except Exception as e:
            print(e)
            return image

    def predict(self, image):
        image = self.upsampler(image, file=self.file)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = transforms.ToTensor()(image)
        image = transforms.Resize((64, 64))(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0)
        image = image.to(DEVICE)
        
        self.model.eval()
        embedding = self.model(image)
        prediction = self.embedding(embedding)
        prediction = torch.argmax(prediction, dim=1)
        
        if self.label:
            return self.ret_label(prediction)
        else:
            return prediction
    
    def ret_label(self, prediction):
        with open(utils.ROOT_PATH+ '/database/label_map.json') as f:
            data = json.load(f)
        for key, value in data.items():
            if value == prediction.item():
                return key
 
class Trainer():
    def __init__(self, root_dir, label_map, embeddings, 
                        batch_size=1, pretrained=True):
        self.root_dir = root_dir
        self.label_map = label_map
        self.pretrained = pretrained
        self.embeddings = embeddings
        self.batch_size = batch_size

    def train(self):
        self.ssl_trainer()
        self.ctl_trainer()

    def ssl_trainer(self):
        model = SingleShotLearningFR(pretrained=True)
        data = SSLFacentDataModule(root_dir=self.root_dir,
                                label_map=self.label_map,
                                batch_size=self.batch_size)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
        trainer.fit(model, data)
        trainer.save_checkpoint(utils.ROOT_PATH + '/weights/ssl_facenet.ckpt')

    def ctl_trainer(self):
        args = utils.config_parse('CONTRACTIVE_LOSS_FR')
        args['num_classes'] = len(os.listdir(self.root_dir)) 
        model = ContractiveLossFR(**args, pretrained=True)
        data = ContractiveLossFREmbeddingsDataModule(self.embeddings, batch_size=self.batch_size)
        data.prepare_data()
        trainer = pl.Trainer(
        accelerator='gpu', devices=1,
        min_epochs=10,
        max_epochs=25,
        )
        trainer.fit(model, data)
        trainer.save_checkpoint(utils.ROOT_PATH + '/weights/contractive_loss.cpkt')


if __name__ == '__main__':
    face_recognizer = Predictor(file=True, label=True)
    prediction = face_recognizer.predict('../test/navin.jpg')
    print(prediction)

    '''
    face_trainer = Trainer(root_dir=utils.ROOT_PATH+ "/database/faces",
                        label_map=utils.ROOT_PATH+'/database/label_map.json',
                        embeddings=utils.ROOT_PATH+'/database/embeddings.csv',
                        batch_size=2)
    face_trainer.train()
    '''