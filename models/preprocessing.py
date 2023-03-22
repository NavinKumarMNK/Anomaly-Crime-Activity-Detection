"@author: NavinKumarMNK"
import cv2 
import torchvision.transforms as T
import numpy as np

class ImagePreProcessing:
    def __init__(self) -> None:
        super(ImagePreProcessing, self).__init__()
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
        ])  
        self.augumentation = T.Compose([
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(90)
        ])
        
