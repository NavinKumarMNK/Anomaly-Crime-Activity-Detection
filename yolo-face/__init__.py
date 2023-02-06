import torch
import cv2
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
sys.path.append(os.path.abspath('../'))


def init():
    global necessary_libraries
    neccessary_libraries = {'cv2': cv2, 'torch': torch}

