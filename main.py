'@Author: "NavinKumarMNK"'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import utils
from scripts import lrcn

device = utils.device()

# Anomally Detection Model for video in inference
# Extract the frames from the video of anomally
# Pass the frames through the LRCN model
# Get the prediction from the model
# extract the faces in the from the model
# return those faces to through the face recognition model

if __name__ == '__main__':
    pass
