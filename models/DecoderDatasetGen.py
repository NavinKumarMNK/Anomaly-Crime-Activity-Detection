'@Author:NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
import utils.utils as utils

from models.EfficientNetv2.Encoder import EfficientNetv2Encoder
from models.SVR.SVRDetector import SVRDetector
from utils.preprocessing import ImagePreProcessing

import torch
import torch.nn as nn
import pytorch_lightning as pl
import cv2
import numpy as np

preprocessing = ImagePreProcessing()
svr = SVRDetector()
cnn = EfficientNetv2Encoder().to('cuda')
file = utils.ROOT_PATH + '/data/anomaly_train.txt'
save_path = utils.ROOT_PATH + '/data/crime_activity.txt'

with open(file, 'r') as f, open(save_path, 'w+') as g:
    for string in f:
        string = string.strip()
        cap = cv2.VideoCapture(utils.ROOT_PATH + '/data/' + string)
        count = 0
        anomaly_count = 0 
        anomaly_count_neg = 0
        anomaly = False
        start = -1
        end = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if (count % 5 != 0):
                continue
            print(start, end, count, anomaly_count, anomaly_count_neg, anomaly)
            if ret:
                frame = np.transpose(frame, (2, 0, 1))
                frame = preprocessing.transforms(torch.from_numpy(frame).to('cuda'))
                frame = preprocessing.preprocess(frame)
                frame = preprocessing.augumentation(frame)
            frame = frame.unsqueeze(0).to('cuda')
            cnn_output = cnn(frame)
            cnn_output = cnn_output.detach().cpu().numpy()
            svr_output = svr.predict(cnn_output)
            if(svr_output > 0.75):
                anomaly = True
                anomaly_count += 1
            elif(svr_output < 0.75 and anomaly):
                anomaly = False
                anomaly_count_neg += 1

            if (anomaly_count > 30):
                start = count - 30*5
            elif (anomaly_count_neg > 30):
                end = count - 30*5
                anomaly_count = 0
                anomaly = False
            count += 1

        #add_str = string.split('/')[1] + "  " + string.split('/')[0] + "  " + str(start[0]) + "  " + str(end[0]) + "  " + str(start[1]) + "  " + str(end[1]) + " "
        
        first_string = string.split('/')[0] + "  " + string.split('/')[1]
        str1 = first_string + "  " + "Normal"             + "0"       + "  " + start
        g.write(str1)
        str2 = first_string + "  " + string.split('/')[0] + start + "  " + end 
        g.write(str2)
        str3 = first_string + "  " + "Normal"             + end   + "  " + cap.get(cv2.CAP_PROP_FRAME_COUNT)
        g.write(str3)
        cap.release()

