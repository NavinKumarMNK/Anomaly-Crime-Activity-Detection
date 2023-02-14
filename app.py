'@Author: "NavinKumarMNK"'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import utils
#from models import ResNet_Encoder, LRCN_Decoder
import cv2 
import numpy as np
from yoloface import YoloFace

device = utils.device()

'''

from flask_sockets import Sockets
from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)
sockets = Sockets(app)

connected_cameras = {}

@sockets.route('/video_feed')
def video_feed(ws):
    # store the IP address of the client in a dictionary
    client_ip = request.remote_addr
    connected_cameras[client_ip] = ws

    while True:
        message = ws.receive()
        if message is None:
            break

        # display the message (frame) using cv2
        frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

 
if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    from geventwebsocket.handler import WebSocketHandler

    server = WSGIServer(('127.0.0.1', 8000), app, handler_class=WebSocketHandler)
    server.serve_forever()

'''
import cv2
def main():
    '''
    # LRCN Decoder
    lrcn_params = utils.config_parse('./', 'LRCN_INFERENCE')
    lrcn_decoder = LRCN_Decoder(lrcn_params)

    # Resnet34 Encoder
    resnet_params = utils.config_parse('./', 'RESNET_INFERENCE')
    resnet_encoder = ResNet_Encoder(resnet_params)

    # SVR Decoder
    svr_params = utils.config_parse('./', 'SVR_INFERENCE')
    svr_decoder = SVR_Decoder(svr_params)

    # Face Detector
    face_detector = YoloFace()
    '''

    



    video_capture = cv2.VideoCapture(0)
    actual_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the step size
    step_size = int(actual_fps / 6)

    # Initialize the frame counter
    frame_counter = 0
    while True:
        # set capture fps to 2
        video_capture.set(cv2.CAP_PROP_FPS, 6)
        ret, frame = video_capture.read()
        frame_counter += 1
        if frame_counter % step_size == 0:
            cv2.imshow("Frame", frame)

        #cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
