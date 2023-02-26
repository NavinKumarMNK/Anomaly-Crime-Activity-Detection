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
import asyncio
import websockets

async def handle_websocket(websocket, path):
    print(f"New connection from {websocket.remote_address}")
    try:
        while True:
            data = await websocket.recv()
            img_bytes = bytearray(data)
            npimg = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Display received image
            cv2.imshow("Server Side", npimg)
            cv2.waitKey(1)
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed from {websocket.remote_address}")

async def main():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        print("Websocket server started.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

'''
import cv2
def main():
    
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
'''
