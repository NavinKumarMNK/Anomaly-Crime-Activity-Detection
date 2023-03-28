'@Author: "NavinKumarMNK"'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.utils as utils
import cv2 
import numpy as np


device = utils.device()
import asyncio
import websockets
app_params = utils.config_parse('APP')
print(app_params)

from models.SVR.SVRDetector import SVRDetector
from models.LSTM.Decoder import LSTMDecoder
from models.EfficientNetv2.Encoder import EfficientNetv2Encoder
from utils.preprocessing import ImagePreProcessing

pre = ImagePreProcessing()
encoder = EfficientNetv2Encoder().to(device)
anomaly_detector = SVRDetector()

async def handle_websocket(websocket, path):
    with torch.no_grad():
        encoder.eval()
        print(f"New connection from {websocket.remote_address}")
        decoder = LSTMDecoder().to(device)
        decoder.eval()
        try:
            count =0 
            neg_count=0
            flag = 0
            action = None
            while True:
                data = await websocket.recv()
                img_bytes = bytearray(data)
                npimg = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                embeddings = encoder(pre.transforms(npimg).to(
                                device).squeeze(0))
                res = anomaly_detector(embeddings.detach().cpu().numpy())
                if (res > app_params['anomaly_threshold']):
                    count+=1
                else :
                    count = 0
                
                if (res < app_params['anomaly_threshold'] / 2):
                    neg_count -= 1    
                
                if (neg_count < -10):
                    flag = 0
                    neg_count = 0   
                    decoder.reset_hidden()
                
                if(count > 10):
                    print("Anomaly Detected")
                    flag = 1
                    count = 0

                if (flag == 1):
                    action = decoder(npimg)

                if (action == -1):
                    flag = 0
                    count = 0
                    neg_count = 0
                else:
                    print("Action : ", action)
                    
                cv2.imshow("APP", npimg)
                cv2.waitKey(1)
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed from {websocket.remote_address}")
            cv2.destroyWindow("APP")

async def main():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        print("Websocket server started.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

