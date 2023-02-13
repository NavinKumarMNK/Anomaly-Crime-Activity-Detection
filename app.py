# server.py
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
    # Get the video stream from the camera
    video_capture = cv2.VideoCapture(0)
    while True:
        # Read the frames from the video stream
        ret, frame = video_capture.read()
        # Display the frames
        cv2.imshow('frame', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
