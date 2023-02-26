import cv2
import numpy as np

# List of IP addresses of the cameras
camera_ips = ['192.168.1.100', '192.168.1.101']

video_streams = {}

for camera_ip in camera_ips:
    video_capture = cv2.VideoCapture("rtsp://" + camera_ip + "/stream")
    video_streams[camera_ip] = video_capture

while True:

    frames = {}
    for camera_ip, video_capture in video_streams.items():
        ret, frame = video_capture.read()
        frames[camera_ip] = frame

    for camera_ip, frame in frames.items():
        cv2.imshow(camera_ip, frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for video_capture in video_streams.values():
    video_capture.release()
cv2.destroyAllWindows()
