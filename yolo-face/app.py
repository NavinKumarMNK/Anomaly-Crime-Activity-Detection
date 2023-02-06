'@Author: NavinKumarMNK'
import cv2
from flask import Flask, render_template, Response, send_from_directory
import webbrowser
import os
import psutil
from process import Process
import multiprocessing as mp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from threading import Thread

app = Flask(__name__, template_folder='./templates')


@app.route('/video-feed')
def video_feed():
    video = Process(os.path.abspath('./temp'), 
                    './weights/yolov7-tinyface.pt', 'live')
    return Response(video.start_capture(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

 
                    
def get_interface_ip(ifname):
    for interface, addrs in psutil.net_if_addrs().items():
        if interface == ifname:
            for addr in addrs:
                if addr.family == 2:
                    return addr.address
    return None


eth0_ip = get_interface_ip('eth0')
wlo1_ip = get_interface_ip('wlo1')

if __name__ == '__main__':
    if eth0_ip:    
        ip = eth0_ip
    elif wlo1_ip:
        ip = wlo1_ip
    port=5005
    '''
    browser = webbrowser.get()
    url=f'http://{ip}:{port}/'
    if browser is None:
        webbrowser.open(url)
    else:
        browser.open(url, new=0)
    '''
    app.run(host='0.0.0.0', debug=True, port=port )