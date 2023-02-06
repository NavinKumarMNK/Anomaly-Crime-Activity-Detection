'@Author: NavinKumarMNK'
import sys, os
sys.path.append(os.path.abspath('../'))
from utils.torch_utils import TracedModel, select_device, time_synchronized
from utils.plots import plot_one_box
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from PIL import Image
from utils.datasets import LoadImages, LoadStreams
from models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
from pathlib import Path
import time
import argparse
import asyncio
import numpy as np
from scripts.Trackers.KalmanTracker import *
from utils import utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
import uuid
from scripts.FaceRecognition import Predictor
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def face_recognition(input_queue, output_queue):
        recognizer = Predictor(file=False, label=True)
        while True:
            if input_queue.empty():
                continue
            else:
                image = input_queue.get()
                name = recognizer.predict(image)
                output_queue.put(name)

class Process():
    def __init__(self, temp_dir, weigths, source, 
                    device = "cuda" if torch.cuda.is_available() else "cpu",
                    recognize=True) -> None:
        self.temp_dir = temp_dir
        self.capture = None
        self.weights = weigths
        self.path = source
        self.source = utils.path2src(source)
        self.device = device
        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
        self.recognize = recognize
        if self.recognize == True:
            self.output_queue = mp.Queue()
            self.input_queue = mp.Queue()
            self.recognizer = mp.Process(target=face_recognition, args=(self.input_queue, self.output_queue))
            self.recognizer.start()


    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f'Elapsed time: {end - start}')
            return result
        return wrapper

    def yolov7(self):
        set_logging()
        if(self.device == "cuda"):
            self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.weights, map_location=
                                self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)

        if (self.source == 'live'):
            self.model = TracedModel(self.model, self.device, 640)
            if self.half:
                self.model.half()
        
            self.source = '0'
            cudnn.benchmark = True
            self.dataset = LoadStreams(self.source, img_size=640, stride=self.stride)
            self.temp_dir = os.path.join(self.temp_dir, 'live')
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

        elif (self.source == 'video'):
            self.model = TracedModel(self.model, self.device, 640)
            if self.half:
                self.model.half()
        
            self.dataset = LoadImages(self.path, img_size=640, stride=self.stride)
            #create a folder with name of video and save inside it
            temp_dir = self.source.split("/")[-1]
            temp_dir = temp_dir.split(".")[0]
            self.temp_dir = os.path.join(self.temp_dir, temp_dir)
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
        else:
            self.dataset = LoadImages(self.path, img_size=640, stride=self.stride)

        for frame in self.process():
            yield frame
           

    @timer
    def process(self):
        if self.device != "cpu":
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))
        
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        temp_identities = [-1]
        faces_name = []
        faces_img = []
        for path, img, im0s, vid_cap in self.dataset:    
            img = np.array(img)
            im0s = np.array(im0s)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            if self.source == 'image':
                img = img.float()
            
            if self.device.type != "cpu" and (
                old_img_b != img.shape[0]
                or old_img_h != img.shape[2]
                or old_img_w != img.shape[3] 
                ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=1)[0]
            
            t1 = time_synchronized()
            pred = self.model(img, augment=1)[0]
            t2 = time_synchronized()
            pred = non_max_suppression(pred, 0.25, 0.45, 
                        classes=None, agnostic=False)
            t3 = time_synchronized()

            for i, det in enumerate(pred):
                if self.source == 'live' or self.source == '0':
                    p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), self.dataset.count
                else:
                    p, s, im0, frame = path, "", im0s, getattr(self.dataset, "frame", 0 )

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    dets_to_sort = np.empty((0, 6))
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack(
                            (dets_to_sort, np.array(
                                [x1, y1, x2, y2, conf, detclass]))
                        )

                    tracked_dets = self.sort_tracker.update(
                            dets_to_sort, 1
                        )

                    if len(tracked_dets) > 0:
                        widths = tracked_dets[:, 2] - tracked_dets[:, 0]
                        heights = tracked_dets[:, 3] - tracked_dets[:, 1]
                        mask = np.logical_and(widths >= 32, heights >= 32)
                        tracked_dets = tracked_dets[mask]

                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = dets_to_sort[:, 4]
                        print(temp_identities)
                        print(identities)
                        if (utils.lists_equal(temp_identities, identities) != True):
                            print("hello")
                            new_faces = utils.get_new_faces(identities, temp_identities)
                            print(new_faces)
                            faces_name = []
                            faces_img = []
                            iter_bbox = [bbox_xyxy[np.where(identities == x)[0][0]] for x in new_faces if x in identities]
                            for box in iter_bbox:
                                try:
                                    x1, y1, x2, y2 = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                                except ValueError:
                                    continue
                                face = im0[y1:y2, x1:x2]
                                faces_img.append(face)
                                if self.recognize == True:
                                    print("asdfsdf")
                                    self.input_queue.put(face)
                                    name = self.output_queue.get()
                                else:
                                    name = "output"
                                try:
                                    faces_name.append(name)
                                    uuid_text = str(uuid.uuid4())
                                    
                                    cv2.imwrite(os.path.join(self.temp_dir, f"{name}_{uuid_text}.jpg"), face)
                                except cv2.error as e:
                                    print(e)
                            temp_identities = identities
                        print(temp_identities)

                    if (faces_name == []):
                        faces_name = None

                    im0 = self.draw_boxes(
                        im0, bbox_xyxy, identities=identities,  
                            
                            color=5 , faces=faces_name
                    )

            if (self.source == '0'):
                ret, buffer = cv2.imencode('.jpg', im0)
                im0 = buffer.tobytes()
                yield im0
            elif(self.source == 'image'):
                yield im0
            elif self.source == 'video':
                cv2.imshow("face", im0)
                cv2.waitKey(1) 
                yield im0

    def draw_boxes(self, img, bbox, identities=None, 
                     color=None, faces=None):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            face = img[y1:y2, x1:x2]
            # line thickness
            tl = (round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)

            id = int(identities[i]) if identities is not None else 0
            try:
                face = str(faces[i]) if faces is not None else "output"
            except Exception as e:
                face = ' '
                print(e)

            color = 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
            label = (
                str(id) + " " + face 
                if identities is not None 
                else "No object"
            )
            # font thickness
            tf = 1
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -
                          1, cv2.LINE_AA)
            cv2.putText(
                img,
                label,
                (x1, y1 - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA
            )
        return img
    
    def start_capture(self):
        # capture faces
        for frame in self.yolov7():
            if(self.source == 'live' or self.source == '0'):
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            elif (self.source == "video"):
                yield frame
            elif (self.source == "image"):
                yield frame
        self.recognizer.kill()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,)
    args = parser.parse_args()
    for frame in Process('./temp', './weights/yolov7-tinyface.pt', args.source).start_capture():
        print(frame)
    