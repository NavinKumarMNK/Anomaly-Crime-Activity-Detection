'@Author: NavinKumarMNK'
import torch
import torch.nn as nn
import sys
if '../../' not in sys.path:
    sys.path.append('../../')
from FaceNet import FaceNet
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import json
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Resize
import numpy as np
from PIL import Image
import pandas as pd
import uuid
from utils import utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatabaseHandler():
    def __init__(self, database=utils.ROOT_PATH+'/database', weights='/weights/face_embeddings.pt') -> None:
        self.database = database
        self.weights = weights
        self.model = FaceNet(model='resnet101', pretrained=False, im_size=64)
        self.model.load_state_dict(torch.load(utils.ROOT_PATH + self.weights))
        self.model.to(DEVICE)
        self.transform = transforms.Compose([
            Resize((64, 64)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def add_label_folder(self):
        self.folders_created = []
        class FolderHandler(FileSystemEventHandler):
            def __init__(self, str) -> None:
                self.str = str
                self.previous_folders = set(os.listdir(str))

            def on_created(self, event):
                # check if the event is for a directory
                if event.is_directory:
                    print("directory created:{}".format(event.src_path))
                    current_folders = set(os.listdir(self.str))
                    new_folders = current_folders - self.previous_folders
                    print(new_folders)
                    for folder in new_folders:
                        folder_path = os.path.join(self.str, folder)
                        DatabaseHandler.folders_created.append(folder_path)
                    self.previous_folders = current_folders
        
        observer = Observer()
        observer.schedule(FolderHandler('./faces'), path='./faces', recursive=True)
        observer.start()

        input("Add label-folders to database/faces . Press Enter once your are done")

        observer.stop()
        observer.join()

        self._face2database()

    def add_label_camera(self):
        self.folders_created = []
        while True:
            label = input("Enter Your Label = ")
            if label == "exit":
                break
            #capture the face through cv2 and add create a folder and add the image to the folder
            folder_path = os.path.join("faces", label)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            else:
                print("User already exists. Create New User")
                continue
            
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                #yolov7 to detect face

                a = input("Press Enter to capture q to quit")
                if a == "q":
                    break
                
                #save the image to the folder
                image_name = str(uuid.uuid4()) + ".jpg"
                image_path = os.path.join(folder_path, image_name)
                cv2.imwrite(image_path, frame)

            cap.release()
            cv2.destroyAllWindows()
            self.folders_created.append(folder_path)
        
        self._face2database()
    
    def _face2database(self):
        with open("label_map.json", "r") as f:
            label_map = json.load(f)
            label_id = len(label_map)
            for folder in self.folders_created:
                label = folder.split("/")[-1]
                if label not in label_map:
                    label_map[label] = len(label_map)

        with open("label_map.json", "w") as f:
            json.dump(label_map, f)

        embeddings = []
        for folder in self.folders_created:
            print("Adding {} to database".format(folder))
            images = os.listdir(folder)
            for image in images:
                image_path = os.path.join(folder, image)
                image = Image.open(image_path)
                image = self.transform(image).unsqueeze(0)
                print(image.shape)
                self.model.eval()
                embedding = self.model(image)
                embedding = embedding.detach().cpu().numpy()
                embedding = embedding.reshape(512)
                embedding = embedding.tolist()
                embeddings.append([label_id] + embedding)

        if os.path.exists("embeddings.csv"):
            df = pd.read_csv("embeddings.csv")
            df = df.concat([df, pd.DataFrame(embeddings, columns=df.columns)])
            df.to_csv("embeddings.csv", index=False)
        else:
            columns = ["label"] + ["embedding_"+str(i) for i in range(512)]
            df = pd.DataFrame(embeddings, columns=columns)
            df.to_csv("embeddings.csv", index=False)

    def yolo_face(self):
        pass

if __name__ == "__main__":
    db = DatabaseHandler()
    db.add_label_folder()
    db.add_label_camera()