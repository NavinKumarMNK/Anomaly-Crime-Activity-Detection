import torch.nn as nn
import torch
import configparser
import os

ROOT_PATH = '/home/windows/Video-Detection'
DATA_PATH = '/home/windows/Data/data'

def current_path():
    return os.path.abspath('./')

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def absolute_path(txt):
        return os.path.join(ROOT_PATH, txt)

# Congifuration file
def config_parse(txt):
    config = configparser.ConfigParser()
    path = ROOT_PATH + '/config.cfg'
    config.read(path)
    params={}
    try:
        for key, value in config[txt].items():
            if 'path' in key.split('_'):
                params[key] = absolute_path(value)
            elif value == 'True' or value== 'False':
                params[key] = True if value == 'True' else False
            elif value.isdigit():
                params[key] = int(value)
            elif '.' in value:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            else:
                params[key] = value
    except KeyError as e:
        print("Invalid key: ", e)
        print(path)    
    
    return params

def label_parser(string):
    params = config_parse('LABELS')
    string = string.lower()
    return int(params[string])

def one_hot_encode(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

def dataset_image_autoencoder(file_path, file_name):
    batch_size = config_parse('AUTOENCODER_DATASET')['batch_size']
    import cv2
    annotation = open(file_path+file_name, 'r').read().splitlines()
    with open(file_path+"auto_encoder.txt", "w") as f:
        for video_path in annotation:
            write_path = video_path
            lst = video_path.split('/')
            video_path = lst[1]
            video_path = os.path.join(file_path, lst[0], video_path) 
            print(video_path)
            count = 16
            i = 0
            while i != count:
                i+=1
                str = f"{write_path} \n"
                f.write(str)
    return f"{file_path}auto_encoder.txt"


