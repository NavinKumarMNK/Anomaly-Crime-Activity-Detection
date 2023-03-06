import torch.nn as nn
import torch
import pytorch_lightning
import configparser
import os

ROOT_PATH = '/home/mnk/MegNav/Projects/Crime-Activity-Detection-and-Suspect-Identification'

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

def dataset_image_autoencoder():
    file_path = ROOT_PATH + "/data/"
    batch_size = config_parse('AUTOENCODER_DATASET')['batch_size']
    import cv2
    annotation = open(file_path+"anomaly_train.txt", 'r').read().splitlines()
    with open(file_path+"auto_encoder.txt", "w+") as f:
        for video_path in annotation:
            video = cv2.VideoCapture(file_path + video_path)
            count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            count = int(count / batch_size) + 1
            i = 0
            while i != count:
                i+=1
                str = f"{video_path} \n"
                f.write(str)
    return f"{file_path}auto_encoder.txt"

if __name__ == '__main__':
    dataset_image_encoderclassifer()
