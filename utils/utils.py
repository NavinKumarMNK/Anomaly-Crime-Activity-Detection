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
            if 'path' in key:
                params[key] = absolute_path(value)
            else:
                params[key] = value
    except KeyError as e:
        print("Invalid key: ", e)
        print(path)    
    
    return params

def label_parser(string):
    params = config_parse('LABELS')
    return int(params[string])

def one_hot_encode(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


