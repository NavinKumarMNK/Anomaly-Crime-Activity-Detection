import torch.nn as nn
import torch
import pytorch_lightning
import configparser
import os

root_folder = os.path.abspath('../')

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def absolute_path(self, txt):
        return os.path.join(root_folder, txt)

# Congifuration file
def config_parse(txt):
    config = configparser.ConfigParser()
    config.read('../model.cfg')
    params={}
    for key, value in config[txt].items():
        if 'path' in key:
            params[key] = absolute_path(value)
        else:
            params[key] = value
    return params

        

