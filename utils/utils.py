import torch.nn as nn
import torch
import pytorch_lightning
import configparser


def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# Congifuration file

class ConfigParser():
    def __init__(self, txt) -> None:
        self.config = configparser.ConfigParser()
        self.txt = txt

    def parse(self):
        if(self.txt == 'LRCN'):
            self.config.read('../model.cfg')
            lrcn_config = dict(self.config.items(['LRCN']))
            return lrcn_config

