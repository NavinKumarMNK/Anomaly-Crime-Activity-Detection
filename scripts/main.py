'@AUTHOR: NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
import utils.utils as utils

# Import the required modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import numpy as np
import wandb

class Main():
    def __init__(self):
        pass

    def run(self):  
        print("Works fine")