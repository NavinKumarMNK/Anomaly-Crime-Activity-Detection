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

class SVRDecoder():
    def __init__(self, input_size, kernel='rbf', C=1.0, gamma='scale'):
        self.input_size = input_size
        self.scaler  = StandardScaler()
        self.svr = SVR(kernel=kernel, C=C, gamma=gamma)
        self.frame_no=0
        self.score = []
        wandb.init()

    def __del__(self):
        wandb.finish()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.svr.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.svr.predict(X_scaled)
        self.frame_no += 1
        self.log_score(scores, self.frame_no)
        return scores

    def save(self, path):
        joblib.dump(self.svr, path+'/svr_model.pkl')
        joblib.dump(self.scaler, path+'/scaler.pkl')

    def load(self, path):
        self.svr = joblib.load(path+'/svr_model.pkl')
        self.scaler = joblib.load(path+'/scaler.pkl')

    def log_score(self, score, frame_number):
        self.score.append(score)
        wandb.log({'score': score, 'frame_number': frame_number})

if __name__ == '__main__':
    import pandas 
    df = pandas.read_csv(utils.absolute_path('../data/svm.csv'))
    from sklearn.model_selection import train_test_split
    
    X = df.iloc[:, 0:1024].values
    y = df.iloc[:, 1024].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    trainer_params = utils.config_parser("../", 'SVR_DECODER') 
    svr = SVRDecoder(**trainer_params)
    svr.fit(X_train, y_train)
    svr.save(utils.absolute_path('../models/weights/svr_model.pkl'))
    
    # Load the model
    svr.load(utils.absolute_path('../models/weights/svr_model.pkl'))
    y_pred = svr.predict(X_test)
    
    # Evaluate
    from sklearn.metrics import mean_squared_error
    print('MSE: ', mean_squared_error(y_test, y_pred))

