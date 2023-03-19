'@AUTHOR: NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import utils.utils as utils

# Import the required modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import numpy as np
#import wandb

class SVRDetector():
    def __init__(self, input_size=1536, kernel='rbf', c=1.0, gamma='scale'):
        self.input_size = input_size
        self.scaler  = StandardScaler()
        self.svr = SVR(kernel=kernel, C=c, gamma=gamma)
        self.frame_no=0
        self.score = []
        try :
            self.load(utils.ROOT_PATH + '/weights')
        except:
            pass
        pass
        #wandb.init()

    def __del__(self):
        pass
        #wandb.finish()

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
        #wandb.log({'score': score, 'frame_number': frame_number})

if __name__ == '__main__':
    import numpy as np
    
    data = np.load(utils.ROOT_PATH + '/data/svr.npy')
    X = data[:, 0:1536]
    y = data[:, 1536]

    trainer_params = utils.config_parse('SVR_DECODER')
    svr = SVRDetector(**trainer_params)
    svr.fit(X, y)
    svr.save(utils.ROOT_PATH + '/weights')

    y_pred = svr.predict(X)
    print(y_pred)

