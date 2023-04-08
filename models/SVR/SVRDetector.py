'@AUTHOR: NavinKumarMNK'
# Add the parent directory to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import utils.utils as utils

# Import the required modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
import wandb

class OneClassSVMDetector():
    def __init__(self, input_size=1024, kernel='rbf', c=1.0, gamma='scale'):
        self.input_size = input_size
        self.scaler  = StandardScaler()
        self.pre_scaler = MinMaxScaler()
        
        self.OneClassSVM = OneClassSVM(kernel=kernel, nu=0.5, gamma=gamma)

        self.frame_no=0
        self.score = []
        try :
            self.load(utils.ROOT_PATH + '/weights')
            print("Model loaded successfully")
        except:
            self.save(utils.ROOT_PATH + '/weights')
        pass
        wandb.init(project="anomaly-detection", name="OneClassSVM")

    def __del__(self):
        pass
        #wandb.finish()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.OneClassSVM.fit(X_scaled, y)
    
    def __call__(self, X):
        return self.predict(X)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.OneClassSVM.predict(X_scaled)
        self.frame_no += 1
        self.log_score(scores, self.frame_no)
        return scores

    def save(self, path):
        joblib.dump(self.OneClassSVM, path+'/OneClassSVM_model.pkl')
        joblib.dump(self.scaler, path+'/scaler.pkl')

    def load(self, path):
        self.OneClassSVM = joblib.load(path+'/OneClassSVM_model.pkl')
        self.scaler = joblib.load(path+'/scaler.pkl')

    def log_score(self, score, frame_number):
        self.score.append(score)
        wandb.log({'score': score, 'frame_number': frame_number})

if __name__ == '__main__':
    import numpy as np
    
    data = np.load(utils.DATA_PATH + '/svr.npy')
    X = data[:, 0:1024]
    y = data[:, 1024]

    trainer_params = utils.config_parse('OneClassSVM_DECODER')
    OneClassSVM = OneClassSVMDetector(**trainer_params)
    OneClassSVM.fit(X, y)
    OneClassSVM.save(utils.ROOT_PATH + '/weights')
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    data = np.load(utils.DATA_PATH + '/svr_test.npy')
    X = data[:, 0:1024]
    y_true = data[:, 1024]

    y_pred = OneClassSVM.predict(X)
    y_pred = np.where(y_pred == -1, 0, 1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')