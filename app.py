"@author: 'NavinKumarMNK'"
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import subprocess
from werkzeug.utils import secure_filename
import os

# Flask app
app = Flask(__name__)
api = Api(app)

# api resources /predict
class Predict(Resource):
    def get(self):
        # run main.py & get images of faces, class of prediction
        return jsonify({'message': 'Hello World!'})

    def post(self):
        #recieve video file and sanitize it 
        data = request.files['video']
        filename = secure_filename(data.filename)
        data.save(filename)

        result = subprocess.run(['python', 'main.py'], stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, shell=True, wait=True, 
                        check=True, universal_newlines=True)
        result = result.stdout
        
        #load images from the inference folder
        # return the images and the class of prediction
        images = []
        for root, dirs, files in os.walk('path/to/folder'):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    images.append(file)

        return jsonify({'message': result,
                        'images': images})


# api endpoints
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
