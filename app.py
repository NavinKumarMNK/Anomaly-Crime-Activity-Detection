"@author: 'NavinKumarMNK'"
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

# Flask app
app = Flask(__name__)
api = Api(app)


# api resources /predict
class Predict(Resource):
    def get(self):
        return jsonify({'message': 'Hello World!'})

    def post(self):
        data = request.get_json()
        return jsonify({'message': 'Hello World!'})


# api endpoints
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
