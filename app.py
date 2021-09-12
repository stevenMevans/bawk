from flask import Flask, send_from_directory
from flask_restful import Api
from flask_cors import CORS  # comment this on deployment

from src.api.pretrained_model_api_handler import PretrainedModelApiHandler
from src.api.custom_model_api_handler import CustomModelApiHandler


app = Flask(__name__)
CORS(app)  # comment this on deployment
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)


api.add_resource(CustomModelApiHandler, '/api/custom')
api.add_resource(PretrainedModelApiHandler, '/api/pretrained')
