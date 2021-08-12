from flask import Flask, send_from_directory
from flask_restful import Api
from flask_cors import CORS  # comment this on deployment
from src.api.keyword_api_handler import KeywordApiHandler


app = Flask(__name__)
CORS(app)  # comment this on deployment
api = Api(app)


api.add_resource(KeywordApiHandler, '/api/asr')
