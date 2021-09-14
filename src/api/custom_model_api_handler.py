import torch

from flask_restful import Resource

from models.inference import inference_from_file
from models.rename_unpickler import renamed_load
from src.api.keyword_api_handler import KeywordApiHandler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = '/app/model_las_updated_final.pth'
encoder_pkl_path = '/app/encoder_las.pkl'
decoder_pkl_path = '/app/decoder_las.pkl'

with open(encoder_pkl_path, 'rb') as convert_file:
    encoder = renamed_load(convert_file)

with open(decoder_pkl_path, 'rb') as convert_file:
    decoder = renamed_load(convert_file)

checkpoint = torch.load(model_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.eval()
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder.eval()


def infer(filepath):
    return inference_from_file(filepath, encoder, decoder)


class CustomModelApiHandler(Resource):
    def post(self):
        return KeywordApiHandler.handle(infer)
