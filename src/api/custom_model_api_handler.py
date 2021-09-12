import torch

from flask_restful import Resource

from models.inference import EncoderRNN, AttnDecoderRNN, inference_from_file
from src.api.keyword_api_handler import KeywordApiHandler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 80 * 401
output_size = 29
hidden_size = 100
enmodel = EncoderRNN(input_size=input_size, hidden_size=hidden_size)
enmodel.load_state_dict(torch.load("/workspace/bawk/models/enc_model", map_location=device))
enmodel.eval()

decmodel = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_size)
decmodel.load_state_dict(torch.load("/workspace/bawk/models/dec_model", map_location=device))
decmodel.eval()


def infer(filepath):
    return inference_from_file(filepath, enmodel, decmodel)


class CustomModelApiHandler(Resource):
    def post(self):
        return KeywordApiHandler.handle(infer)
