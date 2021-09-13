import nemo.collections.asr as nemo_asr
import torch

from flask_restful import Resource

from src.api.keyword_api_handler import KeywordApiHandler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location=device)


def infer(filepath):
    return model.transcribe([filepath], batch_size=32)


class PretrainedModelApiHandler(Resource):
    def post(self):
        return KeywordApiHandler.handle(infer)
