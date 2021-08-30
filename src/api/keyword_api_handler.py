import asyncio
import os
import shutil
import torch
from flask_restful import Resource, reqparse
from uuid import uuid4
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from models.inference import EncoderRNN, AttnDecoderRNN, inference_from_file
from src.api.keyword_detection_service import KeywordDetectionService

# import nemo.collections.asr as nemo_asr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 80 * 401
output_size = 29
hidden_size = 100
# model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')
enmodel = EncoderRNN(input_size=input_size, hidden_size=hidden_size)
enmodel.load_state_dict(torch.load("/workspace/bawk/models/enc_model", map_location=device))
enmodel.eval()

decmodel = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_size)
decmodel.load_state_dict(torch.load("/workspace/bawk/models/dec_model", map_location=device))
decmodel.eval()


async def _remove_file(filepath):
    shutil.rmtree(filepath)


async def _detect_keyword(filedir, filename, keyword):
    filepath = os.path.join(filedir, filename)
    transcriptions = inference_from_file(filepath, enmodel, decmodel)
    detector = KeywordDetectionService(keyword)
    detector.check_text(transcriptions[0])
    _remove_file(filedir)


class KeywordApiHandler(Resource):

    def post(self):
        """
        API Endpoint to detect a keyword in an audio file.
        """
        parser = reqparse.RequestParser()
        parser.add_argument('keyword', type=str)
        parser.add_argument('audio', type=FileStorage, location='files')

        uuid = str(uuid4())
        data_store = os.path.join('../data', uuid)

        args = parser.parse_args()
        file = args.audio
        filename = secure_filename(file.filename)
        if not os.path.exists(data_store):
            os.makedirs(data_store)
        filepath = os.path.join(data_store, filename)
        file.save(filepath)

        transcriptions = inference_from_file(filepath, enmodel, decmodel)
        detector = KeywordDetectionService(args.keyword)
        ret_msg = detector.check_text(transcriptions[0])

        final_ret = {
            "status": "Success",
            "detected": ret_msg,
            "keyword": args.keyword,
            "transcript": transcriptions[0]
        }
        # asyncio.run(_detect_keyword(filedir=data_store, filename=filename, keyword=args.keyword))
        _remove_file(filepath=data_store)

        return final_ret
