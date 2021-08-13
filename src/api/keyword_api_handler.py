from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from uuid import uuid4

import shutil
import os
import asyncio
import nemo.collections.asr as nemo_asr

from src.api.keyword_detection_service import KeywordDetectionService


model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')


async def _remove_file(filepath):
    shutil.rmtree(filepath)


async def _detect_keyword(filedir, filename, keyword):
    filepath = os.path.join(filedir, filename)
    transcriptions = model.transcribe([filepath], batch_size=32)
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

        transcriptions = model.transcribe([filepath], batch_size=32)
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
