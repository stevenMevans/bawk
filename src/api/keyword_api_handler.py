import torch
import asyncio
import os
import shutil

from flask_restful import reqparse
from uuid import uuid4
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.api.keyword_detection_service import KeywordDetectionService


class KeywordApiHandler:

    @staticmethod
    def handle(infer):
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

        transcriptions = infer(filepath)
        detector = KeywordDetectionService(args.keyword)
        ret_msg = detector.check_text(transcriptions[0], spellcheck=True)

        final_ret = {
            "status": "Success",
            "detected": ret_msg,
            "keyword": args.keyword,
            "transcript": transcriptions[0]
        }
        shutil.rmtree(data_store)

        return final_ret
