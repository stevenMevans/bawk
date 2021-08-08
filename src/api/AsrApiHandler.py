from flask_restful import Resource, reqparse
import os
import nemo.collections.asr as nemo_asr


model = nemo_asr.models.ASRModel.from_pretrained("QuartzNet15x5Base-En", map_location='cpu')
# file = os.path.relpath('../data/an197-mgah-b.wav')
transcriptions = model.transcribe(['src/data/an197-mgah-b.wav'], batch_size=32)


class AsrApiHandler(Resource):

    def get(self):        
        return {
            'resultStatus': 'SUCCESS',
            'message': transcriptions
        }

    def post(self):
        print(self)
        parser = reqparse.RequestParser()
        parser.add_argument('type', type=str)
        parser.add_argument('message', type=str)

        args = parser.parse_args()

        print(args)
        # note, the post req from frontend needs to match the strings here (e.g. 'type and 'message')

        request_type = args['type']
        request_json = args['message']
        # ret_status, ret_msg = ReturnData(request_type, request_json)
        # currently just returning the req straight
        ret_status = request_type
        ret_msg = request_json

        if ret_msg:
            message = "Your Message Requested: {}".format(ret_msg)
        else:
            message = "No Msg"

        final_ret = {"status": "Success", "message": message}

        return final_ret