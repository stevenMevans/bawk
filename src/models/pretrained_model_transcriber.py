import subprocess
import os
import glob
import nemo.collections.asr as nemo_asr
import argparse

#quartznet is a better model, because it uses the mozilla voice for training
# To get some interesting errors use stt_en_conformer_ctc_small_ls.  It doesn't use mozilla voice for training

available_pretrained_models = [nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En"),
                    nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small_ls")]
wav_dir = "sound_wav/"

def convert_wav_mp3(mp3_file_path):
  '''convert mp3 to wav
    return path of wav
  '''
  os.makedirs(wav_dir, exist_ok=True)
  output_wav_path = wav_dir +mp3_file_path.split("/")[-1].split(".mp3")[0]+".wav"
  if mp3_file_path.endswith(".mp3"):
    print("Creating wav files in the sound_wav directory")
    os.system(
        f'ffmpeg -i {mp3_file_path} -acodec pcm_s16le -ac 1 -af aresample=resampler=soxr -ar 16000 {output_wav_path} -y'
    )
    return output_wav_path
  else:
    print("need mp3 path")

def transcribe_file(wav_file,pretrained_model):
  '''use the transcribe api to read wav files to return text'''
  print("Transcriptions:")
  txt = pretrained_model.transcribe(wav_file)
  print(txt)

def make_parser():
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser(description='simple pretrained_model transcriber')
  parser.add_argument("--sound_file_dir_or_path", default= "", type=str, help="Directory or path to sound file wav or mp3",required=True)
  return parser

def main(args):
    sound_path = args.sound_file_dir_or_path

    if sound_path.endswith('.wav'):
      transcribe_file(sound_path,available_pretrained_models[1])
      return
    elif os.path.isdir(sound_path):
      sound_list = glob.glob(sound_path+"/*")
      if all(a.endswith('wav') for a in sound_list):
        transcribe_file(sound_list,available_pretrained_models[1])
        return
      else:
        wav_list = [convert_wav_mp3(fle) for fle in sound_list]
        transcribe_file(wav_list,available_pretrained_models[1])
        return


if __name__ == '__main__':
    parser = make_parser()
    #if you want to run from notebook
    # args = parser.parse_args(['--sound_file_dir_or_path','/content/small/CV_unpacked/cv-corpus-6.1-2020-12-11/en/clips'])
    args = parser.parse_args()
    main(args)
