import torch
import torch.nn as nn
import torch.nn.functional as Fi
import torchaudio
import torchaudio.transforms as T
import argparse
import pickle
import pandas as pd
import librosa

hidden_size = 256
sample_rate = 16000

EOS_token =27
SOS_token =28

dictOfindex = {0: ' ',
 1: 'a',
 2: 'b',
 3: 'c',
 4: 'd',
 5: 'e',
 6: 'f',
 7: 'g',
 8: 'h',
 9: 'i',
 10: 'j',
 11: 'k',
 12: 'l',
 13: 'm',
 14: 'n',
 15: 'o',
 16: 'p',
 17: 'q',
 18: 'r',
 19: 's',
 20: 't',
 21: 'u',
 22: 'v',
 23: 'w',
 24: 'x',
 25: 'y',
 26: 'z',
 27: 'EOS',
 28: 'SOS',
 29: 'PAD'}


max_duration = 8
MAX_LENGTH = max_duration*100+1
mels_dims = 80
window_sz = 25
skip = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mel_spectrogram(object):

    def __init__(self):
        self.window_size = 25 / 1000
        self.stride = 10 / 1000
        self.sample_rate = 16000
        self.n_fft = int(self.window_size * self.sample_rate)
        self.win_length = None
        self.hop_length = int(self.sample_rate * self.stride)
        self.n_mels =mels_dims
        self.max_time = max_duration

    def mel_spectrogram(self, waveform):
        # mel_spec = T.MelSpectrogram(
        #         sample_rate=self.sample_rate,
        #         n_fft=self.n_fft,
        #         hop_length=self.hop_length,
        #         center=True,
        #         pad_mode="reflect",
        #         power=2.0,
        #         norm='slaney',
        #         onesided=True,
        #         n_mels=self.n_mels,
        #         mel_scale="htk")

        mels = librosa.feature.melspectrogram(waveform,
                                              sr=self.sample_rate,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length,
                                              center=True,
                                              n_mels=self.n_mels,
                                              htk=True,
                                              norm='slaney')

        # zero_pad = torch.zeros(1, self.sample_rate * self.max_time - waveform.size()[1])
        # padding = torch.cat([waveform, zero_pad], 1)
        # mels = mel_spec(waveform)
        # wave_spec = mels.swapaxes(1, 2)
        # get spectrogram
        wave_spec = mels.swapaxes(0, 1)
        wave_spec = torch.tensor(wave_spec).unsqueeze(0)

        return wave_spec


def evaluate(encoder, decoder, features, max_length=100, beam=1, nbest=1):
    input_tensor = features
    input_length = torch.tensor([input_tensor.shape[1]])

    decoded_words = []

    encoder_outputs = encoder(input_tensor, input_length)
    nbest_hyps = decoder.recognize_beam(encoder_outputs[0], beam, nbest)
    if beam > 1:
        word_index = nbest_hyps[0]['yseq']
    else:
        word_index = nbest_hyps[0]['yseq']
    decoded_word = [dictOfindex[a] for a in word_index]

    return decoded_word

def inference_from_file(wav_path, encoder, decoder):
    # Use the model to predict the label of the waveform
    waveform, sr = torchaudio.load(wav_path)

    #check sample rate
    if sr > sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    waveform = waveform[:, :max_duration * sample_rate]

    waveform = waveform.numpy()
    waveform = waveform.squeeze()

    transformer = Mel_spectrogram()
    mels = transformer.mel_spectrogram(waveform)

    output_words  = evaluate(encoder, decoder, mels)
    output_sentence = ''.join(output_words[1:-1])
    print("transcribe from file: ", output_sentence)
    return output_sentence

def main():
    parser = argparse.ArgumentParser(description='transcribe from file')
    parser.add_argument("--wav_path", type=str, default= "/Users/dami.osoba/work/bawk/src/data/small/train/wav/common_voice_en_21353435.wav",
                        help="path to wav file")
    parser.add_argument("--model_path", type=str,default = "/Users/dami.osoba/work/bawk/models/output/model_las_updated/model_las_updated_final.pth",
                        help="path to model arguments")
    parser.add_argument("--encoder_pkl_path", type=str,default='output/model_las_updated/encoder_las.pkl', help="path to model pickle_file")
    parser.add_argument("--decoder_pkl_path", type=str, default='output/model_las_updated/decoder_las.pkl',
                        help="path to model pickle_file")

    args= parser.parse_args()
    wav_path =  args.wav_path
    model_path = args.model_path
    encoder_pkl_path = args.encoder_pkl_path
    decoder_pkl_path = args.decoder_pkl_path

    with open(encoder_pkl_path, 'rb') as convert_file:
        encoder = pickle.load(convert_file)

    with open(decoder_pkl_path, 'rb') as convert_file:
        decoder = pickle.load(convert_file)

    # encoder = EncoderRNN(mels_dims*MAX_LENGTH, hidden_size).to(device)
    # attn_decoder = AttnDecoderRNN(hidden_size, 29, dropout_p=0.1).to(device)

    # load model weights state_dict
    checkpoint = torch.load(model_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    # enc_path = 'output/enc_model_new'
    # encoder.load_state_dict(torch.load(enc_path))
    # encoder.eval()
    #
    # dec_path = 'output/dec_model_new'
    # attn_decoder.load_state_dict(torch.load(dec_path))
    # attn_decoder.eval()

    path = "/Users/dami.osoba/work/bawk/small_dataset/small/CV_unpacked/cv-corpus-6.1-2020-12-11/en/validated.tsv"
    validated = pd.read_csv(path, sep='\t')

    actual = validated.set_index('path').loc[wav_path.split('/')[-1].replace('wav','mp3')]['sentence']
    print(actual)

    output_sentence = inference_from_file(wav_path,encoder,decoder)
    return output_sentence

if __name__ == "__main__":
    main()
