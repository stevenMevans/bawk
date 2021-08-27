from predict import inference_from_file
import torch
import torch.nn as nn
import torch.nn.functional as Fi
import torchaudio
import torchaudio.transforms as T

hidden_size = 100
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

MAX_LENGTH = 401
max_duration = 4
mels_dims = 80
window_sz = 25
skip = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #         self.embedding = nn.Embedding(input_size, hidden_size,)
        self.embedding = nn.GRU(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, total_length):
        output, _ = self.embedding(input)

        output, hidden = self.gru(output, hidden)
        # m = nn.MaxPool1d(MAX_LENGTH)
        # yy = m(hidden.swapaxes(1, 2))
        # yy = yy.swapaxes(1, 2)
        return output, hidden

    def initHidden(self):
        # return torch.zeros(1, MAX_LENGTH, self.hidden_size, device=device)
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
#         embedded ,_= self.embedding(input)

        embedded = self.dropout(embedded)

        attn_weights = Fi.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = Fi.relu(output)
        output, hidden = self.gru(output, hidden)

        output = Fi.log_softmax(self.out(output[0]), dim=1)
        output_probs = torch.exp(output)
        return output, hidden, attn_weights,output_probs

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

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
        mel_spec = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                onesided=True,
                n_mels=self.n_mels,
                mel_scale="htk")
        zero_pad = torch.zeros(1, self.sample_rate * self.max_time - waveform.size()[1])
        padding = torch.cat([waveform, zero_pad], 1)
        # get spectrogram
        wave_spec = mel_spec(padding)
        wave_spec = wave_spec.swapaxes(1, 2)
        return wave_spec


def evaluate(encoder, decoder, tens,greedy=True, max_length=MAX_LENGTH):
    # greedy = greedy coding or based on sampling from distribution
    with torch.no_grad():
        input_tensor = tens.reshape(1,1,mels_dims*MAX_LENGTH)
        input_length = input_tensor.size(2)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden,MAX_LENGTH)

        decoder_input = torch.tensor([[28]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_output = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention,decoder_probs = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            if greedy:
                topv, topi = decoder_output.data.topk(1)
            else:
                yay = torch.distributions.categorical.Categorical(decoder_probs)
                topi = yay.sample()
            if topi.item() == EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(dictOfindex[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1],decoder_output

def inference_from_file(wav_path, encoder, decoder,greedy=True):
    # Use the model to predict the label of the waveform
    waveform, sr = torchaudio.load(wav_path)

    #check sample rate
    if sr > sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    transformer = Mel_spectrogram()
    mels = transformer.mel_spectrogram(waveform)

    output_words, attentions, _ = evaluate(encoder, decoder, mels,greedy)
    output_sentence = ''.join(output_words[1:-1])
    print("transcribe from file: ", output_sentence)
    return output_sentence

encoder = EncoderRNN(mels_dims*MAX_LENGTH, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, 29, dropout_p=0.1).to(device)

enc_path = 'enc_model'
encoder.load_state_dict(torch.load(enc_path))
encoder.eval()

dec_path = 'dec_model'
attn_decoder.load_state_dict(torch.load(dec_path))
attn_decoder.eval()

inference_from_file("/Users/dami.osoba/work/bawk/src/data/small/train/wav/common_voice_en_119561.wav",encoder,attn_decoder,greedy=False)
