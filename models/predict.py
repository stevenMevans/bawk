import numpy as np
import torch
import torchaudio

from models.constants import *
from models.create_dataset import MelSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, tens, greedy=True, max_length=MAX_LENGTH):
    # greedy = greedy coding or based on sampling from distribution
    with torch.no_grad():
        input_tensor = tens.reshape(1, 1, mels_dims * MAX_LENGTH)
        input_length = input_tensor.size(2)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden, MAX_LENGTH)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_output = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, decoder_probs = decoder(
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

        return decoded_words, decoder_attentions[:di + 1], decoder_output


def evaluateRandomly(transformed_dataset, encoder, decoder, n=10, greedy=True):
    # evaluate some sample data
    for i in range(n):
        choice = np.random.randint(200)
        print(choice)
        actual = transformed_dataset[choice]['sentence']
        ex = transformed_dataset[choice]['waveform']
        output_words, attentions, _ = evaluate(encoder, decoder, ex, greedy)
        output_sentence = ''.join(output_words[1:-1])
        print("#####################")
        print("GIVEN: ", actual, ' PREDICTED: ', output_sentence)
        print('')


def inference_from_file(wav_path, encoder, decoder, greedy=True):
    # Use the model to predict the label of the waveform
    waveform, sr = torchaudio.load(wav_path)

    # check sample rate
    if sr > sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    sample = {}
    sample['waveform'] = waveform
    sample['transcription'] = []
    sample['sentence'] = ""
    transformer = MelSpec()
    mels = transformer(sample)
    ex = mels['waveform']

    output_words, attentions, _ = evaluate(encoder, decoder, ex, greedy)
    output_sentence = ''.join(output_words[1:-1])
    print("transcribe from file: ", output_sentence)
    return output_sentence
