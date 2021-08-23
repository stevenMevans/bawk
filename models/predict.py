import torch
import torchaudio
from constants import *
import numpy as np
from create_dataset import MelSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, tens, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tens.reshape(1,1,mels_dims*MAX_LENGTH)
        input_length = input_tensor.size(2)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden,MAX_LENGTH)


        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_output = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention,decoder_probs = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            # yay = torch.distributions.categorical.Categorical(decoder_probs)
            # topi = yay.sample()
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dictOfindex[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1],decoder_output





def evaluateRandomly(transformed_dataset,encoder, decoder, n=10):
    for i in range(n):
        choice = np.random.randint(200)
        print(choice)
        actual = transformed_dataset[choice]['sentence']
        ex = transformed_dataset[choice]['waveform']
        output_words, attentions,_ = evaluate(encoder, decoder, ex)
        output_sentence = ''.join(output_words)
        print("#####################")
        print("GIVEN: ", actual, ' PREDICTED: ', output_sentence)
        print('')


def inference_from_file(wav_path, transcription, encoder, decoder):
    # Use the model to predict the label of the waveform
    waveform, _ = torchaudio.load(wav_path)
    sentence = transcription.encode(encoding="ascii", errors="ignore").decode().translate(table_trans)
    chars = [b for a in sentence for b in a]
    coded = [28,0] + [char_index[a] for a in chars] + [0,27]
    sample = {}
    sample['waveform'] = waveform
    sample['transcription'] = coded
    sample['sentence'] = sentence
    transformer = MelSpec()
    mels = transformer(sample)
    ex = mels['waveform']

    output_words, attentions, _ = evaluate(encoder, decoder, ex)
    output_sentence = ''.join(output_words)
    print("transcribe from file: ", output_sentence)
    return output_sentence
