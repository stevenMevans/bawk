import torch
from constants import *
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sentence
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
            yay = torch.distributions.categorical.Categorical(decoder_probs)
            topi = yay.sample()
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dictOfindex[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1],decoder_output





def evaluateRandomly(transformed_dataset,encoder, decoder, n=1):
    for i in range(n):
        choice = np.random.randint(200)
        print(choice)
        actual = transformed_dataset[choice]['sentence']
        ex = transformed_dataset[choice]['waveform']
        output_words, attentions,_ = evaluate(encoder, decoder, ex)
        output_sentence = ''.join(output_words)
        print(actual, '<', output_sentence)
        print('')