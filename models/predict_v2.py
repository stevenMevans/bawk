import torch
import torchaudio
from constants import *
from create_dataset import *
import numpy as np
from create_dataset import MelSpec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, features, max_length=100, beam=5, nbest=5):
    with torch.no_grad():
        input_tensor = features[0]
        target_tensor = features[1]
        input_length = features[2]

        input_tensor = features[0].float().to(device)
        target_tensor = features[1].long().to(device)
        input_length = features[2].long().to(device)
        decoded_words = []

        encoder_outputs, _ = encoder(input_tensor, input_length)
        nbest_hyps = decoder.recognize_beam(encoder_outputs[0], beam, nbest)
        word_index = nbest_hyps[0]['yseq']
        decoded_word = [dictOfindex[a] for a in word_index]

    return decoded_word

def evaluateRandomly(transformed_dataset,encoder, decoder, n=1):
    samp_1 = DataLoader(transformed_dataset, batch_size=1, collate_fn=pad_collate,
                        shuffle=True, num_workers=0)
    iterator = iter(samp_1)
    samp = iterator.next()

    for i in range(n):
        iterator = iter(samp_1)
        samp = iterator.next()
        actual = samp[3]
        output_words = evaluate(encoder, decoder, samp, max_length=100)
        output_sentence = ' '.join(output_words[1:-1])
        print(actual, '<', output_sentence)
        print('')

def evaluate_from_file(encoder, decoder, features,beam=5, nbest=5):
    with torch.no_grad():
        input_tensor = features.to(device)
        input_length = torch.tensor([features.size(1)], device=device)
        decoded_words = []

        encoder_outputs, _ = encoder(input_tensor, input_length)
        nbest_hyps = decoder.recognize_beam(encoder_outputs[0], beam, nbest)
        word_index = nbest_hyps[0]['yseq']
        decoded_word = [dictOfindex[a] for a in word_index]

    return decoded_word
