import pandas as pd

from constants import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn

import random
import time
from utils import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion,total_length, max_length=MAX_LENGTH):
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden()

    optimizer.zero_grad()



    input_tensor = input_tensor.float().to(device)
    target_tensor = target_tensor.long().to(device)

    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden,total_length)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention,output_probs = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention,output_probs = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input


            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length




def validate(input_tensor, target_tensor, encoder, decoder, criterion,total_length, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()

        input_tensor = input_tensor.float().to(device)
        target_tensor = target_tensor.long().to(device)

        target_length = target_tensor.size(0)
        encoder.eval()
        decoder.eval()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden,total_length)

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention,output_probs = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention,output_probs = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input


                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

    return loss.item() / target_length


def trainIters(transformed_dataset,encoder, decoder, n_iters,  model_save_path, print_every=1000, learning_rate=0.0001,reload_path=None):
    start = time.time()
    plot_losses_train = []
    plot_losses_valid = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_loss_total_val = 0
    plot_loss_total_val = 0
    loss_dict ={}
    if not reload_path:
        loss_path = model_save_path+"_losses"
    else:
        loss_path = reload_path+"_losses"



    lns = len(transformed_dataset)
    shuffled = np.array(range(lns))
    np.random.shuffle(shuffled)
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                 lr=learning_rate)

    if reload_path:
        checkpoint = torch.load(f'output/{model_save_path}.pth', map_location='cpu')
        # load model weights state_dict
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder = encoder.to(device)
        encoder.train()
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder = decoder.to(device)
        decoder.train()
        optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                     lr=learning_rate)

        print('Previously trained model weights state_dict loaded...')
        # load trained optimizer state_dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['loss']

    train_ln = int(0.8 * lns)
    train_range = shuffled[:train_ln]
    valid_range = shuffled[train_ln:]

    training_examples = np.random.choice(train_range, n_iters)
    valid_examples = np.random.choice(valid_range, n_iters)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters):
        training_pair = transformed_dataset[training_examples[iter - 1]]
        # input_tensor = training_pair['waveform']
        input_tensor = training_pair['waveform'].reshape(1, 1, mels_dims * MAX_LENGTH)
        target_tensor = torch.tensor(training_pair['transcription'], dtype=torch.long, device=device).view(-1, 1)
        tot = input_tensor.size(1)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, tot)
        print_loss_total += loss

        training_pair = transformed_dataset[valid_examples[iter - 1]]
        input_tensor = training_pair['waveform'].reshape(1, 1, mels_dims * MAX_LENGTH)
        target_tensor = torch.tensor(training_pair['transcription'], dtype=torch.long, device=device).view(-1, 1)
        tot = input_tensor.size(1)

        loss_val = validate(input_tensor, target_tensor, encoder,
                            decoder, criterion, tot)
        print_loss_total_val += loss_val


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            plot_losses_train.append(print_loss_avg)

            print_loss_avg_val = print_loss_total_val / print_every
            print_loss_total_val = 0
            plot_losses_valid.append(print_loss_avg_val)

            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / n_iters),
                                              iter, iter / n_iters * 100, print_loss_avg, print_loss_avg_val))
            loss_dict['train'] = plot_losses_train
            loss_dict['valid'] = plot_losses_valid

            if not reload_path:
                save_checkpoint(iter, encoder, decoder, optimizer, loss, model_save_path)
                write_loss(loss_path,loss_dict)
            else:
                save_checkpoint(iter, encoder, decoder, optimizer, loss, reload_path)
                write_loss(loss_path, loss_dict)

    return loss_dict





