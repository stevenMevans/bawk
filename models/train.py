from constants import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import optim
import torch.nn as nn

import random
import time
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion,total_length, max_length=MAX_LENGTH):
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


def save_checkpoint(epoch, encoder, decoder, optimizer, loss,model_save):
    torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'../models/output/{model_save}.pth')

#
# def val(dataset):
#     dataset_size = len(dataset)
#     validation_split = .2
#     indices = list(range(dataset_size))
#     shuffled = np.shuffle(indices)
#     split = int(validation_split * dataset_size)
#     train_indices, val_indices = indices[split:], indices[:split]
#     val_dataset = Subset(dataset, val_indices)



def trainIters(transformed_dataset,encoder, decoder, n_iters,model_save, print_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    lns = len(transformed_dataset)

    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                 lr=learning_rate)

    training_examples = np.random.choice(lns - 1, n_iters)
    # val_size = int(0.1*len(training_examples))+1

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters):
        training_pair = transformed_dataset[training_examples[iter - 1]]
        input_tensor = training_pair['waveform']
        # input_tensor = training_pair['waveform'].reshape(1,1,mels_dims*MAX_LENGTH)


        target_tensor = torch.tensor(training_pair['transcription'], dtype=torch.long, device=device).view(-1, 1)
        tot = input_tensor.size(1)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, tot)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            save_checkpoint(iter, encoder, decoder, optimizer, loss,model_save)


def reload(transformed_dataset,encoder,decoder, n_iters, print_every=1000, learning_rate=0.01):
    checkpoint = torch.load('output/model_simple.pth',map_location='cpu')
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

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    lns = len(transformed_dataset)

    training_examples = np.random.choice(lns - 1, n_iters)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters):
        training_pair = transformed_dataset[training_examples[iter - 1]]
        # input_tensor = training_pair['waveform'].to(device)
        input_tensor = training_pair['waveform'].reshape(1,1,mels_dims*MAX_LENGTH).to(device)


        target_tensor = torch.tensor(training_pair['transcription'], dtype=torch.long, device=device).view(-1, 1)
        tot = input_tensor.size(1)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, tot)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            save_checkpoint(iter, encoder, decoder, optimizer, loss)




