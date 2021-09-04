from constants import *
import matplotlib.pyplot as plt
from create_dataset import pad_collate
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as Fi


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import random
import time
import math

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



def train(features, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = features[0]
    target_tensor = features[1]
    input_length = features[2]

    batch_size = input_tensor.size(0)

    encoder_output, encoder_hidden = encoder(input_tensor ,input_length)
    pred ,actual = decoder(target_tensor ,encoder_output)
    loss = criterion(pred ,actual ,ignore_index=PAD_token ,reduction='sum')


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() /batch_size


def trainIters(transformed_dataset,encoder, decoder, n_iters, print_every=1000, learning_rate=0.01,bth_size=10):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    lns = len(transformed_dataset)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    #     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    #     criterion = nn.NLLLoss()
    criterion = Fi.cross_entropy

    for i in range(1, n_iters):
        rand_sampler = torch.utils.data.RandomSampler(transformed_dataset, replacement=False)
        train_sampler = DataLoader(transformed_dataset, batch_size=10, sampler=rand_sampler, collate_fn=pad_collate)
        iterator = iter(train_sampler)
        features = iterator.next()
        loss = train(features, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            plot_losses.append(print_loss_avg)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))



