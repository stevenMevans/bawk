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


def train(features, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()

    optimizer.zero_grad()

    input_tensor = features[0].float().to(device)
    target_tensor = features[1].long().to(device)
    input_length = features[2].long().to(device)

    batch_size = input_tensor.size(0)

    encoder_output, encoder_hidden = encoder(input_tensor, input_length)
    pred, actual = decoder(target_tensor, encoder_output)
    loss = criterion(pred, actual, ignore_index=PAD_token, reduction='mean')
    #     loss = loss[loss>0].sum()/batch_size
    # #     print(loss)
    # #     yuck.append(criterion(pred,actual,ignore_index=PAD_token,reduction='none'))

    loss.backward()
    optimizer.step()

    return loss.item()


def validate(features, encoder, decoder, optimizer, criterion):
    encoder.eval()
    decoder.eval()

    input_tensor = features[0].float().to(device)
    target_tensor = features[1].long().to(device)
    input_length = features[2].long().to(device)

    batch_size = input_tensor.size(0)

    encoder_output, encoder_hidden = encoder(input_tensor, input_length)
    pred, actual = decoder(target_tensor, encoder_output)
    loss = criterion(pred, actual, ignore_index=PAD_token, reduction='mean')
    #     loss = loss[loss>0].sum()/batch_size
    # #     print(loss)
    # #     yuck.append(criterion(pred,actual,ignore_index=PAD_token,reduction='none'))

    return loss


def save_checkpoint(epoch, encoder, decoder, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, '../models/output/model.pth')


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001, reload=False):
    start = time.time()
    plot_losses = []
    plot_losses_val = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_loss_total_val = 0
    plot_loss_total_val = 0

    lns = len(transformed_dataset)
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                 lr=learning_rate)

    if reload:
        print("loading previously trained model")
        checkpoint = torch.load('../models/output/model.pth', map_location='cpu')
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder = encoder.to(device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder = decoder.to(device)
        encoder.train()
        decoder.train()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()

    #     criterion = nn.NLLLoss()
    criterion = Fi.cross_entropy
    len_all = len(transformed_dataset)
    len_train = int(0.8 * len_all)
    train_ds, val_ds = torch.utils.data.random_split(transformed_dataset, (len_train, len_all - len_train))

    for i in range(1, n_iters):
        rand_sampler = torch.utils.data.RandomSampler(val_ds)
        train_sampler = DataLoader(train_ds, batch_size=32, sampler=rand_sampler, collate_fn=pad_collate)

        #         rand_val_sampler = torch.utils.data.RandomSampler(val_ds)
        #         val_sampler = DataLoader(transformed_dataset, batch_size=10, sampler=rand_val_sampler,collate_fn=pad_collate)

        iterator = iter(train_sampler)
        features = iterator.next()
        loss = train(features, encoder, decoder, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        #         iter_val = iter(val_sampler)
        #         features_val = iter_val.next()
        #         loss_val = validate(features_val, encoder, decoder, optimizer, criterion)
        #         print_loss_total_val += loss_val
        #         plot_loss_total_val += loss_val

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print_loss_avg_val = print_loss_total_val / print_every
            print_loss_total_val = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))
            plot_losses.append(print_loss_avg)
            #             plot_losses_val.append(print_loss_avg_val)
            save_checkpoint(i, encoder, decoder, optimizer, loss)