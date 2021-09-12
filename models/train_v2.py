from constants import *
import matplotlib.pyplot as plt
from las_create_dataset import pad_collate
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
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(features, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()

    optimizer.zero_grad()

    input_tensor = features[0].float().to(device)
    target_tensor = features[1].long().to(device)
    input_length = features[2].long().to(device)

    batch_size = input_tensor.size(0)

    encoder_output = encoder(input_tensor, input_length)
    pred, actual = decoder(target_tensor, encoder_output)
    loss = criterion(pred, actual, ignore_index=PAD_token, reduction='mean')
    #     loss = loss[loss>0].sum()/batch_size
    # #     print(loss)
    # #     yuck.append(criterion(pred,actual,ignore_index=PAD_token,reduction='none'))

    loss.backward()
    optimizer.step()

    return loss.item()


def validate(features, encoder, decoder, optimizer, criterion):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        input_tensor = features[0].float().to(device)
        target_tensor = features[1].long().to(device)
        input_length = features[2].long().to(device)


        encoder_output = encoder(input_tensor, input_length)
        pred, actual = decoder(target_tensor, encoder_output)
        loss = criterion(pred, actual, ignore_index=PAD_token, reduction='mean')
        #     loss = loss[loss>0].sum()/batch_size
        # #     print(loss)
        # #     yuck.append(criterion(pred,actual,ignore_index=PAD_token,reduction='none'))

    return loss


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
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                 lr=learning_rate)

    if reload_path:
        checkpoint = torch.load(f'output/{model_save_path}', map_location='cpu')
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


    criterion = Fi.cross_entropy
    len_all = len(transformed_dataset)
    len_train = int(0.9 * len_all)
    train_ds, val_ds = torch.utils.data.random_split(transformed_dataset, (len_train, len_all - len_train))
    train_sampler = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=pad_collate)
    val_sampler = DataLoader(val_ds, batch_size=10, shuffle=True, collate_fn=pad_collate)
    iterator = iter(train_sampler)
    val_sampler = DataLoader(val_ds, batch_size=32, shuffle=True, collate_fn=pad_collate)
    iter_val = iter(val_sampler)

    for i in range(1, n_iters):

        features = iterator.next()
        loss = train(features, encoder, decoder, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i % 10 == 0:
            features_val = iter_val.next()
            loss_val = validate(features_val, encoder, decoder, optimizer, criterion)
            print_loss_avg_val = loss_val

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            plot_losses_train.append(print_loss_avg)
            plot_losses_valid.append(print_loss_avg_val)

            print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, i / n_iters),
                                              i, i / n_iters * 100, print_loss_avg, print_loss_avg_val))
            loss_dict['train'] = torch.tensor(plot_losses_train).to('cpu').numpy()
            loss_dict['valid'] = torch.tensor(plot_losses_valid).to('cpu').numpy()

            if not reload_path:
                save_checkpoint(iter, encoder, decoder, optimizer, loss, model_save_path)
                write_loss(loss_path, loss_dict)
            else:
                save_checkpoint(iter, encoder, decoder, optimizer, loss, reload_path)
                write_loss(loss_path, loss_dict)

    return loss_dict