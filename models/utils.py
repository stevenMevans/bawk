import math
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

from matplotlib import ticker


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


def save_checkpoint(epoch, encoder, decoder, optimizer, loss,model_save):
    torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'output/{model_save}.pth')


def write_loss(write_path,losses_dict):
    loss_pd = pd.DataFrame(losses_dict)
    loss_pd.to_csv(f"output/{write_path}.csv",header=True)