import torch
import torch.nn as nn
import torch.nn.functional as Fi

from constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #         self.embedding = nn.Embedding(input_size, hidden_size,)
        self.embedding = nn.GRU(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, total_length):
        output, _ = self.embedding(input)

        output, hidden = self.gru(output, hidden)
        # m = nn.MaxPool1d(MAX_LENGTH)
        # yy = m(hidden.swapaxes(1, 2))
        # yy = yy.swapaxes(1, 2)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, MAX_LENGTH, self.hidden_size, device=device)
        # return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        #         output = self.embedding(input)
        output = Fi.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size , self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size , self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.out = nn.Linear((self.hidden_size)*MAX_LENGTH, output_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
#         embedded ,_= self.embedding(input)

        embedded = self.dropout(embedded)
        rng = embedded.size(1)

        for t in range(rng):

            attn_weights = Fi.softmax(
                self.attn(torch.cat((embedded[:,t,:], hidden[0]), 1)), dim=1)

            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))
            output = torch.cat((embedded[:,t,:], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)
            output = Fi.relu(output)
            output, hidden = self.gru(output, hidden)
            output = Fi.log_softmax(self.out(output[0]), dim=1)
            output_probs = torch.exp(output)


        output = Fi.log_softmax(hmm[0], dim=1)
        output_probs = torch.exp(output)


        return output, hidden, attn_weights,output_probs

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)