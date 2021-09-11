from __future__ import print_function, division, unicode_literals
import os
import pandas as pd
# from skimage import io, transform
import numpy as np
import librosa

from io import open

import tqdm
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from constants import *

import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window_size = 25 / 1000
stride = 10 / 1000
sample_rate = 16000
n_fft = int(window_size * sample_rate)
win_length = None
hop_length = int(sample_rate * stride)
n_mels = 80
max_time = 8


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm.tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


class VoiceDataset(Dataset):
    def __init__(self, path_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #         self.landmarks_frame = pd.read_csv(csv_file)
        #         self.root_dir = root_dir
        self.transform = transform
        self.path_frame = path_list

    def __len__(self):
        return len(self.path_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        waveform, _ = librosa.load(self.path_frame.loc[idx][0], sr=16000)
        trans = self.path_frame.loc[idx][1]
        # encode to ascii
        trans = trans.encode(encoding="ascii", errors="ignore").decode().translate(table_trans).lower()
        chars = [b for a in trans for b in a]
        coded = [char_index[a] for a in chars]

        sample = {'waveform': waveform, 'transcription': coded, 'sentence': trans}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MelSpeci(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        self.window_size = 25 / 1000
        self.stride = 10 / 1000
        self.sample_rate = 16000
        self.n_fft = int(self.window_size * self.sample_rate)
        self.win_length = None
        self.hop_length = int(self.sample_rate * self.stride)
        self.n_mels = 80
        self.max_time = 8

    def mel_spectrogram(self, a):
        return librosa.feature.melspectrogram(a,
                                              sr=self.sample_rate,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length,
                                              center=True,
                                              n_mels=self.n_mels,
                                              htk=True,
                                              norm='slaney'
                                              )

    def __call__(self, sample):
        waveform, transcription, sentence = sample['waveform'], sample['transcription'], sample['sentence']
        self.hmm = waveform.squeeze()
        wave_spec = self.mel_spectrogram(self.hmm)
        wave_spec = wave_spec.swapaxes(0, 1)
        wave_spec = torch.tensor(wave_spec, device=device)
        # change transcription list to tensor
        transcription = torch.tensor(transcription, dtype=torch.long, device=device)

        return {'waveform': wave_spec, 'transcription': transcription, 'sentence': sentence}


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature = elem['waveform']
        feature = feature.squeeze()
        trn = elem['transcription']
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        f = elem['waveform']
        trn = elem['transcription']
        sentence = elem['sentence']
        f = f.squeeze()
        input_length = f.shape[0]
        input_dim = f.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float32)
        feature[:f.shape[0], :f.shape[1]] = f.cpu()
        trn = np.pad(trn.cpu(), (0, max_target_len - len(trn)), 'constant', constant_values=29)
        batch[i] = (feature, trn, input_length, sentence)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, sentence = [], [], []

    # Gather in lists, and encode labels as indices
    for a in batch:
        tensors += [a['waveform']]
        targets += [a['transcription']]
        sentence += [a['sentence']]

    # Group the list of tensors into a batched tensor
    tensors = tensors
    #     targets = torch.stack(targets)
    targets = pad_sequence(targets)

    return tensors, targets, sentence


def preprocessi(train_manifest_path, cloud=False):
    # train_manifest_path = '/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json'
    train_manifest_data = read_manifest(train_manifest_path)
    # keep audio < 4s
    if not cloud:
        train_path = [(data['audio_filepath'], data['text']) for data in train_manifest_data if
                      data['duration'] <= max_duration]
    else:
        train_path = [(data['audio_filepath'].replace('validated', 'wav_clips'), data['text']) for data in
                      train_manifest_data
                      if data['duration'] <= max_duration]

    train_path_pd = pd.DataFrame(train_path, columns=['train_path', 'sentence'])
    transformed_dataset = VoiceDataset(path_list=train_path_pd, transform=MelSpeci())
    return transformed_dataset
