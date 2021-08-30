from __future__ import print_function, division, unicode_literals

import json
import pandas as pd
import string
import torch
import torchaudio
import torchaudio.transforms as T
from io import open
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from models.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

table_trans = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}

train_path = []
train_text = []


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

        waveform, _ = torchaudio.load(self.path_frame.loc[idx][0], )

        # transcription for audio
        trans = self.path_frame.loc[idx][1]
        # encode to ascii
        trans = trans.encode(encoding="ascii", errors="ignore").decode().translate(table_trans).lower()
        chars = [b for a in trans for b in a]
        coded = [SOS_token] + [char_index[a] for a in chars] + [EOS_token]

        sample = {'waveform': waveform, 'transcription': coded, 'sentence': trans}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MelSpec(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        self.window_size = window_sz / 1000
        self.stride = skip / 1000
        self.sample_rate = sample_rate
        self.n_fft = int(self.window_size * self.sample_rate)
        self.win_length = None
        self.hop_length = int(self.sample_rate * self.stride)
        self.n_mels = mels_dims
        self.max_time = max_duration
        pass

    #         assert isinstance(output_size, (int, tuple))
    #         self.output_size = output_size

    def mel_spectrogram(self, a):
        mel_spec = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=self.n_mels,
            mel_scale="htk")
        return mel_spec(a)

    def __call__(self, sample):
        waveform, transcription, sentence = sample['waveform'], sample['transcription'], sample['sentence']
        # zero pad waveform
        zero_pad = torch.zeros(1, self.sample_rate * self.max_time - waveform.size()[1])
        padding = torch.cat([waveform, zero_pad], 1)
        # get spectrogram
        wave_spec = self.mel_spectrogram(padding)
        wave_spec = wave_spec.swapaxes(1, 2)
        # change transcription list to tensor
        transcription = torch.tensor(transcription, dtype=torch.long, device=device)

        return {'waveform': wave_spec, 'transcription': transcription, 'sentence': sentence}


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def preprocess(train_manifest_path):
    # train_manifest_path = '/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json'
    train_manifest_data = read_manifest(train_manifest_path)
    # keep audio < 4s
    train_path = [(data['audio_filepath'], data['text']) for data in train_manifest_data if
                  data['duration'] <= max_duration]
    train_path_pd = pd.DataFrame(train_path, columns=['train_path', 'sentence'])
    transformed_dataset = VoiceDataset(path_list=train_path_pd, transform=MelSpec())
    return transformed_dataset
