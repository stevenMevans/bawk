from __future__ import print_function, division,unicode_literals
from io import open
from tqdm.auto import tqdm
import string
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import librosa


import torchaudio
import torchaudio.transforms as T
from models.constants import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

table_trans = str.maketrans(dict.fromkeys(string.punctuation))  # OR {key: None for key in string.punctuation}

train_path = []
train_text = []


def mel_spec(a):
    window_size = window_sz / 1000
    stride = skip / 1000
    n_fft = int(window_size * sample_rate)
    hop_length = int(sample_rate * stride)
    n_mels = mels_dims
    max_time = max_duration

    return librosa.feature.melspectrogram(a,
                                          sr=sample_rate,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          center=True,
                                          n_mels=n_mels,
                                          htk=True,
                                          norm='slaney'
                                          )


def mel_from_wav(wav_path):
    # Use the model to predict the label of the waveform
    waveform, sr = librosa.load(wav_path)
    hmm = waveform.squeeze()
    wave_spec = mel_spec(hmm)
    wave_spec = wave_spec.swapaxes(0, 1)
    wave_spec = torch.tensor(wave_spec, device=device)
    return wave_spec


class VoiceDataset(Dataset):
    def __init__(self, path_list,transform=None):
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
        coded = [SOS_token]+[char_index[a] for a in chars] + [EOS_token]

        sample = {'waveform': waveform, 'transcription': coded, 'sentence': trans}

        if self.transform:
            sample = self.transform(sample)

        return sample


class torch_mel(object):

    def __init__(self):
        self.window_size = 25 / 1000
        self.stride = 10 / 1000
        self.sample_rate = 16000
        self.n_fft = int(self.window_size * self.sample_rate)
        self.win_length = None
        self.hop_length = int(self.sample_rate * self.stride)
        self.n_mels =mels_dims
        self.max_time = max_duration

    def mel_spectrogram(self, waveform):
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
        zero_pad = torch.zeros(1, self.sample_rate * self.max_time - waveform.size()[1])
        padding = torch.cat([waveform, zero_pad], 1)
        # get spectrogram
        wave_spec = mel_spec(padding)
        wave_spec = wave_spec.swapaxes(1, 2)
        return wave_spec



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

def preprocess(train_manifest_path,cloud=False):
    # train_manifest_path = '/Users/dami.osoba/work/bawk/small_dataset/commonvoice_train_manifest.json'
    train_manifest_data = read_manifest(train_manifest_path)
    # keep audio < 4s
    if not cloud:
        train_path = [(data['audio_filepath'], data['text']) for data in train_manifest_data if data['duration'] <= max_duration]
    else:
        train_path = [(data['audio_filepath'].replace('validated','wav_clips'), data['text']) for data in train_manifest_data
                      if data['duration'] <= max_duration]

    train_path_pd = pd.DataFrame(train_path, columns=['train_path','sentence'])
    transformed_dataset = VoiceDataset(path_list=train_path_pd,transform=MelSpec())
    return transformed_dataset


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

        feature = np.zeros((max_input_len, input_dim), dtype=np.float32)
        feature[:f.shape[0], :f.shape[1]] = f
        feature2 = torch.tensor(feature.copy(), dtype=torch.float32, device=device)

        trn2 = np.pad(trn.cpu(), (0, max_target_len - len(trn)), 'constant', constant_values=29)
        trn3 = torch.tensor(trn2.copy(), dtype=torch.long, device=device)
        batch[i] = (feature2, trn2, input_length, sentence)
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



