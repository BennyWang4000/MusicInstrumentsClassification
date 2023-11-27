import torch
import os
from glob import glob
import librosa
import pandas as pd
import numpy as np


class InstrumentsDataset(torch.utils.data.Dataset):
    """Some Information about InstrumentsDataset"""

    def __init__(self, openmic_dir, class2idx_dict, sample_rate=16000, n_mels=128):
        super(InstrumentsDataset, self).__init__()
        self.label_df: pd.DataFrame = pd.read(os.path.join(
            openmic_dir, 'openmic-2018-aggregated-labels.csv'))
        self.audios = glob(os.path.join('audio', '*', '*.ogg'))
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.class2idx_dict = class2idx_dict

    def __getitem__(self, index):
        '''
        return 
        -----
            audio: 
            labels: list<int>
        '''
        sample_key = self.audios[index].split('/')[-1].replace('.ogg', '')
        labels = [self.class2idx_dict[c] for c in
                  self.label_df.loc[self.label_df['sample_key'] == sample_key]['instrument'].tolist()]

        audio, _ = librosa.load(self.audios[index])
        mel = librosa.feature.melspectrogram(
            audio, sr=self.sample_rate, n_mels=self.n_mels)

        log_mel = librosa.logamplitude(mel, ref_power=np.max)

        return log_mel, labels

    def __len__(self):
        return len(self.label_df.index())
