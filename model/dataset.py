import torch
import os
from glob import glob
import pandas as pd
import torchaudio
from torchvggish.torchvggish.vggish import VGGish, vggish_input


class InstrumentsDataset(torch.utils.data.Dataset):

    def __init__(self, openmic_dir, inst2idx_dict, classes, device, is_pre_trained=False, **kwargs):
        super(InstrumentsDataset, self).__init__()
        kwargs = kwargs['kwargs']
        self.label_df: pd.DataFrame = pd.read_csv(os.path.join(
            openmic_dir, 'openmic-2018-aggregated-labels.csv'))
        self.audios = glob(os.path.join(
            openmic_dir, 'audio', '**', '*.ogg'), recursive=True)

        self.is_pre_trained = is_pre_trained
        self.inst2idx_dict = inst2idx_dict
        self.classes = classes
        self.device = device

        if is_pre_trained:
            assert kwargs['pre_trained'] in [
                'vggish'], 'assert pre-trained model in ["vggish",].'
            self.pre_trained = kwargs['pre_trained']

            if self.pre_trained == 'vggish':
                self.pre_trained_model = VGGish(
                    kwargs['pre_trained_path'], postprocess=False, preprocess=False)
                self.pre_trained_model.eval()
        else:
            self.audio_length = kwargs['audio_length']
            self.sample_rate = kwargs['sample_rate']
            self.n_mels = kwargs['n_mels']
            self.n_fft = kwargs['n_fft']
            self.hop_length = kwargs['hop_length']
            self.f_min = kwargs['f_min']
            self.f_max = kwargs['f_max']
            self.n_freqs = kwargs['n_freqs']
            self.win_length = kwargs['win_length']
            self.pre_trained = kwargs['pre_trained']

    def __getitem__(self, index):
        sample_key = self.audios[index].split('/')[-1].replace('.ogg', '')
        y_labels = [0] * self.classes
        labels = [self.inst2idx_dict.get(c, 99) for c in
                  self.label_df.loc[self.label_df['sample_key'] == sample_key]['instrument'].tolist()]

        for label in labels:
            if label < self.classes:
                y_labels[label] = 1

        if self.is_pre_trained:
            if self.pre_trained == 'vggish':
                example = vggish_input.wavfile_to_examples(self.audios[index])
                t = self.pre_trained_model.forward(example)
        else:
            waveform, sr = torchaudio.load(self.audios[index])
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)
            while waveform.size()[1] < self.audio_length:
                waveform = torch.concat((waveform, waveform), 1)
            waveform = waveform[:, :self.audio_length]

            # * ---------------------------------------------------------------------------- #
            # *                                   mel spec                                   #
            # * ---------------------------------------------------------------------------- #

            to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=self.n_fft, n_mels=self.n_mels,
                hop_length=self.hop_length, f_min=self.f_min, f_max=self.f_max, win_length=self.win_length)

            t = to_mel_spectrogram(waveform).log()

            # * ---------------------------------------------------------------------------- #
            # *                                     mfcc                                     #
            # * ---------------------------------------------------------------------------- #
            # transform = torchaudio.transforms.MFCC(
            #     sample_rate=sr,
            #     n_mfcc=128,
            #     melkwargs={"n_fft": 1024, "hop_length": 441, "n_mels": 128, "center": False},)
            # mfcc = transform(waveform)

            # t = torch.Tensor(mfcc).to(self.device)
            # t[t == float('-inf')] = -1.0e+09

        return t.to(self.device), torch.Tensor(y_labels).to(self.device), 0

    def __len__(self):
        return len(self.audios)
