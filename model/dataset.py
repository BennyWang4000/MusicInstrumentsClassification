# %%
import torch
import os
from glob import glob
import pandas as pd
import torchaudio
from torchvggish.torchvggish.vggish import VGGish, vggish_input
import torchaudio.functional as F
# %%


class InstrumentsDataset(torch.utils.data.Dataset):
    def __init__(self, openmic_dir: str, inst2idx_dict: dict[str, int], classes: int, device: str, is_pre_trained=None, signal_args=None, **kwargs):
        '''
        Args:
            openmic_dir (str)
            inst2idx_dict (dict[str, int])
            classes (int)
            device (str)
            is_pre_trained (str, optional): Defaults to None.
        '''
        super(InstrumentsDataset, self).__init__()
        self.label_df: pd.DataFrame = pd.read_csv(os.path.join(
            openmic_dir, 'openmic-2018-aggregated-labels.csv'))
        self.audios = glob(os.path.join(
            openmic_dir, 'audio', '**', '*.ogg'), recursive=True)

        self.is_pre_trained = is_pre_trained
        self.inst2idx_dict = inst2idx_dict
        self.classes = classes
        self.device = device

        if is_pre_trained:
            assert is_pre_trained in [
                'vggish'], 'assert pre-trained model in ["vggish",].'
            self.pre_trained = is_pre_trained
            self.pre_trained_model = VGGish(
                kwargs['pre_trained_path'], postprocess=False, preprocess=False)
            self.pre_trained_model.eval()
        if signal_args != None:
            self.audio_length = signal_args['audio_length']
            self.sample_rate = signal_args['sample_rate']
            self.n_mels = signal_args['n_mels']
            self.n_fft = signal_args['n_fft']
            self.hop_length = signal_args['hop_length']
            self.f_min = signal_args['f_min']
            self.f_max = signal_args['f_max']
            self.n_freqs = signal_args['n_freqs']
            self.win_length = signal_args['win_length']

    def __getitem__(self, index):
        sample_key = self.audios[index].split('/')[-1].replace('.ogg', '')
        y_labels = [0] * self.classes
        labels = [self.inst2idx_dict.get(c, 99) for c in
                  self.label_df.loc[self.label_df['sample_key'] == sample_key]['instrument'].tolist()]

        for label in labels:
            if label < self.classes:
                y_labels[label] = 1

        if self.pre_trained:
            if self.pre_trained == 'vggish':
                sample_input = self.pre_trained_model.forward(
                    vggish_input.wavfile_to_examples(self.audios[index]))

        # # else:
        # waveform, sr = torchaudio.load(self.audios[index])

        # if sr != self.sample_rate:
        #     waveform = F.resample(
        #         waveform, sr, self.sample_rate, resampling_method="sinc_interp_kaiser")
        # waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        # # while waveform.size()[1] < self.audio_length:
        # #     waveform = torch.concat((waveform, waveform), 1)
        # waveform = waveform[:, :self.audio_length]

        # # * ---------------------------------------------------------------------------- #
        # # *                                   mel spec                                   #
        # # * ---------------------------------------------------------------------------- #

        # to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sr, n_fft=self.n_fft, n_mels=self.n_mels,
        #     hop_length=self.hop_length, f_min=self.f_min, f_max=self.f_max, win_length=self.win_length)

        # log_mel = to_mel_spectrogram(waveform).log()

        # # * ---------------------------------------------------------------------------- #
        # # *                                     mfcc                                     #
        # # * ---------------------------------------------------------------------------- #
        # transform = torchaudio.transforms.MFCC(
        #     sample_rate=sr,
        #     n_mfcc=10,
        #     melkwargs={"n_fft": 256, "hop_length": 1252, "n_mels": 10, "center": False},)
        # mfcc = transform(waveform)
        # mfcc = torch.Tensor(mfcc).to(self.device)
        # mfcc[mfcc == float('-inf')] = -1.0e+09

        # return t.to(self.device), torch.Tensor(y_labels).to(self.device), log_mel.squeeze(0).to(self.device), mfcc.squeeze(0).to(self.device)
        # return sample_input.to(self.device), torch.Tensor(y_labels).to(self.device), mfcc.to(self.device).squeeze(0)
        return sample_input.to(self.device), torch.Tensor(y_labels).to(self.device)

    def __len__(self):
        return len(self.audios)
