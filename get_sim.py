'''
- get 5000 instruments
- random pick 500
- correlation
- calculate by matrix
'''
from model import TransformerClassifier
from torchvggish.torchvggish.vggish import VGGish, vggish_input
import torch.nn.functional as F
import torch
import pandas as pd
from glob import glob
import os
from utils import *
from scipy.spatial.distance import pdist, squareform


class InstrumentsClassifier():
    def __init__(self, model_path,  corr_path='./runs/cor.csv', inst_path='./runs/inst.csv', sim_path='./runs/sim.csv', device='cpu', openmic_dir='./data/openmic-2018', vggish_path=PRE_TRAINED_PATHS['vggish']):
        self.model = TransformerClassifier(device=device).to(device)
        self.model.load_state_dict(model_path)
        self.model.eval()
        self.vggish = VGGish(vggish_path, preprocess=False,
                             postprocess=False).to(device)
        self.vggish.eval()
        self.corr_path = corr_path
        self.inst_path = inst_path
        self.sim_path = sim_path
        self.openmic_dir = openmic_dir

    def get_sim(self):
        inst_df = pd.read_csv(self.inst_path)
        sim_df = pd.DataFrame()
        for i, row in inst_df.iterrows():
            row = row.to_dict()

    def corr_sim(self, i_dct_0: dict[str, int], i_dct_1: dict[str, int]):
        corr_df = pd.read_csv(self.corr_path)
        for inst, i0, _, i1 in zip(i_dct_0.items(), i_dct_1.items()):
            pass

    def get_correlation(self, pick=500):
        inst_df = pd.read_csv(self.inst_path)
        corr_df = inst_df.T.sample(n=pick).astype(float).corr(method='pearson')
        corr_df.to_csv(self.corr_path)

    def get_inst(self):
        df = pd.DataFrame()
        audios = glob(os.path.join(
            self.openmic_dir, 'audio', '**', '*.ogg'), recursive=True)
        for audio in audios:
            row = pd.Series()
            res = self.get_inst_by_path(audio)
            for i, r in enumerate(res):
                row[IDX2INST_DICT[i]] = r
            df = pd.concat([df, row.to_frame().T], ignore_index=True)
        df.to_csv(self.inst_path)

    def get_inst_by_path(self, p):
        l = vggish_input.wavfile_to_examples(p)
        return self.get_inst_by_logmel(self, l)

    def get_inst_by_waveform(self, w):
        l = vggish_input.waveform_to_examples(w)
        return self.get_inst_by_logmel(self, l)

    def get_inst_by_logmel(self, l, threshold=0.65):
        t = self.vggish.forward(l)
        return torch.where(F.sigmoid(self.model.forward(t)) > threshold, 1.0, 0.0)


if __name__ == '__main__':
    ic = InstrumentsClassifier('./runs/model.pt')
