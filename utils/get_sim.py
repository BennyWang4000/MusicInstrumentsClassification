'''
- get 5000 instruments
- random pick 500
- correlation
- calculate by matrix
'''
from tqdm import tqdm
from model import TransformerClassifier
from torchvggish.torchvggish.vggish import VGGish, vggish_input
import torch.nn.functional as F
import torch
import pandas as pd
from glob import glob
import os
from utils import *
from scipy.spatial.distance import pdist, squareform
import numpy as np


class InstrumentsClassifier():
    def __init__(self, model_path,
                 corr_path='./runs/corr.csv',
                 inst_path='./runs/inst.csv',
                 simi_path='./runs/simi.csv',
                 spo_dir='./data/spotify_preview_target',
                 spo_genre=['blues', 'rap'],
                 vggish_path=PRE_TRAINED_PATHS['vggish']):
        self.device = torch.device(
            'cuda:0' if cuda.is_available() else 'cpu')
        self.model = TransformerClassifier(device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.train()
        self.vggish = VGGish(vggish_path, preprocess=False,
                             postprocess=False).to(self.device)
        self.vggish.eval()
        self.corr_path = corr_path
        self.inst_path = inst_path
        self.simi_path = simi_path
        self.spo_dir = spo_dir
        self.spo_genre = spo_genre

    def to_simi(self):
        simi_dct = {}
        simi_df = pd.DataFrame()
        inst_df = pd.read_csv(self.inst_path, index_col=[0, 1])
        corr_df = pd.read_csv(self.corr_path, index_col=[0])
        for i, (aud_id, insts) in tqdm(enumerate(inst_df.iterrows()), total=len(inst_df.index)):
            for t_aud_id, t_insts in inst_df.iloc[i:].iterrows():
                simi = self.corr_sim(insts, t_insts, corr_df)
                simi_dct[(aud_id[1], t_aud_id[1])] = simi
        simi_df = pd.Series(simi_dct).unstack()
        simi_df = simi_df.combine_first(simi_df.T).fillna(1.0)
        simi_df.astype('float32')
        simi_df.to_csv(self.simi_path)

    def corr_sim(self, inst_dct: dict[str, int], t_inst_dct: dict[str, int], corr_df):
        simi = 0
        inst_dct = {k: v for k, v in inst_dct.items() if v > 0}
        t_inst_dct = {k: v for k, v in t_inst_dct.items() if v > 0}
        for inst, is_inst in inst_dct.items():
            for t_inst, t_is_inst in {k: v for k, v in t_inst_dct.items() if v > 0}.items():
                simi += corr_df.xs(inst)[t_inst]
        return 0 if len(inst_dct) * len(t_inst_dct) == 0 else simi / (2 * len(inst_dct) * len(t_inst_dct))

    def to_corr(self, pick=500):
        inst_df = pd.read_csv(self.inst_path, index_col=[0, 1])
        corr_df = inst_df.sample(n=pick).astype(
            'float32').corr(method='pearson')
        corr_df += 1
        corr_df = corr_df.apply(
            lambda x: 1.0 / (np.power((np.exp(-5 * x) + 1), 500)))
        corr_df.astype('float32')
        corr_df.to_csv(self.corr_path)

    def to_inst(self):
        df = pd.DataFrame()
        for genre in self.spo_genre:
            genre_df = pd.read_csv(os.path.join(
                self.spo_dir, genre + '_non_duplicated.csv'))
            for n_row, row in tqdm(genre_df.iterrows(), total=len(genre_df.index)):
                is_find = False
                for audio_path in glob(os.path.join(
                        self.spo_dir, '**', '*.mp3'), recursive=True):
                    audio_name = audio_path.split('/')[-1].split('-')[-1]
                    if audio_name == row['spotify_id'] + '.mp3':
                        is_find = True
                        f_row = pd.Series()
                        f_row['spotify_id'] = row['spotify_id']
                        res = self.get_inst_by_path(audio_path)
                        for i, r in enumerate(res):
                            f_row[IDX2INST_DICT[i]] = r
                        df = pd.concat([df, f_row.to_frame().T],
                                       ignore_index=True)
                        break
                if not is_find:
                    print(row['spotify_id'], '\t', row['name'], )
        df.to_csv(self.inst_path)

    def get_inst_by_path(self, p):
        l = vggish_input.wavfile_to_examples(p)
        return self.get_inst_by_logmel(l)

    def get_inst_by_waveform(self, w):
        l = vggish_input.waveform_to_examples(w)
        return self.get_inst_by_logmel(l)

    def get_inst_by_logmel(self, l, threshold=0.5):
        res = torch.zeros(20)
        for i in range(4):
            t = l[i * 5:i * 5 + 10, :, :, :].to(self.device)
            t = self.vggish(t)
            t = self.model(t.unsqueeze(0)).squeeze(0).cpu()
            res += t
        res = F.sigmoid(res)
        return torch.where(res > threshold, 1, 0).tolist()

    def normalization(self):
        df = pd.read_csv(self.simi_path)


if __name__ == '__main__':
    state_path = ''
    ic = InstrumentsClassifier(state_path)
    # ic.to_inst()
    # ic.to_corr()
    # ic.to_simi()
