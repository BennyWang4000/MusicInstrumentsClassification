import numpy as np
from model import TransformerClassifier
from torchvggish.torchvggish.vggish import VGGish, vggish_input
import torch.nn.functional as F
import pandas as pd

class InstrumentsClassifier():
    def __init__(self, model_path, vggish_path, device='cpu'):
        self.model = TransformerClassifier(device=device).to(device)
        self.model.load_state_dict(model_path)
        self.model.eval()

        self.vggish = VGGish(vggish_path, preprocess=False,
                             postprocess=False).to(device)
        self.vggish.eval()
        self.correlation_df= pd.DataFrame()

    def get_sim_by_path(self, p0, p1):
        l0, l1 = vggish_input.wavfile_to_examples(p0, p1)
        return self.get_sim_by_logmel(self, l0, l1)

    def get_sim_by_waveform(self, w0, w1):
        l0, l1 = vggish_input.waveform_to_examples(w0, w1)
        return self.get_sim_by_logmel(self, l0, l1)

    def get_sim_by_logmel(self, l0, l1):
        t0, t1 = self.vggish.forward(l0), self.vggish.forward(l1)
        r0, r1 = F.sigmoid(self.model.forward(
            t0)), F.sigmoid(self.model.forward(t1))

    def 