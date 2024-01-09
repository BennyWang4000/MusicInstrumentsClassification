
import pandas as pd
from utils import *
from torch.utils.data import DataLoader, random_split
from math import floor
from model import InstrumentsDataset, TransformerClassifier
import torch
from tqdm import tqdm
import os
import plotly.graph_objects as go

MODEL_STATE_NAME = ''


if __name__ == '__main__':
    dataset = InstrumentsDataset(
        openmic_dir=OPENMIC_DIR,
        inst2idx_dict=INST2IDX_DICT,
        classes=len(INST2IDX_DICT),
        device=DEVICE,
        is_pre_trained=PRE_TRAINED,
        signal_args=KWARGS_SIGNAL,
        pre_trained_path=PRE_TRAINED_PATHS[PRE_TRAINED])

    print('     name:\t', MODEL_NAME)
    print('   device:\t', DEVICE)

    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerClassifier(classes=len(
        INST2IDX_DICT), device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_STATE_DIR, MODEL_STATE_NAME)))

    running_loss = 0
    sigmoid = torch.nn.Sigmoid()

    metr_df = pd.DataFrame()
    inst_df = pd.DataFrame()
    model.eval()
    progress = tqdm(data_loader)
    iterator = enumerate(progress)
    for j, batch in iterator:
        sample_input, y_label,  = batch
        y_hat_label = model(sample_input)
        y_hat_label = sigmoid(y_hat_label)

        results = compute_accuracy_metrics(
            y_label.cpu(), y_hat_label.cpu(), THRESHOLD)
        row = pd.Series()
        for cat, metrics in results.items():
            if cat in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
                for metric, value in metrics.items():
                    row[cat + '_' + metric] = value
        metr_df = pd.concat(
            [metr_df, row.to_frame().T], ignore_index=True)

    inst_df.to_csv(os.path.join(MODEL_STATE_DIR, 'inst.csv'))
    metr_df.to_csv(os.path.join(MODEL_STATE_DIR, 'metr.csv'))
