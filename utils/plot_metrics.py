# %%
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
metrics_df = pd.read_csv(
    '/home/wirl/wang/MusicInstrumentsClassification/runs/omg_help_7_metr.csv', index_col=[0])

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(5, 4))
    bp1 = ax.boxplot(metrics_df['samples avg_precision'], positions=[1], widths=0.25, sym='', whis=1,
                     patch_artist=True, boxprops=dict(facecolor="C0"), labels=['avg precision'])
    bp2 = ax.boxplot(metrics_df['samples avg_recall'], positions=[2], widths=0.25, sym='', whis=1,
                     patch_artist=True, boxprops=dict(facecolor="C3"), labels=['avg recall'])
    bp2 = ax.boxplot(metrics_df['samples avg_f1-score'], positions=[3], widths=0.25, sym='', whis=1,
                     patch_artist=True, boxprops=dict(facecolor="C2"), labels=['avg f1-score'])
    ax.set_ylim(0.2, 1.1)
    plt.show()

# %%
