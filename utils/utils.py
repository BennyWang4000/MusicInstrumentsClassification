from .config import *
import wandb
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_accuracy_metrics(y_label, y_hat_label, threshold=THRESHOLD):
    y_hat_label = torch.where(
        y_hat_label > threshold, 1.0, 0.0)
    y_label = torch.where(
        y_label > 0, 1.0, 0.0)
    results = classification_report(
        y_label, y_hat_label, target_names=INST_NAME_LST, output_dict=True)
    return results


def save_confusion_matrix(epoch, state, y_label, y_hat_label, threshold=THRESHOLD):
    y_hat_label = torch.where(
        y_hat_label > threshold, 1.0, 0.0)
    y_label = torch.where(
        y_label > 0, 1.0, 0.0)
    f, axes = plt.subplots(4, 5, figsize=(25, 15))
    axes = axes.ravel()
    mcm = multilabel_confusion_matrix(
        y_label, y_hat_label)
    for i_a, cm in enumerate(mcm):
        axes[i_a].matshow(cm, cmap=plt.cm.get_cmap('cividis'))
        for (i, j), z in np.ndenumerate(cm):
            axes[i_a].text(j, i, '{:0.4f}'.format(
                z), ha='center', va='center')

    f.colorbar(plt.matshow([[0]], cmap=plt.cm.get_cmap('cividis')), ax=axes)
    f.savefig(os.path.join(MODEL_STATE_DIR,
                           'confusion', str(epoch) + '_' + state + '.jpg'))
    plt.close('all')


class WandbLogger():
    def __init__(self):
        wandb.init(
            project="MusicInstrumentsClassification",
            config={
                'name': MODEL_NAME,
                'epochs': EPOCHS,
                'is_valid': IS_VALID,
                'k_folds': K_FOLDS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
                'optimizer': OPTIMIZER,
                'criterion': CRITERION,
                'scheduler': SCHEDULER,
                'scheduler_step_size': SCHEDULER_STEP_SIZE,
                'scheduler_factory': SCHEDULER_FACTORY,
                'is_pre_trained_eval': IS_PRE_TRAINED_EVAL,
            }
        )

    def log(self, epoch, state, y_label, y_hat_label, avg_loss):
        results = compute_accuracy_metrics(y_label, y_hat_label)
        wandb.log(
            {str(epoch) + '_' + state + '_ppv': results['macro avg']['precision']})
        wandb.log(
            {str(epoch) + '_' + state + '_recall': results['macro avg']['recall']})
        wandb.log(
            {str(epoch) + '_' + state + '_f1': results['macro avg']['f1-score']})
        wandb.log(
            {str(epoch) + '_' + state + '_micro_ppv': results['micro avg']['precision']})
        wandb.log(
            {str(epoch) + '_' + state + '_weighted_f1': results['weighted avg']['f1-score']})
        # wandb.log(
        #     {str(epoch) + '_' + state + '_weighted_ppv': results['weighted avg']['precision']})
        # wandb.log(
        #     {str(epoch) + '_' + state + '_weighted_recall': results['weighted avg']['recall']})
        wandb.log(
            {str(epoch) + '_' + state + '_samples_f1': results['samples avg']['f1-score']})
        wandb.log(
            {str(epoch) + '_' + state + '_mean_loss': avg_loss})

    # def log(self, epoch, i, state, y_label, y_hat_label, loss):
    #     y_hat_label = torch.where(
    #         y_hat_label > THRESHOLD, 1.0, 0.0)
    #     y_label = torch.where(
    #         y_label > 0, 1.0, 0.0)

    #     acc = 0
    #     ppv = 0
    #     recall = 0
    #     f1 = 0

    #     for y, y_hat in zip(y_label, y_hat_label):
    #         tp, fp, tn, fn = self._perf_measure(
    #             y.tolist(), y_hat.tolist())
    #         p = 0 if (tp+fp) == 0 else (tp)/(tp+fp)
    #         r = 0 if (tp+fn) == 0 else (tp)/(tp+fn)
    #         acc += (tp+tn)/(tp+fp+tn+fn)
    #         ppv += p
    #         recall += r
    #         f1 += 0 if (p + r) == 0 else (2 * p * r) / (p + r)

    #     self.running_acc += acc
    #     self.running_ppv += ppv
    #     self.running_recall += recall
    #     self.running_f1 += f1
    #     self.running_loss += loss

    #     wandb.log(
    #         {str(epoch) + '_' + state + '_mean_loss': self.running_loss / (i + 1)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_mean_ppv': self.running_ppv / ((i + 1) * BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_mean_acc': self.running_acc / ((i + 1) * BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_mean_recall': self.running_recall / ((i + 1) * BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_mean_f1': self.running_f1 / ((i + 1) * BATCH_SIZE)})

    #     wandb.log(
    #         {str(epoch) + '_' + state + '_ppv': ppv / (BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_acc': acc / (BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_recall': recall / (BATCH_SIZE)})
    #     wandb.log(
    #         {str(epoch) + '_' + state + '_f1': f1 / (BATCH_SIZE)})

    # def _perf_measure(self, y_actual, y_hat):
    #     TP = 0
    #     FP = 0
    #     TN = 0
    #     FN = 0

    #     for i in range(len(y_hat)):
    #         if y_actual[i] == y_hat[i] == 1:
    #             TP += 1
    #         if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
    #             FP += 1
    #         if y_actual[i] == y_hat[i] == 0:
    #             TN += 1
    #         if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
    #             FN += 1

    #     return (TP, FP, TN, FN)
