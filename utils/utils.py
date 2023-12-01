from .config import *
import wandb


def wandb_init():
    if IS_WANDB:
        wandb.init(
            project="MusicInstrumentsClassification",
            config={
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
                'optimizer': OPTIMIZER,
                'criterion': CRITERION,
            }
        )


def wandb_log(epoch, i, state, y_label, y_hat_label, loss, acc, ppv):
    y_hat_label = torch.where(
        y_hat_label > 0.5, 1.0, 0.0)
    for y, y_hat in zip(y_label, y_hat_label):
        tp, fp, tn, fn = perf_measure(
            y.tolist(), y_hat.tolist())
        ppv += 0 if (tp+fp) == 0 else tp/(tp+fp)
        acc += tp+tn/(tp+fp+tn+fn)

    wandb.log(
        {str(epoch) + '_' + state + '_mean_loss': loss / (i + 1)})
    wandb.log(
        {str(epoch) + '_' + state + '_mean_acc': acc / ((i + 1) * BATCH_SIZE)})
    wandb.log(
        {str(epoch) + '_' + state + '_mean_ppv': ppv / ((i + 1) * BATCH_SIZE)})

    return ppv, acc


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)
