from .config import *
import wandb


class WandbLogger():
    def __init__(self):
        self.running_loss = 0
        self.running_ppv = 0
        self.running_acc = 0
        if IS_WANDB:
            wandb.init(
                project="MusicInstrumentsClassification",
                config={
                    'name': MODEL_NAME,
                    'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE,
                    'lr': LR,
                    'optimizer': OPTIMIZER,
                    'criterion': CRITERION,
                }
            )

    def log_init(self):
        self.running_loss = 0
        self.running_ppv = 0
        self.running_acc = 0

    def log(self, epoch, i, state, y_label, y_hat_label, loss):
        y_hat_label = torch.where(
            y_hat_label > 0.5, 1.0, 0.0)
        y_label = torch.where(
            y_label > 0, 1.0, 0.0)

        for y, y_hat in zip(y_label, y_hat_label):
            tp, fp, tn, fn = self._perf_measure(
                y.tolist(), y_hat.tolist())
            self.running_ppv += 0 if (tp+fp) == 0 else tp/(tp+fp)
            self.running_acc += (tp+tn)/(tp+fp+tn+fn)

        self.running_loss += loss

        wandb.log(
            {str(epoch) + '_' + state + '_mean_loss': self.running_loss / (i + 1)})
        wandb.log(
            {str(epoch) + '_' + state + '_mean_ppv': self.running_ppv / ((i + 1) * BATCH_SIZE)})
        wandb.log(
            {str(epoch) + '_' + state + '_mean_acc': self.running_acc / ((i + 1) * BATCH_SIZE)})

    def _perf_measure(self, y_actual, y_hat):
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
