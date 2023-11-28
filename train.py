# %%
from config import *
import wandb
from torch.utils.data import DataLoader, random_split
from math import floor
from model import CNN2D, CNN1D, InstrumentsDataset, LSTMModel
import torch
from tqdm import tqdm
import os
# %%


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


def get_true_positive(y, y_hat):
    y_hat_true = torch.where(y_hat > 0.5, 1.0, -1.0)
    true_positive = (y == y_hat_true).sum().float()
    return true_positive / torch.count_nonzero(y)


def get_false_positive(y, y_hat):
    y_hat_true = torch.where(
        y_hat > 0.5, 1.0, -1.0)
    false_positive = torch.where(
        y != y_hat_true, 1.0, 0).sum().float()
    return false_positive / torch.count_nonzero(y_hat_true)


if __name__ == '__main__':
    dataset = InstrumentsDataset(
        openmic_dir=OPENMIC_DIR, class2idx_dict=CLASS2IDX_DICT, device=DEVICE, audio_length=AUDIO_LENGTH,
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, n_freqs=N_FREQS,
        hop_length=HOP_LENGTH, f_max=F_MAX, f_min=F_MIN)
    val_num = floor(dataset.__len__() * VALID_PER)
    test_num = floor(dataset.__len__() * TEST_PER)
    train_num = dataset.__len__() - test_num - val_num

    print('     name:\t', MODEL_NAME)
    print('   device:\t', DEVICE)
    print('train_num:\t', train_num)
    print(' test_num:\t', test_num)
    print('  val_num:\t', val_num)

    train_set, test_set, val_set = random_split(
        dataset, [train_num, test_num, val_num])

    train_loader = DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

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

    model = CNN2D(len(CLASS2IDX_DICT)).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    sigmoid = torch.nn.Sigmoid()

    for epoch in range(EPOCHS):
        print('epoch:\t', epoch)
        running_loss = 0.0
        running_tp = 0.0
        running_fp = 0.0
        model.train()
        progress = tqdm(train_loader)
        iterator = enumerate(progress)
        for i, batch in iterator:
            log_mel, y_label, sr = batch
            y_hat_label = model(log_mel)
            loss = CRITERION(y_hat_label, y_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y_hat_label = sigmoid(y_hat_label)
            y_hat_label = torch.where(
                y_hat_label > 0.5, 1.0, 0.0)
            for y, y_hat in zip(y_label, y_hat_label):
                tp, fp, _, _ = perf_measure(
                    y.tolist(), y_hat.tolist())
                running_tp += tp/(tp+fp + 1)
                running_fp += fp/(tp+fp + 1)
            wandb.log(
                {str(epoch) + '_train_mean_loss': running_loss / (i + 1)})
            wandb.log(
                {str(epoch) + '_train_mean_tp': running_tp / ((i + 1) * BATCH_SIZE)})
            wandb.log(
                {str(epoch) + '_train_mean_fp': running_fp / ((i + 1) * BATCH_SIZE)})

        running_loss = 0.0
        running_tp = 0.0
        running_fp = 0.0
        model.eval()
        progress = tqdm(test_loader)
        iterator = enumerate(progress)
        for i, batch in iterator:
            log_mel, y_label, sample_rate = batch
            y_hat_label = model(log_mel)
            loss = CRITERION(y_hat_label, y_label)
            y_hat_label = sigmoid(y_hat_label)
            running_loss += loss.item()
            for y, y_hat in zip(y_label, y_hat_label):
                tp, fp, _, _ = perf_measure(
                    y.tolist(), y_hat.tolist())
                running_tp += tp/(tp+fp + 1)
                running_fp += fp/(tp+fp + 1)
            wandb.log(
                {str(epoch) + '_test_mean_loss': running_loss / (i + 1)})
            wandb.log(
                {str(epoch) + '_test_mean_tp': running_tp / ((i + 1) * BATCH_SIZE)})
            wandb.log(
                {str(epoch) + '_test_mean_fp': running_fp / ((i + 1) * BATCH_SIZE)})

    running_loss = 0.0
    running_tp = 0.0
    running_fp = 0.0
    model.eval()
    progress = tqdm(test_loader)
    iterator = enumerate(progress)
    for i, batch in iterator:
        log_mel, y_label = batch
        y_hat_label = model(log_mel)
        loss = CRITERION(y_hat_label, y_label)
        y_hat_label = sigmoid(y_hat_label)
        running_loss += loss.item()
        for y, y_hat in zip(y_label, y_hat_label):
            tp, fp, _, _ = perf_measure(
                y.tolist(), y_hat.tolist())
            running_tp += tp/(tp+fp + 1)
            running_fp += fp/(tp+fp + 1)
        wandb.log(
            {'val_mean_loss': running_loss / (i + 1)})
        wandb.log(
            {str(epoch) + '_val_mean_tp': running_tp / ((i + 1) * BATCH_SIZE)})
        wandb.log(
            {str(epoch) + '_val_mean_fp': running_fp / ((i + 1) * BATCH_SIZE)})

    model.normalize_parameters()
    torch.save(model.state_dict(), os.path.join(
        MODEL_STATE_DIR, MODEL_NAME + '.pt'))

# %%
