# %%
from utils import *
import wandb
from torch.utils.data import DataLoader, random_split
from math import floor
from model import CNN2D, CNN1D, InstrumentsDataset, LSTMModel
import torch
from tqdm import tqdm
import os
# %%


if __name__ == '__main__':
    dataset = InstrumentsDataset(
        openmic_dir=OPENMIC_DIR, inst2idx_dict=INST2CLASSIDX_DICT, classes=len(INSTIDX2CLASSIDX_DICT), device=DEVICE, audio_length=AUDIO_LENGTH,
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

    model = CNN2D(len(INST2CLASSIDX_DICT)).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    sigmoid = torch.nn.Sigmoid()

    wandb_init()

    # * ============================ train ============================
    for epoch in range(EPOCHS):
        print('epoch:\t', epoch)
        running_loss = 0.0
        running_acc = 0.0
        running_ppv = 0.0
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
            # y_hat_label = sigmoid(y_hat_label)
            if i % 500 == 0:
                print(y_hat_label)

            running_acc, running_ppv = wandb_log(epoch, i, 'train', y_label, y_hat_label,
                                                 running_loss, running_acc, running_ppv)

        # * ============================ test ============================
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
            # y_hat_label = sigmoid(y_hat_label)
            running_loss += loss.item()

            running_acc, running_ppv = wandb_log(epoch, i, 'test', y_label, y_hat_label,
                                                 running_loss, running_acc, running_ppv)

    # * ============================ val ============================
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
        running_loss += loss.item()
        # y_hat_label = sigmoid(y_hat_label)
        running_acc, running_ppv = wandb_log(epoch, i, 'val', y_label, y_hat_label,
                                             running_loss, running_acc, running_ppv)

    model.normalize_parameters()
    torch.save(model.state_dict(), os.path.join(
        MODEL_STATE_DIR, MODEL_NAME + '.pt'))

# %%
