# %%
from utils import *
from torch.utils.data import DataLoader, random_split
from math import floor
from model import InstrumentsDataset, TransformerClassifier
import torch
from tqdm import tqdm
import os
from sklearn.model_selection import KFold
# %%


if __name__ == '__main__':
    dataset = InstrumentsDataset(
        openmic_dir=OPENMIC_DIR,
        inst2idx_dict=INST2IDX_DICT,
        classes=len(INST2IDX_DICT),
        device=DEVICE,
        is_pre_trained=PRE_TRAINED,
        signal_args=KWARGS_SIGNAL,
        pre_trained_path=PRE_TRAINED_PATHS[PRE_TRAINED])
    test_num = floor(dataset.__len__() * TEST_PER)
    train_num = dataset.__len__() - test_num
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    print('     name:\t', MODEL_NAME)
    print('   device:\t', DEVICE)
    print('train_num:\t', train_num)
    print(' test_num:\t', test_num)

    train_set, test_set = random_split(
        dataset, [train_num, test_num])

    train_loader = DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerClassifier(classes=len(
        INST2IDX_DICT), device=DEVICE).to(DEVICE)
    optimizer = OPTIMIZER(model.parameters(), lr=LR, weight_decay=0.00001)
    scheduler = SCHEDULER(optimizer, mode=SCHEDULER_MODE,
                          patience=SCHEDULER_STEP_SIZE, factor=SCHEDULER_FACTORY)
    running_loss = 0
    sigmoid = torch.nn.Sigmoid()

    if IS_WANDB:
        wandb_logger = WandbLogger()

    for epoch in range(EPOCHS):
        if epoch == 0:
            print(model)

        # * ---------------------------------------------------------------------------- #
        # *                                   training                                   #
        # * ---------------------------------------------------------------------------- #
        print('epoch:\t', epoch)
        running_loss = 0
        model.train()
        progress = tqdm(train_loader)
        iterator = enumerate(progress)
        for i, batch in iterator:
            sample_input, y_label,  = batch
            optimizer.zero_grad()
            y_hat_label = model(sample_input)

            # break
            loss = CRITERION(y_hat_label, y_label)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            y_hat_label = sigmoid(y_hat_label)
            if IS_WANDB:
                running_loss += loss.item()
                wandb_logger.log(epoch, 'train', y_label.cpu(),
                                 y_hat_label.cpu(), running_loss/(i + 1))

        # break
        # # * ---------------------------------------------------------------------------- #
        # # *                                  validation                                  #
        # # * ---------------------------------------------------------------------------- #
        # running_loss = 0.0
        # running_tp = 0.0
        # running_fp = 0.0
        # model.eval()
        # progress = tqdm(valid_loader)
        # iterator = enumerate(progress)
        # for i, batch in iterator:
        #     sample_input, y_label, log_mel, mfcc = batch
        #     y_hat_label = model(sample_input)
        #     loss = CRITERION(y_hat_label, y_label)
        #     # y_hat_label = sigmoid(y_hat_label)

        #     wandb_logger.log(epoch, i, 'val', y_label, y_hat_label,
        #                      running_loss, running_acc, running_ppv)

        # * ---------------------------------------------------------------------------- #
        # *                                    testing                                   #
        # * ---------------------------------------------------------------------------- #

        running_loss = 0
        model.eval()
        progress = tqdm(test_loader)
        iterator = enumerate(progress)
        for i, batch in iterator:
            sample_input, y_label = batch
            y_hat_label = model(sample_input)
            loss = CRITERION(y_hat_label, y_label)
            y_hat_label = sigmoid(y_hat_label)

            if IS_WANDB:
                running_loss += loss.item()
                wandb_logger.log(epoch, 'test', y_label.cpu(),
                                 y_hat_label.cpu(), running_loss/(i + 1))

        # model.normalize_parameters()
        torch.save(model.state_dict(), os.path.join(
            MODEL_STATE_DIR, MODEL_NAME + '_' + str(epoch) + '.pt'))

# %%
