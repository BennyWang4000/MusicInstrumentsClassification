# %%
import wandb
from torch.utils.data import DataLoader
from math import floor
from model import CNN2D, InstrumentsDataset
import torch
import pandas as pd
from tqdm import tqdm
import os
from config import *


def train(dataloader, model):
    for epoch in range(EPOCHS):
        print('epoch:\t', epoch)
        running_loss = 0.0
        progress = tqdm(dataloader)
        iterator = enumerate(progress)
        for _, batch in iterator:
            log_mel, y_label = batch
            OPTIMIZER.zero_grad()
            y_hat_label = model(log_mel)
            loss = CRITERION(y_label, y_hat_label)
            loss.backward()
            OPTIMIZER.step()

            running_loss += loss.item()
            progress.set_postfix({
                'mean loss': running_loss / len(dataloader)})

    model.normalize_parameters()
    torch.save(model.state_dict(), os.path.join(
        MODEL_STATE_DIR, model_name + '.pt'))


def evaulate(model_name, data_path, model_type):
    '''evaluate model performance by hits@1, hits@3, hits@10 or mrr


    '''
    kg_train = KnowledgeGraph(
        df=pd.read_csv(data_path))
    kg_df = kg_train.get_df()

    if model_type == 'TransE':
        model = M.TransEModel(emb_dim=EMB_DIM, n_entities=len(set(kg_df['to'].unique()) | set(kg_df['from'].unique())),
                              n_relations=len(set(kg_df['rel'].unique())), rel2idx=kg_train.rel2ix, ent2idx=kg_train.ent2ix).to(DEVICE)
    elif model_type == 'TransH':
        model = M.TransHModel(emb_dim=EMB_DIM, n_entities=len(set(kg_df['to'].unique()) | set(kg_df['from'].unique())),
                              n_relations=len(set(kg_df['rel'].unique())), rel2idx=kg_train.rel2ix, ent2idx=kg_train.ent2ix).to(DEVICE)
    elif model_type == 'TransR':
        model = M.TransRModel(ent_emb_dim=EMB_DIM, rel_emb_dim=EMB_DIM, n_entities=len(set(kg_df['to'].unique()) | set(kg_df['from'].unique())),
                              n_relations=len(set(kg_df['rel'].unique())), rel2idx=kg_train.rel2ix, ent2idx=kg_train.ent2ix).to(DEVICE)
    elif model_type == 'TransD':
        model = M.TransDModel(ent_emb_dim=EMB_DIM, rel_emb_dim=EMB_DIM, n_entities=len(set(kg_df['to'].unique()) | set(kg_df['from'].unique())),
                              n_relations=len(set(kg_df['rel'].unique())), rel2idx=kg_train.rel2ix, ent2idx=kg_train.ent2ix).to(DEVICE)
    else:
        raise ValueError('model type error')

    model.load_state_dict(torch.load(
        os.path.join(MODEL_STATE_DIR, model_name + '.pt')))
    eval = LinkPredictionEvaluator(model, kg_train)
    eval.evaluate(4)
    eval.print_results(k=[1, 3, 10])


# %%
if __name__ == '__main__':
    dataset = InstrumentsDataset(OPENMIC_DIR, CLASS2IDX_DICT)
    val_num = floor(dataset.__len__() * VALID_PER)
    test_num = floor(dataset.__len__() * TEST_PER)
    train_num = dataset.__len__() - test_num - val_num

    print('   device:\t', DEVICE)
    print('train_num:\t', train_num)
    print(' test_num:\t', test_num)
    print('  val_num:\t', val_num)

    train_set, test_set, val_set = torch.utils.data.random_split(
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

# %%
