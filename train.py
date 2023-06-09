import os
import yaml
import argparse
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

import wandb

from src.utils import seed_everything
from src.dataset import GEM1Dataset
from src.model import GEM1Model


def train(model, loader, optimizer, scheduler, criterion, device, grad_clip=None):
    model.train()
    epoch_loss = 0
    epoch_rmse = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item() * batch.num_graphs
        epoch_rmse += F.mse_loss(output, batch.y, reduction='sum').sqrt().item()
    
    return epoch_loss / len(loader.dataset), epoch_rmse / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_rmse = 0
    
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch.y)
        epoch_loss += loss.item() * batch.num_graphs
        epoch_rmse += F.mse_loss(output, batch.y, reduction='sum').sqrt().item()
    
    return epoch_loss / len(loader.dataset), epoch_rmse / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    preds = []
    for batch in tqdm(loader, desc='Predicting'):
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.detach().cpu())
    
    return np.concatenate(preds)


def main(args):
    with open(args.config) as f:
        cfg = EasyDict(yaml.safe_load(f))

    seed_everything(args.seed)
    config_name = args.config.split('/')[-1].split('.')[0]
    os.makedirs(f'outputs/{config_name}/fold{args.fold}/seed{args.seed}', exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    wandb.init(project=cfg.logger.project, name=config_name, config=cfg)

    full_dataset = GEM1Dataset(root=cfg.dataset.root, mode='train')

    kfold = KFold(n_splits=10, shuffle=True, random_state=123456789)
    for i, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        if i == args.fold:
            break
    train_dataset = full_dataset.copy(train_idx)
    val_dataset = full_dataset.copy(val_idx)
    test_dataset = GEM1Dataset(root=cfg.dataset.root, mode='test')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, 
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
        follow_batch=['edge_attr']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
        follow_batch=['edge_attr']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
        follow_batch=['edge_attr']
    )

    model = GEM1Model(**cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.n_epochs * len(train_loader))
    criterion = torch.nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0
    for epoch in tqdm(range(cfg.train.n_epochs)):
        train_loss, train_rmse = train(model, train_loader, optimizer, scheduler, criterion, device, cfg.train.grad_clip)
        val_loss, val_rmse = validate(model, val_loader, criterion, device)
        wandb.log({
            'train/loss': train_loss,
            'train/rmse': train_rmse,
            'val/loss': val_loss,
            'val/rmse': val_rmse
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), f'outputs/{config_name}/fold{args.fold}/seed{args.seed}/best.ckpt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == cfg.train.patience:
                break
        
    model.load_state_dict(torch.load(f'outputs/{config_name}/fold{args.fold}/seed{args.seed}/best.ckpt'))

    val_preds = test(model, val_loader, device)
    train_df = pd.read_csv('data/train_set.ReorgE.csv')
    val_df = train_df.iloc[val_idx]
    val_df['Reorg_g_pred'] = val_preds[:, 0]
    val_df['Reorg_h_pred'] = val_preds[:, 1]
    val_df.to_csv(f'outputs/{config_name}/fold{args.fold}/seed{args.seed}/val_preds.csv', index=False)

    test_preds = test(model, test_loader, device)
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df[sub_df.columns[1:]] = test_preds
    sub_df.to_csv(f'outputs/{config_name}/fold{args.fold}/seed{args.seed}/test_preds.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    main(args)