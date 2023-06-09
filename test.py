import os
import yaml
import argparse
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import DataLoader

from src.utils import seed_everything
from src.dataset import GEM1Dataset
from src.model import GEM1Model


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

    seed_everything(cfg.seed)
    config_name = args.config.split('/')[-1].split('.')[0]
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    test_dataset = GEM1Dataset(root=cfg.dataset.root, mode='test')

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False,
        num_workers=cfg.dataset.num_workers, pin_memory=cfg.dataset.pin_memory,
        follow_batch=['edge_attr']
    )

    model = GEM1Model(**cfg.model).to(device)
    ckpt_path = args.ckpt_path if args.ckpt_path else f'outputs/{config_name}/ckpts/best.pt'
    model.load_state_dict(torch.load(ckpt_path))

    test_preds = test(model, test_loader, device)
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df[sub_df.columns[1:]] = test_preds
    sub_df.to_csv(f'outputs/{config_name}/submissions/submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default='')

    args = parser.parse_args()
    main(args)