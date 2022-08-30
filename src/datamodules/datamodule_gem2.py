from typing import Union
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import torch
from torch_geometric import transforms as T


class ToOneHot(T.BaseTransform):
    def __init__(self):
        super().__init__()
        
    def _to_one_hot(self, idx, n_max):
        vec = torch.zeros(n_max, dtype=torch.float32)
        vec[int(idx)] = 1.

        return vec

    def _atom2onehot(self, atom):
        atom_type = self._to_one_hot(atom[0], 119)
        aromaticity = self._to_one_hot(atom[6], 2)
        formal_charge = self._to_one_hot(atom[3], 16)
        chirality_tag = self._to_one_hot(atom[1], 4)
        degree = self._to_one_hot(atom[2], 11)
        n_H = self._to_one_hot(atom[4], 9)
        hybridization = self._to_one_hot(atom[5], 6)
        in_ring = self._to_one_hot(atom[7], 2)

        one_hot_vec = torch.cat([atom_type, aromaticity, formal_charge, 
                                 chirality_tag, degree, n_H, hybridization, in_ring])

        return one_hot_vec

    def _bond2onehot(self, bond):
        bond_type = self._to_one_hot(bond[0], 4)
        stereo = self._to_one_hot(bond[1], 6)
        conjugated = self._to_one_hot(bond[2], 2)
        in_ring = self._to_one_hot(bond[3], 2)
        bond_dir = self._to_one_hot(bond[4], 7)

        one_hot_vec = torch.cat([bond_type, stereo, conjugated, in_ring, bond_dir])

        return one_hot_vec
    
    def __call__(self, data):
        x = torch.stack([self._atom2onehot(atom) for atom in data.x], dim=0)
        data.x = x
        
        edge_attr = torch.stack([self._bond2onehot(bond) for bond in data.edge_attr], dim=0)
        data.edge_attr = edge_attr
        
        return data


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold: Union[int, str] = "full",
        num_workers: int = 4,
        batch_size: int = 32,
        virtual_node: bool = False,
        fingerprints: bool = False,
        removeHs: bool = False,
        onehot: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        from src.datamodules.components.dataset_gem2 import TrainDataset, TestDataset
            
        transform = []
        if virtual_node:
            transform.append(T.VirtualNode())
        
        if onehot:
            transform.append(ToOneHot())
            
        transform = T.Compose(transform)
        
        self.full_data = TrainDataset(transform=transform, removeHs=removeHs)
        self.test_data = TestDataset(transform=transform, removeHs=removeHs)
        
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            if self.hparams.fold != "full":
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(self.full_data)):
                    if i == self.hparams.fold:
                        break
                self.train_data = self.full_data.copy(train_idx)
                self.val_data = self.full_data.copy(val_idx)
                
            else:
                kfold = KFold(n_splits=5, shuffle=True, random_state=123456789)
                for i, (train_idx, val_idx) in enumerate(kfold.split(self.full_data)):
                    if i == 0:
                        break
                self.train_data = self.full_data
                self.val_data = self.full_data.copy(val_idx)  # Dummy val data
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            follow_batch=["edge_attr"]
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            follow_batch=["edge_attr"]
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            follow_batch=["edge_attr"]
        )
