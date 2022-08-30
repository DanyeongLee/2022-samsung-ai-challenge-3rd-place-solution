from typing import Union
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fold: Union[int, str] = "full",
        num_workers: int = 4,
        batch_size: int = 32,
        virtual_node: bool = False,
        fingerprints: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        if fingerprints:
            from src.datamodules.components.dataset2d_fp import TrainDataset, TestDataset
        else:
            from src.datamodules.components.dataset2d import TrainDataset, TestDataset
            
        transform = []
        if virtual_node:
            transform.append(T.VirtualNode())
        transform = T.Compose(transform)
        
        self.full_data = TrainDataset(transform=transform)
        self.test_data = TestDataset(transform=transform)
        
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
