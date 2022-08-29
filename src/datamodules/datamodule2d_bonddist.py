from typing import Union
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dense_to_sparse



class DistanceEdge(T.BaseTransform):
    def __call__(self, data):
        pd_g = torch.tensor(euclidean_distances(data.pos_g), dtype=torch.float)
        pd_g.fill_diagonal_(1)
        pd_g = 1 / pd_g
        data.full_edge_index, data.distance_edge_attr_g = dense_to_sparse(pd_g)

        pd_ex = torch.tensor(euclidean_distances(data.pos_ex), dtype=torch.float)
        pd_ex.fill_diagonal_(1)
        pd_ex = 1 / pd_ex
        _, data.distance_edge_attr_ex = dense_to_sparse(pd_ex)
        
        return data


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
        transform.append(DistanceEdge())
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
                self.train_data = self.full_data
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        if type(self.hparams.fold) != str:
            return DataLoader(
                dataset=self.val_data, 
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
        else:
            return None
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
