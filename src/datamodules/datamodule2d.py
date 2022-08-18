from typing import Union
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        fold: Union[int, str] = "full",
        num_workers: int = 4,
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_data", "test_data"], logger=False)
        
        self.full_data = train_data
        self.test_data = test_data
        
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
