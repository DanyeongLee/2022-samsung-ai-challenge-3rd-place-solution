import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MinMetric
from scipy.spatial.transform import Rotation as R


class BaseNet(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 30,
        n_rotate: int = 10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.MSELoss()
        
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        
        self.val_rmse_best = MinMetric()
        
    def random_rotate(self, pos):
        random_theta = (np.random.rand(3) - 0.5) * np.pi * 2
        r = R.from_euler("xyz", random_theta)
        r = r.as_matrix().T
        r = torch.tensor(r, dtype=torch.float).to(pos.device)
    
        return pos.mm(r)

    def random_rotate_batch(self, batch):
        batch.pos_g = self.random_rotate(batch.pos_g)
        batch.pos_ex = self.random_rotate(batch.pos_ex)
        
    def forward(self, batch):
        return self.net(batch)
    
    def on_train_start(self):
        self.val_rmse_best.reset()
        
    def step(self, batch):
        pred = self(batch)
        loss = self.criterion(pred, batch.y)
        
        return loss, pred, batch.y
    
    def training_step(self, batch, batch_idx):
        losses = 0.
        
        for _ in range(self.hparams.n_rotate):
            self.random_rotate_batch(batch)
            loss, pred, target = self.step(batch)
            losses += loss
            
        losses /= self.hparams.n_rotate
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        preds = []
        
        for _ in range(self.hparams.n_rotate):
            self.random_rotate_batch(batch)
            loss, pred, targets = self.step(batch)
            preds.append(pred.unsqueeze(-1))
        
        preds = torch.cat(preds, axis=-1).mean(-1)
        
        self.val_rmse.update(preds, targets)
        
    def validation_epoch_end(self, outputs):
        # get val metric from current epoch
        epoch_rmse = self.val_rmse.compute()
        
        # log epoch metrics
        metrics = {"val/rmse": epoch_rmse}
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # log best metric
        self.val_rmse_best.update(epoch_rmse)
        self.log("val/rmse_best", self.val_rmse_best.compute(), on_epoch=True, prog_bar=True)

        # reset val metrics
        self.val_rmse.reset()
    
    def predict_step(self, batch, batch_idx):
        preds = []
        
        for _ in range(self.hparams.n_rotate):
            self.random_rotate_batch(batch)
            loss, pred, targets = self.step(batch)
            preds.append(pred.unsqueeze(-1))
        
        preds = torch.cat(preds, axis=-1).mean(-1)
        
        
        return preds
    
    def on_predict_epoch_end(self, outputs):
        preds = np.array(torch.cat(outputs[0]))
        
        sub_df = pd.read_csv("data/sample_submission.csv")
        sub_df["Reorg_g"] = preds[:, 0]
        sub_df["Reorg_ex"] = preds[:, 1]
        sub_df.to_csv("submission.csv", sep=",", index=False)

        print("Saved submission file!")
        
    def configure_optimizers(self):
        n_steps = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs * n_steps
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
