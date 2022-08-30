import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MinMetric


class BaseNet(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 30,
        optimizer: str = "adamw",
        loss_fn: str = "mse"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "classifier"])
        self.encoder = encoder
        self.classifier = classifier
        
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        elif loss_fn == "huber":
            self.criterion = nn.HuberLoss()
        elif loss_fn == "rmse":
            self.criterion = lambda pred, target: torch.sqrt(F.mse_loss(pred, target))
            
        if optimizer == "adamw":
            self.optim = torch.optim.AdamW
        elif optimizer == "adam":
            self.optim = torch.optim.Adam
            
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        
        self.val_rmse_best = MinMetric()
        
    def forward(self, batch):
        return self.classifier(self.encoder(batch))
    
    def on_train_start(self):
        self.val_rmse_best.reset()
        
    def step(self, batch):
        pred = self(batch)
        loss = self.criterion(pred, batch.y)
        
        return loss, pred, batch.y
    
    def training_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
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
        _, preds, _ = self.step(batch)
        
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
        
        optimizer = self.optim(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs * n_steps
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    

