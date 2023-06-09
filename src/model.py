from typing import List
import torch.nn as nn
from src.layers import GEM1Encoder, MLPClassifier


class GEM1Model(nn.Module):
    def __init__(
        self,
        embed_dim: int = 300,
        encoder_dropout: float = 0.15,
        last_act: bool = False,
        n_convs: int = 5,
        pool: str = "add",
        dist_max: float = 3.0,
        dist_unit: float = 0.1,
        length_max: float = 2.0,
        length_unit: float = 0.1,
        angle_unit: float = 0.1,
        gamma: float = 10.,
        mlp_hidden_dims: List = [256, 256, 256],
        batch_norm: bool = True,
        mlp_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GEM1Encoder(embed_dim, encoder_dropout, last_act, n_convs, pool, 
                                   dist_max, dist_unit, length_max, length_unit,
                                   angle_unit, gamma)
        self.classifier = MLPClassifier(
            embed_dim, mlp_hidden_dims, batch_norm, mlp_dropout
        )

    def forward(self, batch):
        x = self.encoder(batch)
        x = self.classifier(x)
        return x