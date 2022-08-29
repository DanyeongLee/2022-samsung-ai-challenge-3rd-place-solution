from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma):
        super().__init__()
        self.centers = torch.tensor(centers, dtype=torch.float).unsqueeze(0)
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (N, 1).
        Returns:
            y(tensor): (N, n_centers)
        """
        x = x.view(-1, 1)
        
        return torch.exp(-self.gamma * torch.square(x - self.centers.to(x.device)))
        

class BondLengthRBF(nn.Module):
    """
    Bond Length Encoder using Radial Basis Functions
    """
    def __init__(self, embed_dim, rbf_params=None):
        super().__init__()

        if rbf_params is None:
            self.rbf_params = (np.arange(0, 2, 0.1), 10.0)   # (centers, gamma)
        else:
            self.rbf_params = rbf_params

        centers, gamma = self.rbf_params
        self.rbf = RBF(centers, gamma)
        self.fc = nn.Linear(len(centers), embed_dim)

    def forward(self, bond_lengths):
        rbf_x = self.rbf(bond_lengths)
        out_embed = self.fc(rbf_x)
        
        return out_embed
    

class BondAngleRBF(nn.Module):
    """
    Bond Length Encoder using Radial Basis Functions
    """
    def __init__(self, embed_dim, rbf_params=None):
        super().__init__()

        if rbf_params is None:
            self.rbf_params = (np.arange(0, np.pi, 0.1), 10.0)   # (centers, gamma)
        else:
            self.rbf_params = rbf_params

        centers, gamma = self.rbf_params
        self.rbf = RBF(centers, gamma)
        self.fc = nn.Linear(len(centers), embed_dim)

    def forward(self, bond_lengths):
        rbf_x = self.rbf(bond_lengths)
        out_embed = self.fc(rbf_x)
        
        return out_embed


class MLP(nn.Module):
    def __init__(
        self,
        embed_dim
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        return self.main(x)

    

class GINBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        dropout,
        last_act
    ):
        super().__init__()
        self.gin = gnn.GINEConv(MLP(embed_dim))
        self.ln = gnn.LayerNorm(embed_dim)
        self.gn = gnn.GraphNorm(embed_dim)
        self.do = nn.Dropout(dropout)
        self.act = nn.ReLU() if last_act else nn.Identity()
        
    def forward(self, x, edge_index, edge_attr, batch):
        out = self.gin(x, edge_index, edge_attr)
        out = self.ln(out, batch)
        out = self.gn(out, batch)
        out = self.do(out)
        out = self.act(out)
        
        return x + out
    
    
    
class GEM1(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        dropout: float = 0.1,
        last_act: bool = True,
        n_layers: int = 3,
        pool: str = "mean"
    ):
        super().__init__()
        self.embed_atom = nn.Linear(9, embed_dim)
        self.embed_bond = nn.Linear(3, embed_dim)
        self.embed_bond_length = BondLengthRBF(embed_dim)
        self.embed_bond_angle = BondAngleRBF(embed_dim)
        
        self.atom_gin_layers = nn.ModuleList([GINBlock(embed_dim, dropout, last_act) for _ in range(n_layers)])
        self.bond_gin_layers = nn.ModuleList([GINBlock(embed_dim, dropout, last_act) for _ in range(n_layers)])
        
        if pool == "mean":
            self.pool = gnn.global_mean_pool
        elif pool == "add":
            self.pool = gnn.global_add_pool
        
        self.n_layers = n_layers
        
        
    def forward(self, batch):
        atom_x = self.embed_atom(batch.x)
        
        edge_x = self.embed_bond(batch.edge_attr)
        edge_x = edge_x + self.embed_bond_length(batch.bond_lengths_g) + self.embed_bond_length(batch.bond_lengths_ex)
        
        angle_x = self.embed_bond_angle(batch.bond_bond_angles_g) + self.embed_bond_angle(batch.bond_bond_angles_ex)
        
        for layer_idx in range(self.n_layers):
            bond_gin = self.bond_gin_layers[layer_idx]
            edge_x = bond_gin(edge_x, batch.bond_bond_index, angle_x, batch.edge_attr_batch)
            
            atom_gin = self.atom_gin_layers[layer_idx]
            atom_x = atom_gin(atom_x, batch.edge_index, edge_x, batch.batch)
        
        x = self.pool(atom_x, batch.batch)
        
        return x
    

class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: List = [256, 256],
        batch_norm: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]) if batch_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout()
                ) for i in range(len(hidden_dims) - 1)
            ]
        )
        
        self.fc_last = nn.Linear(hidden_dims[-1], 2)
        
    def forward(self, x):
        x = self.fc1(x)
        
        for fc in self.fc_list:
            x = fc(x)
        
        return self.fc_last(x)