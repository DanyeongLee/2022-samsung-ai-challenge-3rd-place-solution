from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding



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
        
        return torch.exp(-self.gamma * torch.square(x - self.centers.type_as(x)))
        
    
class RBFEmbedding(nn.Module):
    """
    Bond Length Encoder using Radial Basis Functions
    """
    def __init__(self, embed_dim, rbf_params=None):
        super().__init__()
        centers, gamma = rbf_params
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
            nn.BatchNorm1d(embed_dim * 2),
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
    
    
    
class GEM1Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        dropout: float = 0.1,
        last_act: bool = False,
        n_layers: int = 3,
        pool: str = "mean",
        dist_max: float = 3.0,
        dist_unit: float = 0.1,
        length_max: float = 2.0,
        length_unit: float = 0.1,
        angle_unit: float = 0.1,
        gamma: float = 10.
    ):
        super().__init__()
        self.embed_atom = AtomEncoder(embed_dim)
        self.embed_atom_dist = RBFEmbedding(embed_dim, (np.arange(0, dist_max, dist_unit), gamma))
        self.proj_atom = nn.Linear(2*embed_dim, embed_dim)
        
        self.embed_bond = BondEncoder(embed_dim)
        
        self.embed_bond_length = RBFEmbedding(embed_dim, (np.arange(0, length_max, length_unit), gamma))
        self.project_length = nn.Linear(embed_dim*2, embed_dim)
        
        self.embed_bond_angle = RBFEmbedding(embed_dim, (np.arange(0, np.pi, angle_unit), gamma))
        self.project_angle = nn.Linear(embed_dim*2, embed_dim)
        
        self.atom_gin_layers = nn.ModuleList(
            [GINBlock(embed_dim, dropout, True) for _ in range(n_layers - 1)] + [GINBlock(embed_dim, dropout, last_act)]
        )
        self.bond_gin_layers = nn.ModuleList(
            [GINBlock(embed_dim, dropout, True) for _ in range(n_layers - 1)] + [GINBlock(embed_dim, dropout, last_act)]
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        if pool == "mean":
            self.pool = gnn.global_mean_pool
        elif pool == "add":
            self.pool = gnn.global_add_pool
        
        self.n_layers = n_layers
        
        
    def forward(self, batch):
        atom_x = self.embed_atom(batch.x.to(torch.long))
        atom_diff = self.embed_atom_dist(F.pairwise_distance(batch.pos_g, batch.pos_ex))
        atom_x = self.proj_atom(torch.cat([atom_x, atom_diff], dim=1))
        
        edge_x = self.embed_bond(batch.edge_attr.to(torch.long))
        length_feat = torch.cat([self.embed_bond_length(batch.bond_lengths_g), 
                                 self.embed_bond_length(batch.bond_lengths_ex)], dim=1)
        edge_x = edge_x + self.project_length(length_feat)
        
        angle_feat = torch.cat([self.embed_bond_angle(batch.bond_bond_angles_g), 
                                self.embed_bond_angle(batch.bond_bond_angles_ex)], dim=1)
        angle_x = self.project_angle(angle_feat)
        
        for layer_idx in range(self.n_layers):
            bond_gin = self.bond_gin_layers[layer_idx]
            edge_x = bond_gin(edge_x, batch.bond_bond_index, angle_x, batch.edge_attr_batch)
            
            atom_gin = self.atom_gin_layers[layer_idx]
            atom_x = atom_gin(atom_x, batch.edge_index, edge_x, batch.batch)
        
        x = self.pool(atom_x, batch.batch)
        x = self.final_norm(x)
        
        return x
    


class MLPClassifier(nn.Module):
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
    
    

    
    

    