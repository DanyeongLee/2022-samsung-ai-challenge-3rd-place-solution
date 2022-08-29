import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.main = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.main(x)


class GIN(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        gin_mlp_hidden_dim: int = 512,
        gin_hidden_dim: int = 256,
        n_fc_layers: int = 3,
        fc_hidden_dim: int = 256,
        fc_dropout: float = 0.5,
        pool: str = "mean"
    ):
        super().__init__()
        if pool == "add":
            self.pool = gnn.global_add_pool
        elif pool == "mean":
            self.pool = gnn.global_mean_pool

        self.gnn = gnn.Sequential("x, edge_index, edge_attr, batch", [
            (gnn.GINEConv(MLP(input_dim, gin_mlp_hidden_dim, gin_hidden_dim), edge_dim=3), "x, edge_index, edge_attr -> x1"),
            (gnn.GINEConv(MLP(gin_hidden_dim, gin_mlp_hidden_dim, gin_hidden_dim), edge_dim=3), "x1, edge_index, edge_attr -> x2"),
            (gnn.GINEConv(MLP(gin_hidden_dim, gin_mlp_hidden_dim, gin_hidden_dim), edge_dim=3), "x2, edge_index, edge_attr -> x3"),
            (lambda x1, x2, x3, batch: [self.pool(x, batch) for x in [x1, x2, x3]], "x1, x2, x3, batch -> xs"),
            (gnn.JumpingKnowledge("cat"), "xs -> x")
        ])
        
        
        fc_input_dim = gin_hidden_dim * 3
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )
        
        self.fc_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc_hidden_dim, fc_hidden_dim),
                nn.BatchNorm1d(fc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ) for _ in range(n_fc_layers - 2)
        ])
        self.fc_last = nn.Linear(fc_hidden_dim, 2)
        
    def forward(self, batch):
        x = torch.cat([batch.x, batch.pos_g, batch.pos_ex], axis=1)
        
        x = self.gnn(x, batch.edge_index, batch.edge_attr, batch.batch)
        
        x = self.fc1(x)
        
        for fc_layer in self.fc_list:
            x = fc_layer(x)
            
        x = self.fc_last(x)
        
        return x
    
