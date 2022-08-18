import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class GATNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        gat_hidden_dim: int = 64,
        edge_dim: int = 3,
        heads: int = 4,
        n_gat_layers: int = 3,
        n_fc_layers: int = 3,
        fc_hidden_dim: int = 256,
        fc_dropout: float = 0.5
    ):
        super().__init__()
        self.gat1 = gnn.GATv2Conv(in_channels=input_dim, 
                                  out_channels=gat_hidden_dim, heads=heads, edge_dim=edge_dim)
        self.gat_list = nn.ModuleList([
            gnn.GATv2Conv(in_channels=gat_hidden_dim*heads, 
                          out_channels=gat_hidden_dim, heads=heads, edge_dim=edge_dim)
            for _ in range(n_gat_layers - 1)
        ])
        self.fc1 = nn.Sequential(
            nn.Linear(gat_hidden_dim * heads, fc_hidden_dim),
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
        self.do = nn.Dropout(fc_dropout)
        self.fc_last = nn.Linear(fc_hidden_dim, 2)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        
        for gat_layer in self.gat_list:
            x = gat_layer(x, edge_index, edge_attr)
            x = F.relu(x)
            
        x = gnn.global_mean_pool(x, batch)
        x = self.fc1(x)
        
        for fc_layer in self.fc_list:
            x = fc_layer(x)
            
        x = self.fc_last(x)
        
        return x

