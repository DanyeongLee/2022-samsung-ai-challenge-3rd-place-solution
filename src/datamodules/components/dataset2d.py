import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 


def mol2graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return x, edge_attr, edge_index


def get_coordinate_features(mol):
    conf = mol.GetConformer()
    return conf.GetPositions()

def get_mol_data(root, prefix, y=None):
    if prefix.startswith("train"):
        set_dir = "train_set"
    else:
        set_dir = "test_set"
        
    ex = Chem.MolFromMolFile(f"{root}/{set_dir}/{prefix}_ex.mol", removeHs=False)
    g = Chem.MolFromMolFile(f"{root}/{set_dir}/{prefix}_g.mol", removeHs=False)
    
    # Atom features
    X, edge_attr, edge_index = mol2graph(ex)
    
    # Atom 3D coordinates
    co_ex = get_coordinate_features(ex)
    co_g = get_coordinate_features(g)
            
    X = np.concatenate([X, co_ex, co_g], axis=1)
    
    X = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([y], dtype=torch.float)
            
    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y)
        

def get_datalist(df, root):
    data_list = []
    if "Reorg_g" in df.columns:
        for _, item in tqdm(df.iterrows()):
            y = [item.Reorg_g, item.Reorg_ex]
            data = get_mol_data(root, item[0], y)
            data_list.append(data)
    else:
        for _, item in tqdm(df.iterrows()):
            data = get_mol_data(root, item[0])
            data_list.append(data)
        
    return data_list


class TrainDataset(InMemoryDataset):
    def __init__(
        self,
        root="/data/project/danyoung/reorg/data/mol_files",
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        mol_list = os.listdir(os.path.join(self.root, "train_set"))
        mol_list = [os.path.join(self.root, "train_set", file) for file in mol_list]
            
        return mol_list

    @property
    def processed_file_names(self):
        return ["2d_dataset_train.pt"]

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(f"{self.root}/../train_set.ReorgE.csv")
        data_list = get_datalist(df, self.root)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    
class TestDataset(InMemoryDataset):
    def __init__(
        self,
        root="/data/project/danyoung/reorg/data/mol_files", 
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        mol_list = os.listdir(os.path.join(self.root, "test_set"))
        mol_list = [os.path.join(self.root, "test_set", file) for file in mol_list]
            
        return mol_list

    @property
    def processed_file_names(self):
        return ["2d_dataset_test.pt"]

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(f"{self.root}/../test_set.csv")
        data_list = get_datalist(df, self.root)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])