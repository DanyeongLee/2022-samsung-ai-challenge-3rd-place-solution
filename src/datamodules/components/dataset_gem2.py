import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleRad

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


from ogb.utils.features import allowable_features


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
    
def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


bond_dir_list = ["NONE", "BEGINWEDGE", "BEGINDASH", "ENDDOWNRIGHT", "ENDUPRIGHT", "EITHERDOUBLE", "UNKNOWN"]


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                allowable_features['possible_is_in_ring_list'].index(bond.IsInRing()),
                bond_dir_list.index(str(bond.GetBondDir()))
            ]
    
    return bond_feature


def mol2graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    conf = mol.GetConformer()
    
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
        edge_lengths_list = []
        
        for bond_idx, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)
            edge_length = GetBondLength(conf, i, j)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edge_lengths_list.append(edge_length)
            
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_lengths_list.append(edge_length)
            
        bond_bond_list = []
        bond_bond_angles_list = []
            
        for edge_idx, edge in enumerate(edges_list):
            i, j = edge
            for another_edge_idx, another_edge in enumerate(edges_list):
                if j == another_edge[0]:  # connected
                    bond_bond_list.append((edge_idx, another_edge_idx))
                    bond_bond_angles_list.append(GetAngleRad(conf, i, j, another_edge[1]))
                    

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
        bond_lengths = np.array(edge_lengths_list, dtype=np.float32)
        
        bond_bond_index = np.array(bond_bond_list, dtype=np.int64).T
        bond_bond_angles = np.array(bond_bond_angles_list, dtype=np.float32)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return x, edge_attr, edge_index, bond_lengths, bond_bond_index, bond_bond_angles


def get_coordinate_features(mol):
    conf = mol.GetConformer()
    return conf.GetPositions()

def get_mol_data(root, prefix, removeHs, y=None):
    if prefix.startswith("train"):
        set_dir = "train_set"
    else:
        set_dir = "test_set"
        
    ex = Chem.MolFromMolFile(f"{root}/{set_dir}/{prefix}_ex.mol", removeHs=removeHs)
    g = Chem.MolFromMolFile(f"{root}/{set_dir}/{prefix}_g.mol", removeHs=removeHs)
    
    # Atom features
    X, edge_attr, edge_index, bond_lengths_ex, bond_bond_index, bond_bond_angles_ex = mol2graph(ex)
    X, edge_attr, edge_index, bond_lengths_g, bond_bond_index, bond_bond_angles_g = mol2graph(g)
    
    bond_lengths_ex = torch.tensor(bond_lengths_ex, dtype=torch.float)
    bond_lengths_g = torch.tensor(bond_lengths_g, dtype=torch.float)
    
    bond_bond_index = torch.tensor(bond_bond_index, dtype=torch.long)
    
    bond_bond_angles_ex = torch.tensor(bond_bond_angles_ex, dtype=torch.float)
    bond_bond_angles_g = torch.tensor(bond_bond_angles_g, dtype=torch.float)
    
    # Atom 3D coordinates
    co_ex = get_coordinate_features(ex)
    co_g = get_coordinate_features(g)
    
    X = torch.tensor(X, dtype=torch.float)
    co_ex = torch.tensor(co_ex, dtype=torch.float)
    co_g = torch.tensor(co_g, dtype=torch.float)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([y], dtype=torch.float)
            
    return Data(x=X, pos_g=co_g, pos_ex=co_ex, 
                edge_index=edge_index, edge_attr=edge_attr, 
                bond_lengths_ex=bond_lengths_ex, bond_lengths_g=bond_lengths_g,
                bond_bond_index=bond_bond_index, 
                bond_bond_angles_ex=bond_bond_angles_ex, bond_bond_angles_g=bond_bond_angles_g,
                y=y)
        

def get_datalist(df, root, removeHs):
    data_list = []
    if "Reorg_g" in df.columns:
        for _, item in tqdm(df.iterrows()):
            y = [item.Reorg_g, item.Reorg_ex]
            data = get_mol_data(root, item[0], removeHs, y)
            data_list.append(data)
    else:
        for _, item in tqdm(df.iterrows()):
            data = get_mol_data(root, item[0], removeHs)
            data_list.append(data)
        
    return data_list


class TrainDataset(InMemoryDataset):
    def __init__(
        self,
        root="/data/project/danyoung/reorg/data/mol_files",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        removeHs=False
    ):
        self.removeHs = removeHs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        mol_list = os.listdir(os.path.join(self.root, "train_set"))
        mol_list = [os.path.join(self.root, "train_set", file) for file in mol_list]
            
        return mol_list

    @property
    def processed_file_names(self):
        if self.removeHs:
            return ["gem2_dataset_train.pt"]
        else:
            return ["gem2_dataset_H_train.pt"]

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(f"{self.root}/../train_set.ReorgE.csv")
        data_list = get_datalist(df, self.root, self.removeHs)

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
        pre_filter=None,
        removeHs=False
    ):
        self.removeHs = removeHs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        mol_list = os.listdir(os.path.join(self.root, "test_set"))
        mol_list = [os.path.join(self.root, "test_set", file) for file in mol_list]
            
        return mol_list

    @property
    def processed_file_names(self):
        if self.removeHs:
            return ["gem2_dataset_test.pt"]
        else:
            return ["gem2_dataset_H_test.pt"]

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(f"{self.root}/../test_set.csv")
        data_list = get_datalist(df, self.root, self.removeHs)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])