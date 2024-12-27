import os
import gzip
import json
import pandas as pd
from itertools import repeat
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool

import rdkit
from rdkit import Chem

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from . import dataset_utils


class MolPair_SingleGraph(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.raw_filepath = os.path.join(self.root, "raw/smiles_descriptions.json")
        with open(self.raw_filepath, 'r') as f:
            description_dict = json.load(f)
        self.SMILES_list = list(description_dict.keys())
        self.description_list = list(description_dict.values())
        
        super(MolPair_SingleGraph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'graph.pt'

    def get_graph(self, smi):
        mol = Chem.MolFromSmiles(smi)
        graph = dataset_utils.mol_to_graph_data_obj_simple(mol)
        return graph
        
    def process(self):
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            RDKit_mol = Chem.MolFromSmiles(SMILES)
            graph = dataset_utils.mol_to_graph_data_obj_simple(RDKit_mol)
            graph_list.append(graph)
        # with mp.Pool(int(mp.cpu_count()/2)) as pool:
        #     graph_list = list(tqdm(pool.imap(self.get_graph, self.SMILES_list), total=len(self.SMILES_list)))


        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]
        description = self.description_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return SMILES, description, data

    def __len__(self):
        return len(self.SMILES_list)
    

class MolPair_PairGraph(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.raw_SMILES_filepath = os.path.join(self.root, "physical_prop/allset/raw_data.csv")
        df = pd.read_csv(self.raw_SMILES_filepath)
        self.SMILES_list = df['smiles'].tolist()
        
        super(MolPair_SingleGraph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'graph.pt'

    def process(self):
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            RDKit_mol = Chem.MolFromSmiles(SMILES)
            graph = dataset_utils.mol_to_graph_data_obj_simple(RDKit_mol)
            graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return SMILES, data

    def __len__(self):
        return len(self.SMILES_list)
    