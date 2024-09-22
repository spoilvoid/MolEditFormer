import os
import gzip
import json
import pandas as pd
from itertools import repeat
from tqdm.auto import tqdm


import rdkit
from rdkit import Chem

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from . import dataset_utils


class PubChemEdit(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None, default_filenum=32):
        self.root = root

        # only for `process` function
        self.SDF_filepath = os.path.join(self.root, "raw/structures.sdf.gz")
        self.json_filedir = os.path.join(self.root, "raw/record")
        self.json_filenum = default_filenum
        self.SMILES_filepath = os.path.join(self.root, "processed/SMILES.csv")
        self.description_filepath = os.path.join(self.root, "processed/description.csv")
        
        # using InMemoryDataset.process to process smiles data to graph data
        super(PubChemEdit, self).__init__(root, transform, pre_transform, pre_filter)

        SMILES_df = pd.read_csv(self.SMILES_filepath)
        description_df = pd.read_csv(self.description_filepath)
        
        self.SMILES_list = SMILES_df["SMILES"].tolist()
        self.description_list = description_df["description"].tolist()
        self.graphs, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'graph.pt'

    def process(self):
        self.CID_list, self.description_list, self.SMILES_list  = [], [], []
        print(f"Processing {self.json_filenum} .json record files")
        for i in range(self.json_filenum):
            with open(os.path.join(self.json_filedir, f"Record_Description_{i}.json")) as file:
                data = json.load(file)
            file.close()
            for item in tqdm(data):
                self.CID_list.append(int(item["CID"]))
                self.SMILES_list.append(item["RDKit_IsoSmiles"])
                
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
                self.description_list.append(" ".join(raw_descriptions))
        
        # save extra no-graph data
        SMILES_df = pd.DataFrame({"CID": self.CID_list, "SMILES": self.SMILES_list})
        SMILES_df.to_csv(self.SMILES_filepath, index=None)

        description_df = pd.DataFrame({"CID": self.CID_list, "description": self.description_list})
        description_df.to_csv(self.description_filepath, index=None)

        gzip_loader = gzip.open(self.SDF_filepath)
        suppl = Chem.ForwardSDMolSupplier(gzip_loader)

        graph_list = []
        for mol in tqdm(suppl):
            graph = dataset_utils.mol_to_graph_data_obj_simple(mol)
            graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])

    def get(self, idx):
        SMILES = self.SMILES_list[idx]
        description  = self.description_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return SMILES, description, data

    def __len__(self):
        return len(self.SMILES_list)
