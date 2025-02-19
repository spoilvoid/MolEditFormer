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

from MolEditFormer.datasets import dataset_utils
from MolEditFormer.datasets.dataset_utils import DESCRIPTION_MODE, VERSION


class PubChemEdit_ZINC250k(Dataset):
    def __init__(self, root, mode="full", can_smiles=True, version="v1"):
        if mode not in DESCRIPTION_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
        self.mode = mode
        if version not in VERSION:
            raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
        if version in ["v1", "v2"]:
            self.root = osp.join(self.root, "valued_"+version)
        elif version in ["v3", "v4"]:
            self.root = osp.join(self.root, "tagged_"+version)
        self.can_smiles = can_smiles

        self.PubchemEdit_filepath = os.path.join(self.root, "raw", "PubChemEdit.json")
        self.additional_ZINC250k_filepath = os.path.join(self.root, "raw", "additional_ZINC250k.csv")

        self.description_filepath = os.path.join(self.root, self.mode, "processed_description.csv")
        self.SMILES_filepath = os.path.join(self.root, self.mode, "processed_SMILES.csv")
        if not os.path.exists(os.path.join(self.root, self.mode)):
            os.makedirs(os.path.join(self.root, self.mode))
        
        if os.path.exists(self.description_filepath) and os.path.exists(self.SMILES_filepath):
            SMILES_df = pd.read_csv(self.SMILES_filepath)
            description_df = pd.read_csv(self.description_filepath)
            self.SMILES_list = SMILES_df["smiles"].tolist()
            self.description_list = description_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.CID_list, self.description_list, self.SMILES_list  = [], [], []
        print("Processing PubChemEdit")
        with open(self.PubchemEdit_filepath) as file:
            PubChemEdit_data = json.load(file)
        file.close()
        for item in tqdm(PubChemEdit_data):
            self.CID_list.append(int(item["CID"]))
            if self.can_smiles:
                self.SMILES_list.append(item["RDKit_CanSmiles"])
            else:
                self.SMILES_list.append(item["RDKit_IsoSmiles"])
            
            if self.mode == "full":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
            elif self.mode == "main":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            self.description_list.append(" ".join(raw_descriptions))
        
        print("Processing additional ZINC250k")
        additional_ZINC250k_df = pd.read_csv(self.additional_ZINC250k_filepath)
        self.CID_list.extend(len(additional_ZINC250k_df) * [-1])
        self.SMILES_list.extend(additional_ZINC250k_df["smiles"].tolist())
        self.description_list.extend(additional_ZINC250k_df["description"].tolist())
        
        SMILES_df = pd.DataFrame({"CID": self.CID_list, "smiles": self.SMILES_list})
        SMILES_df.to_csv(self.SMILES_filepath, index=None)

        description_df = pd.DataFrame({"CID": self.CID_list, "description": self.description_list})
        description_df.to_csv(self.description_filepath, index=None)

    def __getitem__(self, idx):
        SMILES = self.SMILES_list[idx]
        description  = self.description_list[idx]
        return SMILES, description

    def __len__(self):
        return len(self.SMILES_list)
