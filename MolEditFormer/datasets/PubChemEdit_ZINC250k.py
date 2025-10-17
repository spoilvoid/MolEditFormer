import os
import os.path as osp
import random
import numpy as np
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
from MolEditFormer.datasets.dataset_utils import DESCRIPTION_MODE, VALUE_TYPE, PROPERTY_TYPE
from MolEditFormer.datasets.dataset_utils import get_scaffold


class PubChemEdit_ZINC250k(Dataset):
    def __init__(self, root, value_type="continuous", property_type="name", mode="main", mixed=False, mixed_config=None, scaffold_hint=False):
        if mode not in DESCRIPTION_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
        self.mode = mode
        if value_type not in VALUE_TYPE:
            raise ValueError(f"Invalid value_type: {value_type}, choose from {VALUE_TYPE}")
        self.value_type = value_type
        if property_type not in PROPERTY_TYPE:
            raise ValueError(f"Invalid property_type: {property_type}, choose from {PROPERTY_TYPE}")
        self.property_type = property_type
        self.root = root
        self.mixed = mixed
        self.mixed_config = mixed_config
        self.scaffold_hint = scaffold_hint

        self.PubchemEdit_filepath = os.path.join(self.root, "raw", "PubChemEdit.json")
        self.additional_ZINC250k_filepath = os.path.join(self.root, "raw", "additional_ZINC250k.csv")

        self.processed_filepath = os.path.join(self.root, self.mode, "processed.csv")
        os.makedirs(osp.join(self.root, self.mode), exist_ok=True)
        
        if osp.exists(self.processed_filepath):
            processed_df = pd.read_csv(self.processed_filepath)
            self.encoder_input_SMILES_list = processed_df["encoder_input_smiles"].tolist()
            self.decoder_input_SMILES_list = processed_df["decoder_input_smiles"].tolist()
            self.description_list = processed_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.CID_list, self.description_list, self.encoder_input_SMILES_list, self.decoder_input_SMILES_list  = [], [], [], []
        raw_smiles_list = []

        print("Processing PubChemEdit")
        with open(self.PubchemEdit_filepath) as file:
            PubChemEdit_data = json.load(file)
        file.close()

        for item in tqdm(PubChemEdit_data):
            raw_smiles_list.append(item["RDKit_IsoSmiles"])
            if self.mode == "full":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
            elif self.mode == "main":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            if self.scaffold_hint:
                input_smiles_scaffold = get_scaffold(item["RDKit_IsoSmiles"])
                raw_descriptions.append(f"This molecule has SMILES scaffold: {input_smiles_scaffold}.")

            raw_descriptions.append(f"This molecule has SMILES scaffold: {input_smiles_scaffold}.")

            self.description_list.append(" ".join(raw_descriptions))
        
        print("Processing additional ZINC250k")
        additional_ZINC250k_df = pd.read_csv(self.additional_ZINC250k_filepath)
        raw_smiles_list.extend(additional_ZINC250k_df["smiles"].tolist())
        if self.scaffold_hint:
            additional_input_smiles_scaffold = additional_ZINC250k_df["smiles"].apply(get_scaffold).tolist()
            additional_descriptions = additional_ZINC250k_df["description"].apply(lambda x: x + f" This molecule has SMILES scaffold: {additional_input_smiles_scaffold[additional_ZINC250k_df[additional_ZINC250k_df['description'] == x].index[0]]}.").tolist()
            self.description_list.extend(additional_descriptions)
        else:
            self.description_list.extend(additional_ZINC250k_df["description"].tolist())

        print("Processing mixed config for SMILES")
        if self.mixed:
            mix_num = int(len(raw_smiles_list) * (self.mixed_config["non2can_ratio"] + self.mixed_config["non2non_ratio"]))
            non2can_num = int(len(raw_smiles_list) * self.mixed_config["non2can_ratio"])
            mix_indices = random.sample(range(len(raw_smiles_list)), mix_num)
            non2can_indices = mix_indices[:non2can_num]
            non2non_indices = mix_indices[non2can_num:]
            for idx in tqdm(range(len(raw_smiles_list))):
                if idx not in mix_indices:
                    self.encoder_input_SMILES_list.append(raw_smiles_list[idx])
                    self.decoder_input_SMILES_list.append(raw_smiles_list[idx])
                elif idx in non2can_indices:
                    self.encoder_input_SMILES_list.append(dataset_utils.shuffle_atom_order(raw_smiles_list[idx]))
                    self.decoder_input_SMILES_list.append(raw_smiles_list[idx])
                elif idx in non2non_indices:
                    self.encoder_input_SMILES_list.append(dataset_utils.shuffle_atom_order(raw_smiles_list[idx]))
                    self.decoder_input_SMILES_list.append(dataset_utils.shuffle_atom_order(raw_smiles_list[idx]))
        else:
            self.encoder_input_SMILES_list = raw_smiles_list
            self.decoder_input_SMILES_list = raw_smiles_list

        processed_df = pd.DataFrame({"encoder_input_smiles": self.encoder_input_SMILES_list, "decoder_input_smiles": self.decoder_input_SMILES_list, "description": self.description_list})
        processed_df.to_csv(self.processed_filepath, index=None)

    def __getitem__(self, idx):
        encoder_input_SMILES = self.encoder_input_SMILES_list[idx]
        decoder_input_SMILES = self.decoder_input_SMILES_list[idx]
        description  = self.description_list[idx]
        return encoder_input_SMILES, decoder_input_SMILES, description

    def __len__(self):
        return len(self.encoder_input_SMILES_list)
