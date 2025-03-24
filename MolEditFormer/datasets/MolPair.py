import os
import os.path as osp
import gzip
import json
import random
import re
import numpy as np
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

from MolEditFormer.datasets import dataset_utils
from MolEditFormer.datasets.dataset_utils import DESCRIPTION_MODE, RETRIEVAL_MODE, VERSION, TEXT_REQUIREMENTS_V1, TEXT_REQUIREMENTS_V2, TEXT_REQUIREMENTS_V3, TEXT_REQUIREMENTS_V4, LEVEL_LABELS_PROP_NAME, LEVEL_LABELS_PROP_EXPLANATION, TASK_REFERENCE


class MolPair_SingleGraph(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.raw_filepath = osp.join(self.root, "raw/smiles_descriptions.json")
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
        return osp.join(self.root, 'processed')

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
        self.raw_SMILES_filepath = osp.join(self.root, "physical_prop/allset/raw_data.csv")
        df = pd.read_csv(self.raw_SMILES_filepath)
        self.SMILES_list = df['smiles'].tolist()
        
        super(MolPair_SingleGraph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

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


class MolPair_PairSmiles(Dataset):
    def __init__(self, root, template_path, mode="main", max_num_pairs_per_task=40000, version="v1"):
        self.template_path = template_path
        if mode not in DESCRIPTION_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
        self.mode = mode
        if version not in VERSION:
            raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
        if version in ["v1", "v2"]:
            self.root = osp.join(root, "valued_"+version)
        elif version in ["v3", "v4"]:
            self.root = osp.join(root, "tagged_"+version)
        self.version = version
        self.max_num_pairs_per_task = max_num_pairs_per_task

        self.PubchemEdit_filepath = osp.join(self.root, "raw", "description", "PubChemEdit.json")
        self.additional_ZINC250k_filepath = osp.join(self.root, "raw", "description", "additional_ZINC250k.csv")
        
        self.pair_dir = osp.join(self.root, "raw", "pair")

        self.description_filepath = osp.join(self.root, f"num{self.max_num_pairs_per_task}_"+self.mode, "processed_description.csv")
        self.pair_filepath = osp.join(self.root, f"num{self.max_num_pairs_per_task}_"+self.mode, "processed_pair.csv")
        if not osp.exists(osp.join(self.root, f"num{self.max_num_pairs_per_task}_"+self.mode)):
            os.makedirs(osp.join(self.root, f"num{self.max_num_pairs_per_task}_"+self.mode))
        
        if osp.exists(self.description_filepath) and osp.exists(self.pair_filepath):
            pair_df = pd.read_csv(self.pair_filepath)
            description_df = pd.read_csv(self.description_filepath)
            self.input_smiles_list = pair_df["smiles1"].tolist()
            self.output_smiles_list = pair_df["smiles2"].tolist()
            self.description_list = description_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.input_smiles_list, self.output_smiles_list, self.description_list = [], [], []
        
        description_dict = {}
        print("Processing PubChemEdit")
        with open(self.PubchemEdit_filepath) as file:
            PubChemEdit_data = json.load(file)
        file.close()
        for item in tqdm(PubChemEdit_data):
            if self.mode == "full":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
            elif self.mode == "main":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            description_dict[item["RDKit_IsoSmiles"]] = " ".join(raw_descriptions)

        print("Processing additional ZINC250k")
        additional_ZINC250k_df = pd.read_csv(self.additional_ZINC250k_filepath)
        for idx, row in tqdm(additional_ZINC250k_df.iterrows(), total=len(additional_ZINC250k_df)):
            description_dict[row["smiles"]] = row["description"]
        
        print("Processing templates")
        with open(self.template_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.template_list = [line.strip() for line in lines]

        print("Processing pairs")
        for pair_file in tqdm(os.listdir(self.pair_dir)):
            match = re.search(r'task_(\d+)\.csv', pair_file)
            if match:
                task_id = match.group(1)
            else:
                raise ValueError(f"Invalid pair file: {pair_file}")

            pair_df = pd.read_csv(os.path.join(self.pair_dir, pair_file))

            if len(pair_df) > self.max_num_pairs_per_task:
                pair_df = pair_df.sample(n=self.max_num_pairs_per_task)
            for idx, row in pair_df.iterrows():
                template = random.choice(self.template_list)
                task_description = re.sub(r'\${input}', 'the above molecule', template)
                if self.version == "v1":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[task_id], task_description)
                elif self.version == "v2":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[task_id], task_description)
                elif self.version == "v3":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[task_id], task_description)
                    for key, value in row.items():
                        match = re.search(r'smiles1_level(\d+)', key)
                        if match:
                            input_prop_num = match.group(1)
                            task_description = re.sub(r'\${input_level'+input_prop_num+'}', value, task_description)
                        match = re.search(r'smiles2_level(\d+)', key)
                        if match:
                            input_prop_num = match.group(1)
                            task_description = re.sub(r'\${output_level'+input_prop_num+'}', value, task_description)
                
                self.description_list.append(description_dict[row["smiles1"]] + " " + task_description)
                self.input_smiles_list.append(row["smiles1"])
                self.output_smiles_list.append(row["smiles2"])

        pair_df = pd.DataFrame({"smiles1": self.input_smiles_list, "smiles2": self.output_smiles_list})
        pair_df.to_csv(self.pair_filepath, index=None)

        description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
        description_df.to_csv(self.description_filepath, index=None)

    def __getitem__(self, idx):
        input_smiles = self.input_smiles_list[idx]
        output_smiles = self.output_smiles_list[idx]
        description = self.description_list[idx]
        return input_smiles, output_smiles, description

    def __len__(self):
        return len(self.input_smiles_list)


class MolPair_PairSmiles_Test(Dataset):
    def __init__(self, root, template_path, task_id, mode="random", version="v1"):
        self.template_path = template_path
        if mode not in RETRIEVAL_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
        self.mode = mode
        if version not in VERSION:
            raise ValueError(f"Invalid ask_version: {version}, choose from {VERSION}")
        if version in ["v1", "v2"]:
            self.root = osp.join(root, "valued_"+version)
        elif version in ["v3", "v4"]:
            self.root = osp.join(root, "tagged_"+version)
        self.version = version
        self.task_id = str(task_id)
        self.raw_filepath = osp.join(self.root, "raw", f"raw_data.csv")

        self.processed_filepath = osp.join(self.root, self.mode, f"task_{task_id}_input.csv")
        if not osp.exists(osp.join(self.root, self.mode)):
            os.makedirs(osp.join(self.root, self.mode))
        
        if osp.exists(self.processed_filepath):
            processed_df = pd.read_csv(self.processed_filepath)
            self.input_smiles_list = processed_df["smiles"].tolist()
            self.description_list = processed_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.input_smiles_list, self.description_list = [], []
        
        print("Processing zero-shot raw file") 
        raw_df = pd.read_csv(self.raw_filepath)

        print("Processing templates")
        with open(self.template_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.template_list = [line.strip() for line in lines]

        print("Processing pairs")
        if self.mode == "random":
            for idx, row in tqdm(raw_df.iterrows()):
                template = random.choice(self.template_list)
                task_description = re.sub(r'\${input}', 'the above molecule', template)
                if self.version == "v1":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.task_id], task_description)
                elif self.version == "v2":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.task_id], task_description)
                elif self.version == "v3":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.task_id], task_description)
                    task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
                    for name_key, ref_prop in TASK_REFERENCE[self.task_id].items():
                        prop_level = row[ref_prop+"_level"]
                        task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
                self.description_list.append(row["description"] + " " + task_description)
                self.input_smiles_list.append(row["smiles"])
        elif self.mode == "iterative":
            for idx, row in tqdm(raw_df.iterrows()):
                TASK_REFERENCE[self.task_id] 
                for template in self.template_list:
                    task_description = re.sub(r'\${input}', 'the above molecule', template)
                    if self.version == "v1":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.task_id], task_description)
                    elif self.version == "v2":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.task_id], task_description)
                    elif self.version == "v3":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.task_id], task_description)
                        task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
                        for name_key, ref_prop in TASK_REFERENCE[self.task_id].items():
                            prop_level = row[ref_prop+"_level"]
                            task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
                    self.description_list.append(row["description"] + " " + task_description)
                    self.input_smiles_list.append(row["smiles"])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
        description_df.to_csv(self.processed_filepath, index=None)

    def __getitem__(self, idx):
        input_smiles = self.input_smiles_list[idx]
        description = self.description_list[idx]
        return input_smiles, description

    def __len__(self):
        return len(self.input_smiles_list)


class MolPair_DockingSmiles(Dataset):
    def __init__(self, root, template_path, mode="main", target_name="COX2", version="v1"):
        self.template_path = template_path
        if mode not in DESCRIPTION_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
        self.mode = mode
        if version not in VERSION:
            raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
        if version in ["v1", "v2"]:
            self.root = osp.join(root, "valued_"+version)
        elif version in ["v3", "v4"]:
            self.root = osp.join(root, "tagged_"+version)
        self.version = version
        self.target_name = target_name

        self.PubchemEdit_filepath = osp.join(self.root, "raw", "description", "PubChemEdit.json")
        self.target_normal_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_normal_description.csv")
        self.target_docking_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_docking_description.csv")
        
        self.pair_filepath = osp.join(self.root, "raw", "pair", f"{target_name}.csv")

        self.processed_description_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_description.csv")
        self.processed_pair_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_pair.csv")
        if not osp.exists(osp.join(self.root, f"{target_name}_"+self.mode)):
            os.makedirs(osp.join(self.root, f"{target_name}_"+self.mode))
        
        if osp.exists(self.processed_description_filepath) and osp.exists(self.processed_pair_filepath):
            pair_df = pd.read_csv(self.processed_pair_filepath)
            description_df = pd.read_csv(self.processed_description_filepath)
            self.input_smiles_list = pair_df["smiles1"].tolist()
            self.output_smiles_list = pair_df["smiles2"].tolist()
            self.description_list = description_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.input_smiles_list, self.output_smiles_list, self.description_list = [], [], []
        
        normal_description_dict, docking_description_dict = {}, {}
        print("Processing PubChemEdit")
        with open(self.PubchemEdit_filepath) as file:
            PubChemEdit_data = json.load(file)
        file.close()
        for item in tqdm(PubChemEdit_data):
            if self.mode == "full":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
            elif self.mode == "main":
                raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            normal_description_dict[item["RDKit_IsoSmiles"]] = " ".join(raw_descriptions)

        print("Processing additional normal descriptions")
        additional_normal_description_df = pd.read_csv(self.target_normal_description_filepath)
        for idx, row in tqdm(additional_normal_description_df.iterrows(), total=len(additional_normal_description_df)):
            normal_description_dict[row["smiles"]] = row["description"]

        print("Processing additional docking descriptions")
        docking_description_df = pd.read_csv(self.target_docking_description_filepath)
        for idx, row in tqdm(docking_description_df.iterrows(), total=len(docking_description_df)):
            docking_description_dict[row["smiles"]] = row["description"]
        
        print("Processing templates")
        with open(self.template_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.template_list = [line.strip() for line in lines]

        print("Processing pairs")
        pair_df = pd.read_csv(self.pair_filepath)
        for idx, row in pair_df.iterrows():
            template = random.choice(self.template_list)
            task_description = re.sub(r'\${input}', 'the above molecule', template)
            if self.version == "v1":
                task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
            elif self.version == "v2":
                task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
            elif self.version == "v3":
                task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
                for key, value in row.items():
                    match = re.search(r'smiles1_level(\d+)', key)
                    if match:
                        input_prop_num = match.group(1)
                        task_description = re.sub(r'\${input_level'+input_prop_num+'}', value, task_description)
                    match = re.search(r'smiles2_level(\d+)', key)
                    if match:
                        input_prop_num = match.group(1)
                        task_description = re.sub(r'\${output_level'+input_prop_num+'}', value, task_description)
            
            self.description_list.append(normal_description_dict[row["smiles1"]] + " " + docking_description_dict[row["smiles1"]] + " " + task_description)
            self.input_smiles_list.append(row["smiles1"])
            self.output_smiles_list.append(row["smiles2"])

        pair_df = pd.DataFrame({"smiles1": self.input_smiles_list, "smiles2": self.output_smiles_list})
        pair_df.to_csv(self.processed_pair_filepath, index=None)

        description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
        description_df.to_csv(self.processed_description_filepath, index=None)

    def __getitem__(self, idx):
        input_smiles = self.input_smiles_list[idx]
        output_smiles = self.output_smiles_list[idx]
        description = self.description_list[idx]
        return input_smiles, output_smiles, description

    def __len__(self):
        return len(self.input_smiles_list)
    

class MolPair_DockingSmiles_Test(Dataset):
    def __init__(self, root, template_path, target_name="COX2", mode="main", version="v1"):
        self.template_path = template_path
        if mode not in RETRIEVAL_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
        self.mode = mode
        if version not in VERSION:
            raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
        if version in ["v1", "v2"]:
            self.root = osp.join(root, "valued_"+version)
        elif version in ["v3", "v4"]:
            self.root = osp.join(root, "tagged_"+version)
        self.version = version
        self.target_name = target_name

        self.raw_filepath = osp.join(self.root, "raw", f"{target_name}.csv")

        self.processed_filepath = osp.join(self.root, self.mode, f"{target_name}_input.csv")
        if not osp.exists(osp.join(self.root, self.mode)):
            os.makedirs(osp.join(self.root, self.mode))
        
        if osp.exists(self.processed_filepath):
            processed_df = pd.read_csv(self.processed_filepath)
            self.input_smiles_list = processed_df["smiles"].tolist()
            self.description_list = processed_df["description"].tolist()
        else:
            self.process()

    def process(self):
        self.input_smiles_list, self.description_list = [], []
        
        print("Processing raw file")
        raw_df = pd.read_csv(self.raw_filepath)
        
        print("Processing templates")
        with open(self.template_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.template_list = [line.strip() for line in lines]

        print("Processing pairs")
        if self.mode == "random":
            for idx, row in tqdm(raw_df.iterrows()):
                template = random.choice(self.template_list)
                task_description = re.sub(r'\${input}', 'the above molecule', template)
                if self.version == "v1":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
                elif self.version == "v2":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
                elif self.version == "v3":
                    task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
                    task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
                    for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
                        prop_level = row[ref_prop+"_level"]
                        task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
                self.description_list.append(row["description"] + " " + task_description)
                self.input_smiles_list.append(row["smiles"])
        elif self.mode == "iterative":
            for idx, row in tqdm(raw_df.iterrows()):
                for template in self.template_list:
                    task_description = re.sub(r'\${input}', 'the above molecule', template)
                    if self.version == "v1":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
                    elif self.version == "v2":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
                    elif self.version == "v3":
                        task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
                        task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
                        for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
                            prop_level = row[ref_prop+"_level"]
                            task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
                    self.description_list.append(row["description"] + " " + task_description)
                    self.input_smiles_list.append(row["smiles"])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
        description_df.to_csv(self.processed_filepath, index=None)

    def __getitem__(self, idx):
        input_smiles = self.input_smiles_list[idx]
        description = self.description_list[idx]
        return input_smiles, description

    def __len__(self):
        return len(self.input_smiles_list)