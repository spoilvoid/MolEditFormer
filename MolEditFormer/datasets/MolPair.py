import os
import os.path as osp
import gzip
import json
import random
import re
from typing import Union
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
from MolEditFormer.datasets.dataset_utils import DESCRIPTION_MODE, RETRIEVAL_MODE, VALUE_TYPE, PROPERTY_TYPE
from MolEditFormer.datasets.dataset_utils import TASK_DICT, PROPERTY_EXPLANATION
from MolEditFormer.datasets.dataset_utils import LEVEL_LABELS_PROPERTY_NAME, LEVEL_LABELS_PROPERTY_EXPLANATION, LEVEL_LBAEL_GAP
from MolEditFormer.datasets.dataset_utils import get_scaffold, get_property_label


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
    def __init__(
        self, 
        root: str,
        template_path: str,
        value_type: str = "continuous",
        property_type: str = "name",
        mode: str = "main",
        max_num_pairs_per_task: int = 40000,
        scaffold_hint: bool = False,
    ):
        self.template_path = template_path
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
        self.scaffold_hint = scaffold_hint
        self.max_num_pairs_per_task = max_num_pairs_per_task

        self.PubchemEdit_filepath = osp.join(self.root, "raw", "description", "PubChemEdit.json")
        self.additional_ZINC250k_filepath = osp.join(self.root, "raw", "description", "additional_ZINC250k.csv")
        
        self.pair_dir = osp.join(self.root, "raw", "pair")

        processed_folder_name = f"{self.mode}_num{self.max_num_pairs_per_task}_scaffold" if scaffold_hint else f"{self.mode}_num{self.max_num_pairs_per_task}"
        self.description_filepath = osp.join(self.root, processed_folder_name, "processed_description.csv")
        self.pair_filepath = osp.join(self.root, processed_folder_name, "processed_pair.csv")
        os.makedirs(osp.join(self.root, processed_folder_name), exist_ok=True)
        
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

            print(f"Processing task {task_id} from {pair_file}")
            pair_df = pd.read_csv(os.path.join(self.pair_dir, pair_file))
            if len(pair_df) > self.max_num_pairs_per_task:
                pair_df = pair_df.sample(n=self.max_num_pairs_per_task)

            for idx, row in pair_df.iterrows():
                template = random.choice(self.template_list)
                task_description = re.sub(r'\${input}', 'the above molecule', template)

                task_requirement = ""
                for verb, prop_list in TASK_DICT[task_id].items():
                    for prop in prop_list:
                        input_prop = row[f"smiles1_{prop}"]
                        output_prop = row[f"smiles2_{prop}"]

                        if self.property_type == "name":
                            verb_explanation = verb
                            prop_explanation = prop
                            LEVEL_LABELS = LEVEL_LABELS_PROPERTY_NAME
                        elif self.property_type == "explanation":
                            verb_explanation = PROPERTY_EXPLANATION[verb]
                            prop_explanation = PROPERTY_EXPLANATION[prop]
                            LEVEL_LABELS = LEVEL_LABELS_PROPERTY_EXPLANATION
                        
                        if self.value_type == "continuous":
                            task_requirement += f"{verb_explanation} {prop_explanation} from {input_prop:.2f} to {output_prop:.2f} and "
                        elif self.value_type == "discrete":
                            input_prop_level = get_property_label(x=input_prop, property_name=prop, level_dict=LEVEL_LABELS)
                            output_prop_level = get_property_label(x=output_prop, property_name=prop, level_dict=LEVEL_LABELS)
                            level_num = abs((output_prop - input_prop) / float(LEVEL_LBAEL_GAP[prop]))
                            task_requirement += f"{verb_explanation} {prop_explanation} from {input_prop_level} to {output_prop_level} with {level_num:.2f} levels and "
                task_requirement = task_requirement[:-5]
                task_description = re.sub(r'\${requirement}', task_requirement, task_description)

                if self.scaffold_hint:
                    input_smiles_scaffold = get_scaffold(row["smiles1"])
                    scaffold_pattern = r'\${scaffold}' if input_smiles_scaffold else r' \${scaffold}'
                    task_description = re.sub(scaffold_pattern, input_smiles_scaffold.replace("\\", "\\\\"), task_description)
                
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
    def __init__(
        self, 
        root: str,
        template_path: str,
        task_id: Union[int, str],
        value_type: str = "continuous",
        property_type: str = "name",
        mode: str = "random",
        threshold_dict: dict = {},
        scaffold_hint: bool = False,
    ):
        self.template_path = template_path
        if mode not in RETRIEVAL_MODE:
            raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
        self.mode = mode
        if value_type not in VALUE_TYPE:
            raise ValueError(f"Invalid value_type: {value_type}, choose from {VALUE_TYPE}")
        self.value_type = value_type
        if property_type not in PROPERTY_TYPE:
            raise ValueError(f"Invalid property_type: {property_type}, choose from {PROPERTY_TYPE}")
        self.property_type = property_type
        self.root = root
        self.scaffold_hint = scaffold_hint
        self.threshold_dict = threshold_dict
        self.task_id = str(task_id)
        self.raw_filepath = osp.join(self.root, "raw", f"raw_data.csv")

        processed_folder_name = f"{self.mode}_scaffold" if scaffold_hint else self.mode
        self.processed_filepath = osp.join(self.root, processed_folder_name, f"task_{task_id}_input.csv")
        os.makedirs(osp.join(self.root, processed_folder_name), exist_ok=True)
        
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
        for idx, row in tqdm(raw_df.iterrows()):
            if self.mode == "random":
                template_list = [random.choice(self.template_list)]
            elif self.mode == "iterative":
                template_list = self.template_list
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
            
            for template in template_list:
                task_description = re.sub(r'\${input}', 'the above molecule', template)

                task_requirement = ""
                for verb, prop_list in TASK_DICT[self.task_id].items():
                    for prop in prop_list:
                        input_prop = row[prop]
                        optimized_threshold = self.threshold_dict.get(prop, 0)
                        if verb == "increase":
                            output_prop = row[prop] + optimized_threshold
                            comparison = "greater than"
                        elif verb == "decrease":
                            output_prop = row[prop] - optimized_threshold
                            comparison = "less than"

                        if self.property_type == "name":
                            verb_explanation = verb
                            prop_explanation = prop
                            LEVEL_LABELS = LEVEL_LABELS_PROPERTY_NAME
                        elif self.property_type == "explanation":
                            verb_explanation = PROPERTY_EXPLANATION[verb]
                            prop_explanation = PROPERTY_EXPLANATION[prop]
                            LEVEL_LABELS = LEVEL_LABELS_PROPERTY_EXPLANATION
                        
                        if self.value_type == "continuous":
                            task_requirement += f"{verb_explanation} {prop_explanation} from {input_prop:.2f} to {comparison} {output_prop:.2f} and "
                        elif self.value_type == "discrete":
                            input_prop_level = get_property_label(x=input_prop, property_name=prop, level_dict=LEVEL_LABELS)
                            output_prop_level = get_property_label(x=output_prop, property_name=prop, level_dict=LEVEL_LABELS)
                            level_num = abs((output_prop - input_prop) / float(LEVEL_LBAEL_GAP[prop]))
                            task_requirement += f"{verb_explanation} {prop_explanation} from {input_prop_level} to {comparison} {output_prop_level} with at least {level_num:.2f} levels and "
                task_requirement = task_requirement[:-5]
                task_description = re.sub(r'\${requirement}', task_requirement, task_description)

                if self.scaffold_hint:
                    input_smiles_scaffold = get_scaffold(row["smiles1"])
                    scaffold_pattern = r'\${scaffold}' if input_smiles_scaffold else r' \${scaffold}'
                    task_description = re.sub(scaffold_pattern, input_smiles_scaffold, task_description)
                
                self.description_list.append(row["description"] + " " + task_description)
                self.input_smiles_list.append(row["smiles"])

        description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
        description_df.to_csv(self.processed_filepath, index=None)

    def __getitem__(self, idx):
        input_smiles = self.input_smiles_list[idx]
        description = self.description_list[idx]
        return input_smiles, description

    def __len__(self):
        return len(self.input_smiles_list)


# class MolPair_DockingSmiles_BindingAffinity(Dataset):
#     def __init__(self, root, template_path, mode="main", target_name="COX2", version="v1"):
#         self.template_path = template_path
#         if mode not in DESCRIPTION_MODE:
#             raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
#         self.mode = mode
#         if version not in VERSION:
#             raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
#         if version in ["v1", "v2"]:
#             self.root = osp.join(root, "valued_"+version)
#         elif version in ["v3", "v4"]:
#             self.root = osp.join(root, "tagged_"+version)
#         self.version = version
#         self.target_name = target_name

#         self.PubchemEdit_filepath = osp.join(self.root, "raw", "description", "PubChemEdit.json")
#         self.target_normal_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_normal_description.csv")
#         self.target_docking_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_docking_description.csv")
        
#         self.pair_filepath = osp.join(self.root, "raw", "pair", f"{target_name}.csv")

#         self.processed_description_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_description.csv")
#         self.processed_pair_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_pair.csv")
#         if not osp.exists(osp.join(self.root, f"{target_name}_"+self.mode)):
#             os.makedirs(osp.join(self.root, f"{target_name}_"+self.mode))
        
#         if osp.exists(self.processed_description_filepath) and osp.exists(self.processed_pair_filepath):
#             pair_df = pd.read_csv(self.processed_pair_filepath)
#             description_df = pd.read_csv(self.processed_description_filepath)
#             self.input_smiles_list = pair_df["smiles1"].tolist()
#             self.output_smiles_list = pair_df["smiles2"].tolist()
#             self.description_list = description_df["description"].tolist()
#         else:
#             self.process()

#     def process(self):
#         self.input_smiles_list, self.output_smiles_list, self.description_list = [], [], []
        
#         normal_description_dict, docking_description_dict = {}, {}
#         print("Processing PubChemEdit")
#         with open(self.PubchemEdit_filepath) as file:
#             PubChemEdit_data = json.load(file)
#         file.close()
#         for item in tqdm(PubChemEdit_data):
#             if self.mode == "full":
#                 raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
#             elif self.mode == "main":
#                 raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
#             else:
#                 raise ValueError(f"Invalid mode: {self.mode}")
#             normal_description_dict[item["RDKit_IsoSmiles"]] = " ".join(raw_descriptions)

#         print("Processing additional normal descriptions")
#         additional_normal_description_df = pd.read_csv(self.target_normal_description_filepath)
#         for idx, row in tqdm(additional_normal_description_df.iterrows(), total=len(additional_normal_description_df)):
#             normal_description_dict[row["smiles"]] = row["description"]

#         print("Processing additional docking descriptions")
#         docking_description_df = pd.read_csv(self.target_docking_description_filepath)
#         for idx, row in tqdm(docking_description_df.iterrows(), total=len(docking_description_df)):
#             docking_description_dict[row["smiles"]] = row["description"]
        
#         print("Processing templates")
#         with open(self.template_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#         self.template_list = [line.strip() for line in lines]

#         print("Processing pairs")
#         pair_df = pd.read_csv(self.pair_filepath)
#         for idx, row in pair_df.iterrows():
#             template = random.choice(self.template_list)
#             task_description = re.sub(r'\${input}', 'the above molecule', template)
#             if self.version == "v1":
#                 task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
#             # elif self.version == "v2":
#             #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
#             elif self.version == "v3":
#                 task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
#                 for key, value in row.items():
#                     match = re.search(r'smiles1_level(\d+)', key)
#                     if match:
#                         input_prop_num = match.group(1)
#                         task_description = re.sub(r'\${input_level'+input_prop_num+'}', value, task_description)
#                     match = re.search(r'smiles2_level(\d+)', key)
#                     if match:
#                         input_prop_num = match.group(1)
#                         task_description = re.sub(r'\${output_level'+input_prop_num+'}', "another level", task_description)
            
#             self.description_list.append(normal_description_dict[row["smiles1"]] + " " + docking_description_dict[row["smiles1"]] + " " + task_description)
#             self.input_smiles_list.append(row["smiles1"])
#             self.output_smiles_list.append(row["smiles2"])

#         pair_df = pd.DataFrame({"smiles1": self.input_smiles_list, "smiles2": self.output_smiles_list})
#         pair_df.to_csv(self.processed_pair_filepath, index=None)

#         description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
#         description_df.to_csv(self.processed_description_filepath, index=None)

#     def __getitem__(self, idx):
#         input_smiles = self.input_smiles_list[idx]
#         output_smiles = self.output_smiles_list[idx]
#         description = self.description_list[idx]
#         return input_smiles, output_smiles, description

#     def __len__(self):
#         return len(self.input_smiles_list)
    

# class MolPair_DockingSmiles_BindingAffinity_Test(Dataset):
#     def __init__(self, root, template_path, target_name="COX2", mode="main", version="v1"):
#         self.template_path = template_path
#         if mode not in RETRIEVAL_MODE:
#             raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
#         self.mode = mode
#         if version not in VERSION:
#             raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
#         if version in ["v1", "v2"]:
#             self.root = osp.join(root, "valued_"+version)
#         elif version in ["v3", "v4"]:
#             self.root = osp.join(root, "tagged_"+version)
#         self.version = version
#         self.target_name = target_name

#         self.raw_filepath = osp.join(self.root, "raw", f"{target_name}.csv")

#         self.processed_filepath = osp.join(self.root, self.mode, f"{target_name}_input.csv")
#         if not osp.exists(osp.join(self.root, self.mode)):
#             os.makedirs(osp.join(self.root, self.mode))
        
#         if osp.exists(self.processed_filepath):
#             processed_df = pd.read_csv(self.processed_filepath)
#             self.input_smiles_list = processed_df["smiles"].tolist()
#             self.description_list = processed_df["description"].tolist()
#         else:
#             self.process()

#     def process(self):
#         self.input_smiles_list, self.description_list = [], []
        
#         print("Processing raw file")
#         raw_df = pd.read_csv(self.raw_filepath)
        
#         print("Processing templates")
#         with open(self.template_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#         self.template_list = [line.strip() for line in lines]

#         print("Processing pairs")
#         if self.mode == "random":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 template = random.choice(self.template_list)
#                 task_description = re.sub(r'\${input}', 'the above molecule', template)
#                 if self.version == "v1":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
#                 # elif self.version == "v2":
#                 #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
#                 elif self.version == "v3":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
#                     task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
#                     for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                         prop_level = row[ref_prop+"_level"]
#                         task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                 self.description_list.append(row["description"] + " " + task_description)
#                 self.input_smiles_list.append(row["smiles"])
#         elif self.mode == "iterative":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 for template in self.template_list:
#                     task_description = re.sub(r'\${input}', 'the above molecule', template)
#                     if self.version == "v1":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[self.target_name], task_description)
#                     # elif self.version == "v2":
#                     #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[self.target_name], task_description)
#                     elif self.version == "v3":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[self.target_name], task_description)
#                         task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
#                         for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                             prop_level = row[ref_prop+"_level"]
#                             task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                     self.description_list.append(row["description"] + " " + task_description)
#                     self.input_smiles_list.append(row["smiles"])
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")
        
#         description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
#         description_df.to_csv(self.processed_filepath, index=None)

#     def __getitem__(self, idx):
#         input_smiles = self.input_smiles_list[idx]
#         description = self.description_list[idx]
#         return input_smiles, description

#     def __len__(self):
#         return len(self.input_smiles_list)
    

# class MolPair_DockingSmiles_pIC50(Dataset):
#     def __init__(self, root, template_path, mode="main", target_name="2QBR", max_num_pairs_per_task=100, version="v1"):
#         self.template_path = template_path
#         if mode not in DESCRIPTION_MODE:
#             raise ValueError(f"Invalid mode: {mode}, choose from {DESCRIPTION_MODE}")
#         self.mode = mode
#         if version not in VERSION:
#             raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
#         if version in ["v1", "v2"]:
#             self.root = osp.join(root, "valued_"+version)
#         elif version in ["v3", "v4"]:
#             self.root = osp.join(root, "tagged_"+version)
#         self.version = version
#         self.target_name = target_name
#         self.max_num_pairs_per_task = max_num_pairs_per_task

#         self.PubchemEdit_filepath = osp.join(self.root, "raw", "description", "PubChemEdit.json")
#         self.target_normal_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_normal_description.csv")
#         self.target_docking_description_filepath = osp.join(self.root, "raw", "description", f"{target_name}_docking_description.csv")
        
#         self.pair_dir = osp.join(self.root, "raw", "pair")
#         self.other_pair_filepath = osp.join(self.root, "raw", "pair", f"{target_name}_other.csv")

#         self.processed_description_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_description.csv")
#         self.processed_pair_filepath = osp.join(self.root, f"{target_name}_"+self.mode, "processed_pair.csv")
#         if not osp.exists(osp.join(self.root, f"{target_name}_"+self.mode)):
#             os.makedirs(osp.join(self.root, f"{target_name}_"+self.mode))
        
#         if osp.exists(self.processed_description_filepath) and osp.exists(self.processed_pair_filepath):
#             pair_df = pd.read_csv(self.processed_pair_filepath)
#             description_df = pd.read_csv(self.processed_description_filepath)
#             self.input_smiles_list = pair_df["smiles1"].tolist()
#             self.output_smiles_list = pair_df["smiles2"].tolist()
#             self.description_list = description_df["description"].tolist()
#         else:
#             self.process()

#     def process(self):
#         self.input_smiles_list, self.output_smiles_list, self.description_list = [], [], []
        
#         normal_description_dict, docking_description_dict = {}, {}
#         print("Processing PubChemEdit")
#         with open(self.PubchemEdit_filepath) as file:
#             PubChemEdit_data = json.load(file)
#         file.close()
#         for item in tqdm(PubChemEdit_data):
#             if self.mode == "full":
#                 raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label not in ["Pharmacology/Biochemistry", "Others"]]
#             elif self.mode == "main":
#                 raw_descriptions = [desc for label, desc in item["Description"].items() if desc != "" and label in ["MolecularStructure/Classification", "FunctionalGroups", "PhysicalProperty/ChemicalProperty", "CalculatedProperties"]]
#             else:
#                 raise ValueError(f"Invalid mode: {self.mode}")
#             normal_description_dict[item["RDKit_IsoSmiles"]] = " ".join(raw_descriptions)

#         print("Processing additional normal descriptions")
#         additional_normal_description_df = pd.read_csv(self.target_normal_description_filepath)
#         for idx, row in tqdm(additional_normal_description_df.iterrows(), total=len(additional_normal_description_df)):
#             normal_description_dict[row["smiles"]] = row["description"]

#         print("Processing additional docking descriptions")
#         docking_description_df = pd.read_csv(self.target_docking_description_filepath)
#         for idx, row in tqdm(docking_description_df.iterrows(), total=len(docking_description_df)):
#             docking_description_dict[row["smiles"]] = row["description"]
        
#         print("Processing templates")
#         with open(self.template_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#         self.template_list = [line.strip() for line in lines]

#         print("Processing active2active pairs")
#         for pair_file in tqdm(os.listdir(self.pair_dir)):
#             if pair_file == f"{self.target_name}_other.csv":
#                 continue
#             match = re.search(rf'{self.target_name}_task_(\d+)_active2active\.csv', pair_file)
#             if match:
#                 task_id = match.group(1)
#             else:
#                 raise ValueError(f"Invalid pair file: {pair_file}")

#             pair_df = pd.read_csv(os.path.join(self.pair_dir, pair_file))

#             if len(pair_df) > self.max_num_pairs_per_task:
#                 pair_df = pair_df.sample(n=self.max_num_pairs_per_task)
#             for idx, row in pair_df.iterrows():
#                 template = random.choice(self.template_list)
#                 task_description = re.sub(r'\${input}', 'the above molecule', template)
#                 if self.version == "v1":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_active2active"], task_description)
#                     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V1[task_id], task_description)
#                 # elif self.version == "v2":
#                 #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_active2active"], task_description)
#                 #     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V2[task_id], task_description)
#                 elif self.version == "v3":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_active2active"], task_description)
#                     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V3[task_id], task_description)
#                     for key, value in row.items():
#                         match = re.search(r'smiles1_level(\d+)', key)
#                         if match:
#                             input_prop_num = match.group(1)
#                             task_description = re.sub(r'\${input_level'+input_prop_num+'}', value, task_description)
#                         match = re.search(r'smiles2_level(\d+)', key)
#                         if match:
#                             input_prop_num = match.group(1)
#                             task_description = re.sub(r'\${output_level'+input_prop_num+'}', value, task_description)
                
#                 self.description_list.append(normal_description_dict[row["smiles1"]] + " " + docking_description_dict[row["smiles1"]] + " " + task_description)
#                 self.input_smiles_list.append(row["smiles1"])
#                 self.output_smiles_list.append(row["smiles2"])

#         print("Processing other pairs")
#         other_pair_df = pd.read_csv(self.other_pair_filepath)
#         for idx, row in other_pair_df.iterrows():
#             template = random.choice(self.template_list)
#             task_description = re.sub(r'\${input}', 'the above molecule', template)
#             if self.version == "v1":
#                 task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_other"], task_description)
#                 if row["task"] == "active":
#                     task_description = re.sub(r'\${task_requirement}', "to at least 6.5", task_description)
#                 elif row["task"] == "increase":
#                     task_description = re.sub(r'\${task_requirement}', "by at least 1", task_description)
#                     # task_description = re.sub(r'\${task_requirement}', "to at least 5.5", task_description)
#             # elif self.version == "v2":
#             #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_other"], task_description)
#             elif self.version == "v3":
#                 task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_other"], task_description)
#                     # task_description = re.sub(r'\${task_requirement}', "to at least 5.5", task_description)
#                 for key, value in row.items():
#                     match = re.search(r'smiles1_level(\d+)', key)
#                     if match:
#                         input_prop_num = match.group(1)
#                         task_description = re.sub(r'\${input_level'+input_prop_num+'}', value, task_description)
#                     match = re.search(r'smiles2_level(\d+)', key)
#                     if match:
#                         input_prop_num = match.group(1)
#                         if row["task"] == "active":
#                             task_description = re.sub(r'\${output_level'+input_prop_num+'}', "at least slightly high pIC50 with certain activity", task_description)
#                         elif row["task"] == "increase":
#                             task_description = re.sub(r'\${output_level'+input_prop_num+'}', "another level", task_description)
#                             # task_description = re.sub(r'\${output_level'+input_prop_num+'}', "at least moderate pIC50 with activity uncertain but leaning toward inactive", task_description)

#             self.description_list.append(normal_description_dict[row["smiles1"]] + " " + docking_description_dict[row["smiles1"]] + " " + task_description)
#             self.input_smiles_list.append(row["smiles1"])
#             self.output_smiles_list.append(row["smiles2"])

#         pair_df = pd.DataFrame({"smiles1": self.input_smiles_list, "smiles2": self.output_smiles_list})
#         pair_df.to_csv(self.processed_pair_filepath, index=None)

#         description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
#         description_df.to_csv(self.processed_description_filepath, index=None)

#     def __getitem__(self, idx):
#         input_smiles = self.input_smiles_list[idx]
#         output_smiles = self.output_smiles_list[idx]
#         description = self.description_list[idx]
#         return input_smiles, output_smiles, description

#     def __len__(self):
#         return len(self.input_smiles_list)
    

# class MolPair_DockingSmiles_pIC50_active2active_Test(Dataset):
#     def __init__(self, root, template_path, task_id, target_name="2QBR", mode="main", version="v1"):
#         self.template_path = template_path
#         if mode not in RETRIEVAL_MODE:
#             raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
#         self.mode = mode
#         if version not in VERSION:
#             raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
#         if version in ["v1", "v2"]:
#             self.root = osp.join(root, "valued_"+version)
#         elif version in ["v3", "v4"]:
#             self.root = osp.join(root, "tagged_"+version)
#         self.version = version
#         self.target_name = target_name
#         self.task_id = str(task_id)

#         self.raw_filepath = osp.join(self.root, "raw", f"{target_name}_task_{task_id}_active2active.csv")

#         self.processed_filepath = osp.join(self.root, self.mode, f"{target_name}_task_{task_id}_active2active_input.csv")
#         if not osp.exists(osp.join(self.root, self.mode)):
#             os.makedirs(osp.join(self.root, self.mode))
        
#         if osp.exists(self.processed_filepath):
#             processed_df = pd.read_csv(self.processed_filepath)
#             self.input_smiles_list = processed_df["smiles"].tolist()
#             self.description_list = processed_df["description"].tolist()
#         else:
#             self.process()

#     def process(self):
#         self.input_smiles_list, self.description_list = [], []
        
#         print("Processing raw file")
#         raw_df = pd.read_csv(self.raw_filepath)
        
#         print("Processing templates")
#         with open(self.template_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#         self.template_list = [line.strip() for line in lines]

#         print("Processing pairs")
#         if self.mode == "random":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 template = random.choice(self.template_list)
#                 task_description = re.sub(r'\${input}', 'the above molecule', template)
#                 if self.version == "v1":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_active2active"], task_description)
#                     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V1[self.task_id], task_description)
#                 # elif self.version == "v2":
#                 #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_active2active"], task_description)
#                 elif self.version == "v3":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_active2active"], task_description)
#                     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V3[self.task_id], task_description)
#                     task_description = re.sub(r'\${output_level(\d+)}', "another level", task_description)
#                     for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                         prop_level = row[ref_prop+"_level"]
#                         task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                 self.description_list.append(row["description"] + " " + task_description)
#                 self.input_smiles_list.append(row["smiles"])
#         elif self.mode == "iterative":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 for template in self.template_list:
#                     task_description = re.sub(r'\${input}', 'the above molecule', template)
#                     if self.version == "v1":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_other"], task_description)
#                         task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V1[self.task_id], task_description)
#                     # elif self.version == "v2":
#                     #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_other"], task_description)
#                     #     task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V2[self.task_id], task_description)
#                     elif self.version == "v3":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_other"], task_description)
#                         task_description = re.sub(r'\${task_requirement}', TEXT_REQUIREMENTS_V3[self.task_id], task_description)
#                         task_description = re.sub(r'\${output_level(\d+)}', "another_level", task_description)
#                         for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                             prop_level = row[ref_prop+"_level"]
#                             task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                     self.description_list.append(row["description"] + " " + task_description)
#                     self.input_smiles_list.append(row["smiles"])
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")
        
#         description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
#         description_df.to_csv(self.processed_filepath, index=None)

#     def __getitem__(self, idx):
#         input_smiles = self.input_smiles_list[idx]
#         description = self.description_list[idx]
#         return input_smiles, description

#     def __len__(self):
#         return len(self.input_smiles_list)
    
    
# class MolPair_DockingSmiles_pIC50_other_Test(Dataset):
#     def __init__(self, root, template_path, target_name="2QBR", mode="main", version="v1"):
#         self.template_path = template_path
#         if mode not in RETRIEVAL_MODE:
#             raise ValueError(f"Invalid mode: {mode}, choose from {RETRIEVAL_MODE}")
#         self.mode = mode
#         if version not in VERSION:
#             raise ValueError(f"Invalid version: {version}, choose from {VERSION}")
#         if version in ["v1", "v2"]:
#             self.root = osp.join(root, "valued_"+version)
#         elif version in ["v3", "v4"]:
#             self.root = osp.join(root, "tagged_"+version)
#         self.version = version
#         self.target_name = target_name

#         self.raw_filepath = osp.join(self.root, "raw", f"{target_name}_other.csv")

#         self.processed_filepath = osp.join(self.root, self.mode, f"{target_name}_other_input.csv")
#         if not osp.exists(osp.join(self.root, self.mode)):
#             os.makedirs(osp.join(self.root, self.mode))
        
#         if osp.exists(self.processed_filepath):
#             processed_df = pd.read_csv(self.processed_filepath)
#             self.input_smiles_list = processed_df["smiles"].tolist()
#             self.description_list = processed_df["description"].tolist()
#         else:
#             self.process()

#     def process(self):
#         self.input_smiles_list, self.description_list = [], []
        
#         print("Processing raw file")
#         raw_df = pd.read_csv(self.raw_filepath)
        
#         print("Processing templates")
#         with open(self.template_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#         self.template_list = [line.strip() for line in lines]

#         print("Processing pairs")
#         if self.mode == "random":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 template = random.choice(self.template_list)
#                 task_description = re.sub(r'\${input}', 'the above molecule', template)
#                 if self.version == "v1":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_other"], task_description)
#                     task_description = re.sub(r'\${task_requirement}', "to at least 6.5", task_description)
#                 # elif self.version == "v2":
#                 #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_other"], task_description)
#                 elif self.version == "v3":
#                     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_other"], task_description)
#                     task_description = re.sub(r'\${output_level(\d+)}', "at least slightly high pIC50 with certain activity", task_description)
#                     for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                         prop_level = row[ref_prop+"_level"]
#                         task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                 self.description_list.append(row["description"] + " " + task_description)
#                 self.input_smiles_list.append(row["smiles"])
#         elif self.mode == "iterative":
#             for idx, row in tqdm(raw_df.iterrows()):
#                 for template in self.template_list:
#                     task_description = re.sub(r'\${input}', 'the above molecule', template)
#                     if self.version == "v1":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V1[f"{self.target_name}_other"], task_description)
#                         task_description = re.sub(r'\${task_requirement}', "to at least 6.5", task_description)
#                     # elif self.version == "v2":
#                     #     task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V2[f"{self.target_name}_other"], task_description)
#                     elif self.version == "v3":
#                         task_description = re.sub(r'\${requirement}', TEXT_REQUIREMENTS_V3[f"{self.target_name}_other"], task_description)
#                         task_description = re.sub(r'\${output_level(\d+)}', "at least slightly high pIC50 with certain activity", task_description)
#                         for name_key, ref_prop in TASK_REFERENCE[self.target_name].items():
#                             prop_level = row[ref_prop+"_level"]
#                             task_description = re.sub(r'\${'+name_key+'}', prop_level, task_description)
#                     self.description_list.append(row["description"] + " " + task_description)
#                     self.input_smiles_list.append(row["smiles"])
#         else:
#             raise ValueError(f"Invalid mode: {self.mode}")
        
#         description_df = pd.DataFrame({"smiles": self.input_smiles_list, "description": self.description_list})
#         description_df.to_csv(self.processed_filepath, index=None)

#     def __getitem__(self, idx):
#         input_smiles = self.input_smiles_list[idx]
#         description = self.description_list[idx]
#         return input_smiles, description

#     def __len__(self):
#         return len(self.input_smiles_list)