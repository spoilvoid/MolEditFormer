import os
import json
import pandas as pd
import numpy as np
import random
import argparse
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool

from tqdm import tqdm

import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


TASK_THRESHOLD = {
    101: {"logP": 0.84},
    102: {"logP": 4.34},
    103: {"QED": 0.9},
    104: {"QED": 0.6},
    105: {"TPSA": 32.92},
    106: {"TPSA": 89.53},
    107: {"HBA": 6},
    108: {"HBD": 2},
    201: {"logP": 1.65, "HBA": 6},
    202: {"logP": 4.14, "HBA": 5},
    203: {"logP": 1.55, "HBD": 2},
    204: {"logP": 4.14, "HBD": 2},
    205: {"logP": 1.89, "TPSA": 33.71},
    206: {"logP": 1.65, "TPSA": 91.48},
}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="data/MolPair/mol_single/version_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--store_dir", type=str, default="data/MolPair/extra_set")
    parser.add_argument("--version", type=int, default=3)
    args = parser.parse_args()


    random.seed(args.seed)


    with open(os.path.join(args.file_dir, "raw", "smiles_descriptions.json")) as file:
        description_dict = json.load(file)
    file.close()
    smiles_list = list(description_dict.keys())


    df = pd.read_csv(os.path.join(args.file_dir, "raw", "raw_data.csv"))


    def process(input_smi):
        smi_row = df[df['smiles'] == input_smi]
        if not smi_row.empty:
            smi_dict = smi_row.to_dict(orient='records')[0]
            return {"logP": smi_dict["logP"], "QED": smi_dict["QED"], "TPSA": smi_dict["TPSA"], "HBA": smi_dict["HBA"], "HBD": smi_dict["HBD"]}
        else:
            mol = Chem.MolFromSmiles(input_smi)
            if mol is None:
                return None
            return {"logP": Descriptors.MolLogP(mol), "QED": QED.qed(mol), "TPSA": Descriptors.TPSA(mol), "HBA": Descriptors.NumHAcceptors(mol), "HBD": Descriptors.NumHDonors(mol)}
        
    with Pool(args.num_workers) as p:
        prop_dict_list = list(tqdm(p.imap(process, smiles_list), total=len(smiles_list)))


    def check_threshold(prop_dict, task):
        if task in TASK_THRESHOLD:
            flag = True
            for prop_name, threshold in TASK_THRESHOLD[task].items():
                if task in [101, 201, 203, 205, 206] and prop_name == "logP":
                    if prop_dict[prop_name] > threshold:
                        flag = False
                        break
                if task in [102, 202, 204, 205, 206] and prop_name == "logP":
                    if prop_dict[prop_name] < threshold:
                        flag = False
                        break
                if task == 103 and prop_name == "QED":
                    if prop_dict[prop_name] < threshold:
                        flag = False
                        break
                if task == 104 and prop_name == "QED":
                    if prop_dict[prop_name] > threshold:
                        flag = False
                        break
                if task in [105, 205] and prop_name == "TPSA":
                    if prop_dict[prop_name] > threshold:
                        flag = False
                        break
                if task in [106, 206] and prop_name == "TPSA":
                    if prop_dict[prop_name] < threshold:
                        flag = False
                        break
                if task in [107, 201, 202] and prop_name == "HBA":
                    if prop_dict[prop_name] < threshold:
                        flag = False
                        break
                if task in [108, 203, 204] and prop_name == "HBD":
                    if prop_dict[prop_name] < threshold:
                        flag = False
                        break
            return flag
        else:
            return False
    

    aggregate_description_dict = {}
    aggregate_df = pd.DataFrame()
    for task in list(range(101, 109)) + list(range(201, 207)):
        process_task = partial(check_threshold, task=task)
        with Pool(args.num_workers) as p:
            bool_results = list(tqdm(p.imap(process_task, prop_dict_list), total=len(prop_dict_list)))
        task_smiles_list = [smi for flag, smi in zip(bool_results, smiles_list) if flag]
        task_description_dict = {smi: description_dict[smi] for smi in task_smiles_list}
        task_df = df[df['smiles'].isin(task_smiles_list)]

        aggregate_description_dict.update(task_description_dict)
        aggregate_df = pd.concat([aggregate_df, task_df])

        store_dir = os.path.join(args.store_dir, f"version_{args.version}")
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        if not os.path.exists(os.path.join(store_dir, f"task_{task}")):
            os.makedirs(os.path.join(store_dir, f"task_{task}"))
        with open(os.path.join(store_dir, f"task_{task}", "smiles_descriptions.json"), 'w') as file:
            json.dump(task_description_dict, file, indent=4)
        file.close()
        task_df.to_csv(os.path.join(store_dir, f"task_{task}", "raw_data.csv"), index=False)

    if not os.path.exists(os.path.join(store_dir, "raw")):
        os.makedirs(os.path.join(store_dir, "raw"))
    if not os.path.exists(os.path.join(store_dir, "processed")):
        os.makedirs(os.path.join(store_dir, "processed"))
    with open(os.path.join(store_dir, "raw", "smiles_descriptions.json"), 'w') as file:
        json.dump(aggregate_description_dict, file, indent=4)
    file.close()
    aggregate_df.to_csv(os.path.join(store_dir, "raw", "raw_data.csv"), index=False)