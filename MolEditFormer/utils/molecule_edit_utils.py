import os
import os.path as osp
import numpy as np
import copy
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from MolEditFormer.models import MLP
from MolEditFormer.utils import PlogP


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]

prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func



DESCRIPTION_DICT = {
    101: "This molecule is soluble in water.",
    102: "This molecule is insoluble in water.",
    103: "This molecule is like a drug.",
    104: "This molecule is not like a drug.",
    105: "This molecule has high permeability.",
    106: "This molecule has low permeability.",
    107: "This molecule has more hydrogen bond acceptors.",
    108: "This molecule has more hydrogen bond donors.",

    # 109: "This molecule has high bioavailability.",
    # 110: "This molecule has low toxicity.",
    # 111: "This molecule is metabolically stable.",
    
    201: "This molecule is soluble in water and has more hydrogen bond acceptors.",
    202: "This molecule is insoluble in water and has more hydrogen bond acceptors.",
    203: "This molecule is soluble in water and has more hydrogen bond donors.",
    204: "This molecule is insoluble in water and has more hydrogen bond donors.",
    205: "This molecule is soluble in water and has high permeability.",
    206: "This molecule is soluble in water and has low permeability.",

    # 301: "This molecule looks like Penicillin.",
    # 302: "This molecule looks like Aspirin.",
    # 303: "This molecule looks like Caffeine.",
    # 304: "This molecule looks like Cholesterol.",
    # 305: "This molecule looks like Dopamine.",
    # 306: "This molecule looks like Cysteine.",
    # 307: "This molecule looks like Glutathione.",
    
    # 401: "This molecule is tested positive in an assay that are inhibitors and substrates of an enzyme protein. It uses molecular oxygen inserting one oxygen atom into a substrate, and reducing the second into a water molecule.",
    # 402: "This molecule is tested positive in an assay for Anthrax Lethal, which acts as a protease that cleaves the N-terminal of most dual specificity mitogen-activated protein kinase kinases.",
    # 403: "This molecule is tested positive in an assay for Activators of ClpP, which cleaves peptides in various proteins in a process that requires ATP hydrolysis and has a limited peptidase activity in the absence of ATP-binding subunits.",
    # 404: "This molecule is tested positive in an assay for activators involved in the transport of proteins between the endosomes and the trans Golgi network.",
    # 405: "This molecule is an inhibitor of a protein that prevents the establishment of the cellular antiviral state by inhibiting ubiquitination that triggers antiviral transduction signal and inhibits post-transcriptional processing of cellular pre-mRNA.",
    # 406: "This molecule is tested positive in the high throughput screening assay to identify inhibitors of the SARS coronavirus 3C-like Protease, which cleaves the C-terminus of replicase polyprotein at 11 sites.",

    # 501: "This molecule is more soluble in water than in oil.",
    # 502: "This molecule is less soluble in water than in oil.",
    # 503: "This molecule has higher logP.",
    # 504: "This molecule has lower logP.",
    # 505: "This molecule has lower TPSA.",
    # 506: "This molecule has higher TPSA.",
    # 507: "This molecule has less hydrogen bond acceptors.",
    # 508: "This molecule has less hydrogen bond donors.",
    # 509: "This molecule has high molecular weight.",
    # 510: "This molecule has low molecular weight.",

    # 601: "This molecule has lower logP.",
    # 602: "This molecule has higher logP.",
    # 603: "This molecule has higher QED.",
    # 604: "This molecule has lower QED.",
    # 605: "This molecule has lower TPSA.",
    # 606: "This molecule has higher TPSA.",
    # 607: "This molecule has more HBA.",
    # 608: "This molecule has more HBD.",

    # 601: "This molecule has lower logP and more HBA.",
    # 602: "This molecule has higher logP and more HBA.",
    # 603: "This molecule has lower logP and more HBD.",
    # 604: "This molecule has higher logP and more HBD.",
    # 605: "This molecule has lower logP and lower TPSA.",
    # 606: "This molecule has lower logP and higher TPSA.",
}


HARD_THRESHOLD_DICT = {
    101: [0.5],
    102: [0.5],
    103: [0.1],
    104: [0.1],
    105: [10],
    106: [10],
    107: [1],
    108: [1],

    201: [0.5, 1],
    202: [0.5, 1],
    203: [0.5, 1],
    204: [0.5, 1],
    205: [0.5, 10],
    206: [0.5, 10],
}


# https://pubchem.ncbi.nlm.nih.gov/compound/5904
# Penicillin_SMILES = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
Penicillin_SMILES = "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/2244
# Aspirin_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
Aspirin_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/2519
# Caffeine_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
Caffeine_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# https://pubchem.ncbi.nlm.nih.gov/compound/5997
# Cholesterol_SMILES = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
Cholesterol_SMILES = "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"

# https://pubchem.ncbi.nlm.nih.gov/compound/681
# Dopamine_SMILES = "C1=CC(=C(C=C1CCN)O)O"
Dopamine_SMILES = "NCCc1ccc(O)c(O)c1"

# https://pubchem.ncbi.nlm.nih.gov/compound/5862
# Cysteine_SMILES = "C(C(C(=O)O)N)S"
Cysteine_SMILES = "NC(CS)C(=O)O"

# https://pubchem.ncbi.nlm.nih.gov/compound/124886
# Glutathione_SMILES = "C(CC(=O)NC(CS)C(=O)NCC(=O)O)C(C(=O)O)N"
Glutathione_SMILES = "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O"


# def get_edit_SMILES_list(args):
#     SMILES_list = []
#     if args.test:
#         if args.edit_SMILES is not None:
#             SMILES_list.append(args.edit_SMILES)
#     else:
#         f = open(args.edit_SMILES_filepath, 'r')
#         lines = f.readlines()
#         for line in lines:
#             SMILES = line.strip()
#             if len(SMILES) > 0:
#                 SMILES_list.append(SMILES)
#     return SMILES_list


# def get_edit_prompt(args):
#     if args.test:
#         if args.edit_prompt is not None:
#             return args.input_description
#     else:
#         if args.edit_task_id not in DESCRIPTION_DICT.keys():
#             raise ValueError
#         else:
#             print("Use {} descrition.".format(args.edit_task_id))
#             return DESCRIPTION_DICT[args.edit_task_id]


# def load_CLIP_molecule_branch(args):
#     if args.molecule_type == "2DGraph" or args.molecule_type == "all":
#         print(f"Loading 2DGraph model from {args.graph_model_path}")
#         molecule_dim = args.gnn_emb_dim
#         molecule_node_model = GNN(
#             num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
#             JK=args.JK, drop_ratio=args.dropout_ratio,
#             gnn_type=args.gnn_type)
#         molecule_model = GNN_graphpred(
#             num_layer=args.num_layer,
#             emb_dim=args.gnn_emb_dim,
#             JK=args.JK,
#             graph_pooling=args.graph_pooling,
#             num_tasks=1,
#             molecule_node_model=molecule_node_model)
        
#     if args.molecule_type == "3DGraph" or args.molecule_type == "all":
#         pass
#     if args.molecule_type == "SMILES" or args.molecule_type == "all":
#         pass
#     state_dict = torch.load(args.mol_model_path, map_location='cpu')
#     molecule_model.load_state_dict(state_dict)

#     print(f"Loading molecule projector from {args.mol_projector_path}")
#     mol2latent = nn.Linear(molecule_dim, args.SSL_emb_dim)
#     state_dict = torch.load(args.mol_projector_path, map_location='cpu')
#     mol2latent.load_state_dict(state_dict)
    
#     return molecule_model, mol2latent


# def load_CLIP_text_branch(args):
#     text_dim = args.text_emb_dim
#     text_tokenizer = AutoTokenizer.from_pretrained(args.text_pretrain_dir)
#     text_model = AutoModel.from_pretrained(args.text_pretrain_dir)

#     print(f"Loading text model from {args.text_model_path}")
#     state_dict = torch.load(args.text_model_path, map_location='cpu')
#     text_model.load_state_dict(state_dict)

#     print(f"Loading text projector from {args.text_projector_path}")
#     text2latent = nn.Linear(text_dim, args.SSL_emb_dim)
#     state_dict = torch.load(args.text_projector_path, map_location='cpu')
#     text2latent.load_state_dict(state_dict)
    
#     return (text_model, text_tokenizer), text2latent


# def load_space_projector(args):
#     gen2joint_projector = MLP(args.gen_emb_dim, [args.SSL_emb_dim, args.SSL_emb_dim])
#     joint2gen_projector = MLP(args.SSL_emb_dim, [args.gen_emb_dim, args.gen_emb_dim])

#     if args.resume:
#         if osp.exists(args.gen2joint_projector_path):
#             print(f"Loading gen2joint_space_projector from {args.gen2joint_projector_path}")
#             state_dict = torch.load(args.gen2joint_projector_path, map_location='cpu')
#             gen2joint_projector.load_state_dict(state_dict)
#         else:
#             print(f"{args.gen2joint_projector_path} does not exist, random initialization.")
            
#         if osp.exists(args.joint2gen_projector_path):
#             print(f"Loading joint2gen_space_projector from {args.joint2gen_projector_path}")
#             state_dict = torch.load(args.joint2gen_projector_path, map_location='cpu')
#             joint2gen_projector.load_state_dict(state_dict)
#         else:
#             print(f"{args.joint2gen_projector_path} does not exist, random initialization.")
    
#     return gen2joint_projector, joint2gen_projector


def get_can_smiles(smi):
    """
    Canonicalize a SMILES without atom mapping
    """
    if isinstance(smi, float):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None
    

def get_molecule_similarity(mol_a, mol_b):
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    return sim


def get_murcko_scaffold(mol):
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
        return scaffold_smiles
    except:
        return ""


def get_BRICS_scaffold(mol):
    try:
        fragment_list = BRICS.BRICSDecompose(mol, returnMols=True)
        max_carbon_count, max_atom_count = 0, 0
        for frag_mol in fragment_list:
            atoms = frag_mol.GetAtoms()
            atom_count = len(atoms)

            carbon_count = 0
            for atom in atoms:
                if atom.GetSymbol() == 'C':
                    carbon_count += 1
                if atom.GetIsotope() != 0:
                    atom.SetIsotope(0)
            
            if atom_count > max_atom_count or (atom_count == max_atom_count and carbon_count > max_carbon_count):
                max_carbon_count, max_atom_count = carbon_count, atom_count
                max_frag_mol = frag_mol
        max_frag_smiles = Chem.MolToSmiles(max_frag_mol)
        return max_frag_smiles
    except:
        return ""


def dynamic_similarity_threshold(avg_mw, alpha: float = 0.7, upper_threshold: float = 0.6, lower_threshold: float = 0.2, mw_threshold: int = 250, power: float = 0.5, smooth: bool = False):
    if smooth:
        return upper_threshold * (1 - np.exp(-alpha * avg_mw))
    elif avg_mw >= mw_threshold:
        return upper_threshold
    else:
        return lower_threshold + (upper_threshold - lower_threshold) * ((avg_mw / mw_threshold) ** power)


def check_main_substructure(mol1, mol2, threshold: float = 0.5):
    mw1, mw2 = Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)
    atom_num1, atom_num2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()

    murcko_scaffold_smi, murcko_scaffold_smi2 = get_murcko_scaffold(mol1), get_murcko_scaffold(mol2)
    murcko_scaffold_mol1, murcko_scaffold_mol2 = Chem.MolFromSmiles(murcko_scaffold_smi), Chem.MolFromSmiles(murcko_scaffold_smi2)
    murcko_scaffold_mw1, murcko_scaffold_mw2 = Descriptors.MolWt(murcko_scaffold_mol1), Descriptors.MolWt(murcko_scaffold_mol2)
    murcko_scaffold_atom_num1, murcko_scaffold_atom_num2 = murcko_scaffold_mol1.GetNumAtoms(), murcko_scaffold_mol2.GetNumAtoms()
    murcko_scaffold_result = (murcko_scaffold_smi == murcko_scaffold_smi2) and (murcko_scaffold_mw1/float(mw1)>=threshold or murcko_scaffold_atom_num1/float(atom_num1)>=threshold) and (murcko_scaffold_mw2/float(mw2)>=threshold or murcko_scaffold_atom_num2/float(atom_num2)>=threshold)

    BRICS_scaffold_smi, BRICS_scaffold_smi2 = get_BRICS_scaffold(mol1), get_BRICS_scaffold(mol2)
    BRICS_scaffold_mol1, BRICS_scaffold_mol2 = Chem.MolFromSmiles(BRICS_scaffold_smi), Chem.MolFromSmiles(BRICS_scaffold_smi2)
    BRICS_scaffold_mw1, BRICS_scaffold_mw2 = Descriptors.MolWt(BRICS_scaffold_mol1), Descriptors.MolWt(BRICS_scaffold_mol2)
    BRICS_scaffold_atom_num1, BRICS_scaffold_atom_num2 = BRICS_scaffold_mol1.GetNumAtoms(), BRICS_scaffold_mol2.GetNumAtoms()
    BRICS_scaffold_result = (BRICS_scaffold_smi == BRICS_scaffold_smi2) and (BRICS_scaffold_mw1/float(mw1)>=threshold or BRICS_scaffold_atom_num1/float(atom_num1)>=threshold) and (BRICS_scaffold_mw2/float(mw2)>=threshold or BRICS_scaffold_atom_num2/float(atom_num2)>=threshold)

    return murcko_scaffold_result or BRICS_scaffold_result


def evaluate_SMILES_list(SMILES_list, description):
    print("SMILES_list:", SMILES_list)
    mol_list = []
    for SMILES in SMILES_list:
        mol = Chem.MolFromSmiles(SMILES)
        # Chem.SanitizeMol(mol)
        # print(SMILES, mol)
        if mol is None:
            continue
        mol_list.append(mol)
    print("valid mol list:", len(mol_list))

    if len(mol_list) < 3:
        return [False]

    if ("soluble" in description and "insoluble" not in description) or "more soluble in water than in oil" in description or "higher logP" in description:
        props = ["MolLogP"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif "insoluble" in description or "less soluble in water than in oil" in description  or "lower logP" in description:
        props = ["MolLogP"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule is more like a drug.", "This molecule is like a drug."]:
        props = ["qed"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule is less like a drug.", "This molecule is not like a drug."]:
        props = ["qed"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has higher permeability.", "This molecule has high permeability.", "This molecule has lower TPSA."]:
        props = ["TPSA"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has lower permeability.", "This molecule has low permeability.", "This molecule has higher TPSA."]:
        props = ["TPSA"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has higher molecular weight.", "This molecule has high molecular weight."]:
        props = ["MolWt"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has lower molecular weight.", "This molecule has low molecular weight."]:
        props = ["MolWt"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has more hydrogen bond acceptors."]:
        props = ["NumHAcceptors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]
    
    elif description in ["This molecule has less hydrogen bond acceptors."]:
        props = ["NumHAcceptors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif description in ["This molecule has more hydrogen bond donors."]:
        props = ["NumHDonors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] < value_list[2]:
            answer = [True]
        else:
            answer = [False]
    
    elif description in ["This molecule has less hydrogen bond donors."]:
        props = ["NumHDonors"]
        prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]
        value_list = []
        for name, func in prop_pred:
            for SMILES, mol in zip(SMILES_list, mol_list):
                value = func(mol)
                value_list.append(value)
                print("{} & {:.5f}".format(SMILES, value))
        if value_list[0] > value_list[2]:
            answer = [True]
        else:
            answer = [False]

    elif "penicillin" in description or "Penicillin" in description:
        target_mol = Chem.MolFromSmiles(Penicillin_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between penicillin and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between penicillin and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "aspirin" in description or "Aspirin" in description:
        target_mol = Chem.MolFromSmiles(Aspirin_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between aspirin and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between aspirin and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "caffeine" in description or "Caffeine" in description:
        target_mol = Chem.MolFromSmiles(Caffeine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between caffeine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between caffeine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "cholesterol" in description or "Cholesterol" in description:
        target_mol = Chem.MolFromSmiles(Cholesterol_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between cholesterol and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between cholesterol and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "dopamine" in description or "Dopamine" in description:
        target_mol = Chem.MolFromSmiles(Dopamine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between dopamine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between dopamine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity:
            answer = [True]
        else:
            answer = [False]

    elif "cysteine" in description or "Cysteine" in description:
        target_mol = Chem.MolFromSmiles(Cysteine_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between cysteine and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between cysteine and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    elif "glutathione" in description or "Glutathione" in description:
        target_mol = Chem.MolFromSmiles(Glutathione_SMILES)
        original_SMILES = SMILES_list[0]
        original_mol = mol_list[0]
        original_similarity = get_molecule_similarity(target_mol, original_mol)
        print("similarity between glutathione and original molecules\n{} & {:.5f}".format(original_SMILES, original_similarity))

        edited_SMILES = SMILES_list[2]
        edited_mol = mol_list[2]
        edited_similarity = get_molecule_similarity(target_mol, edited_mol)
        print("similarity between glutathione and edited molecules\n{} & {:.5f}".format(edited_SMILES, edited_similarity))
        if edited_similarity > original_similarity: # check original_similarity >< 0.8
            answer = [True]
        else:
            answer = [False]

    else:
        print("Not implemented.")
        answer = [False]

    return answer


def evaluate_molecular_edit_result(input_smi, output_smi, task_id, threshold_list=[0]):
    '''
    Args:
        input_smi: str # input SMILES
        output_smi: str # output SMILES
        task_id: int # pre-defined task id in DESCRIPTION_DICT above
        threshold_list: threshold_list for each sub task
    
    Returns:
        output: bool # success or not
        reason: str # fail reason
    '''
    # check smiles validation
    input_mol = Chem.MolFromSmiles(input_smi)
    output_mol = Chem.MolFromSmiles(output_smi)
    if input_mol is None and output_mol is None:
        return False, "invalid input and output"
    elif input_mol is None and output_mol is not None:
        return False, "invalid input"
    elif input_mol is not None and output_mol is None:
        return False, "invalid output"

    success_flag = True
    reason_list = []
    if task_id == 101:
        prop = "MolLogP"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop + threshold < input_prop:
            return True, "success"
        else:
            return False, "not enough low logP caused failure"
    
    elif task_id == 102:
        prop = "MolLogP"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop > input_prop + threshold:
            return True, "success"
        else:
            return False, "not enough low logP caused failure"

    elif task_id == 103:
        prop = "qed"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop > input_prop + threshold:
            return True, "success"
        else:
            return False, "not enough high QED caused failure"
    
    elif task_id == 104:
        prop = "qed"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop + threshold < input_prop:
            return True, "success"
        else:
            return False, "not enough low QED caused failure"

    elif task_id == 105:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop + threshold < input_prop:
            return True, "success"
        else:
            return False, "not enough low TPSA caused failure"
    
    elif task_id == 106:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop > input_prop + threshold:
            return True, "success"
        else:
            return False, "not enough high TPSA caused failure"

    elif task_id == 107:
        prop = "NumHAcceptors"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop > input_prop + threshold:
            return True, "success"
        else:
            return False, "not enough high HBA caused failure"

    elif task_id == 108:
        prop = "NumHDonors"
        threshold = threshold_list[0]
        input_prop = prop2func[prop](input_mol)
        output_prop = prop2func[prop](output_mol)
        if output_prop > input_prop + threshold:
            return True, "success"
        else:
            return False, "not enough high HBD caused failure"

    elif task_id == 201:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=101, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=107, threshold_list=[threshold_list[1]])
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"

    elif task_id == 202:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=102, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=107, threshold_list=[threshold_list[1]])
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"

    elif task_id == 203:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=101, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=108, threshold_list=[threshold_list[1]])
        reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"

    elif task_id == 204:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=102, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=108, threshold_list=[threshold_list[1]])
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"

    elif task_id == 205:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=101, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=105, threshold_list=[threshold_list[1]])
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"

    elif task_id == 206:
        result_01, reason_01 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=101, threshold_list=[threshold_list[0]])
        result_02, reason_02 = evaluate_molecular_edit_result(input_smi, output_smi, task_id=106, threshold_list=[threshold_list[1]])
        if result_01 and result_02:
            return True, "success"
        else:
            reason_list = [reason.split("caused failure")[0].strip() for reason in [reason_01, reason_02] if reason != "success"]
            return False, " ".join(reason_list) + " caused failure"


def evaluate_latent_optimization_result(input_smi, output_smi, task_name, sim_threshold=0.4):
    '''
    Args:
        input_smi: str # input SMILES
        output_smi: str # output SMILES
        task_name: int # pre-defined task name "QED_constrained_optimization" and "PlogP_constrained_optimization
        sim_threshold: float # similarity threshold 0.2/0.4/0.6
    
    Returns:
        output: bool # success or not
        reason: str # fail reason
    '''
    # check smiles validation
    input_mol = Chem.MolFromSmiles(input_smi)
    output_mol = Chem.MolFromSmiles(output_smi)
    if input_mol is None and output_mol is None:
        return False, "invalid input and output"
    elif input_mol is None and output_mol is not None:
        return False, "invalid input"
    elif input_mol is not None and output_mol is None:
        return False, "invalid output"

    success_flag = True
    reason_list = []
    if task_name == "QED_constrained_optimization":
        sim = get_molecule_similarity(input_mol, output_mol)
        input_prop = Descriptors.qed(input_mol)
        output_prop = Descriptors.qed(output_mol)
        if output_prop < 0.9:
            success_flag = False
            if input_prop >= output_prop:
                reason_list.append("not increase QED")
            else:
                reason_list.append("not enough high QED")
        if sim < sim_threshold:
            success_flag = False
            reason_list.append("low similarity")
    elif task_name == "PlogP_constrained_optimization":
        sim = get_molecule_similarity(input_mol, output_mol)
        input_prop = PlogP.calculateScore(input_mol)
        output_prop = PlogP.calculateScore(output_mol)
        if output_prop <= input_prop:
            success_flag = False
            reason_list.append("not increase PlogP")
        if sim < sim_threshold:
            success_flag = False
            reason_list.append("low similarity")

    if success_flag:
        return success_flag, "success"
    else:
        return success_flag, ", ".join(reason_list) + "caused failure"


def process_ligand_smiles(input_can_smi, output_dir, index=0):
    """
    Process ligand SMILES to .pdbqt file
    """
    filepath = osp.join(output_dir, f"smiles_{index}.pdbqt")

    if not osp.exists(filepath):
        try:
            os.system(f"echo '{input_can_smi}' | obabel -i smi -o pdbqt -O {filepath} --gen3d")
        except:
            return False

    return filepath


def calculate_grid_box(pdbqt_file, buffer=5.0):
    xs, ys, zs = [], [], []

    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):  # 解析原子坐标
                xs.append(float(line[30:38].strip()))
                ys.append(float(line[38:46].strip()))
                zs.append(float(line[46:54].strip()))
                print(float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip()))

    if not xs or not ys or not zs:
        raise ValueError("No atom coordinates found in the PDBQT file")

    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    center_z = (min(zs) + max(zs)) / 2

    size_x = max(xs) - min(xs) + buffer
    size_y = max(ys) - min(ys) + buffer
    size_z = max(zs) - min(zs) + buffer

    print(f"Center: x={center_x:.2f}, y={center_y:.2f}, z={center_z:.2f}")
    print(f"Size: x={size_x:.2f}, y={size_y:.2f}, z={size_z:.2f}")

    return center_x, center_y, center_z, size_x, size_y, size_z


def run_vina(receptor, ligand, center_x, center_y, center_z, size_x, size_y, size_z, output="docking_result.pdbqt", log="docking.log"):
    vina_cmd = [
        "vina",
        "--receptor", receptor,
        "--ligand", ligand,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", str(size_x),
        "--size_y", str(size_y),
        "--size_z", str(size_z),
        "--out", output,
        "--log", log
    ]
    result = subprocess.run(vina_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Vina failed:", result.stderr)
        return None
    return output


def read_log(log_filepath):
    best_affinity, best_rmsd_l_b, best_rmsd_u_b = None, None, None
    with open(log_filepath) as log:
        for line in log:
            if line.strip().startswith("1"):  # 找到模式1的行
                cols = line.split()
                if len(cols) >= 2:
                    best_affinity = float(cols[1])
                    best_rmsd_l_b = float(cols[2])
                    best_rmsd_u_b = float(cols[3])
                    break
    return best_affinity, best_rmsd_l_b, best_rmsd_u_b


def autodock_vina_pipeline(ligand_pdbqt, output_filepath, log_filepath, protein_pdbqt, grid_box_config=None):
    if grid_box_config is None:
        center_x, center_y, center_z, size_x, size_y, size_z = calculate_grid_box(protein_pdbqt)
    else:
        center_x, center_y, center_z = grid_box_config["center_x"], grid_box_config["center_y"], grid_box_config["center_z"]
        size_x, size_y, size_z = grid_box_config["size_x"], grid_box_config["size_y"], grid_box_config["size_z"]
    
    output_file = run_vina(protein_pdbqt, ligand_pdbqt, center_x, center_y, center_z, size_x, size_y, size_z, output=output_filepath, log=log_filepath)
    best_affinity, best_rmsd_l_b, best_rmsd_u_b = read_log(log_filepath)

    if output_file is None:
        return "failed", "failed", "failed"
    else:
        return best_affinity, best_rmsd_l_b, best_rmsd_u_b