import os
import os.path as osp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

from models import MegaMolBART, GNN, GNN_graphpred, MLP


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


DESCRIPTION_DICT = {
    101: "This molecule is soluble in water.",
    102: "This molecule is insoluble in water.",
    103: "This molecule is like a drug.",
    104: "This molecule is not like a drug.",
    105: "This molecule has high permeability.",
    106: "This molecule has low permeability.",
    107: "This molecule has more hydrogen bond acceptors.",
    108: "This molecule has more hydrogen bond donors.",

    109: "This molecule has high bioavailability.",
    110: "This molecule has low toxicity.",
    111: "This molecule is metabolically stable.",
    
    201: "This molecule is soluble in water and has more hydrogen bond acceptors.",
    202: "This molecule is insoluble in water and has more hydrogen bond acceptors.",
    203: "This molecule is soluble in water and has more hydrogen bond donors.",
    204: "This molecule is insoluble in water and has more hydrogen bond donors.",
    205: "This molecule is soluble in water and has high permeability.",
    206: "This molecule is soluble in water and has low permeability.",

    301: "This molecule looks like Penicillin.",
    302: "This molecule looks like Aspirin.",
    303: "This molecule looks like Caffeine.",
    304: "This molecule looks like Cholesterol.",
    305: "This molecule looks like Dopamine.",
    306: "This molecule looks like Cysteine.",
    307: "This molecule looks like Glutathione.",
    
    401: "This molecule is tested positive in an assay that are inhibitors and substrates of an enzyme protein. It uses molecular oxygen inserting one oxygen atom into a substrate, and reducing the second into a water molecule.",
    402: "This molecule is tested positive in an assay for Anthrax Lethal, which acts as a protease that cleaves the N-terminal of most dual specificity mitogen-activated protein kinase kinases.",
    403: "This molecule is tested positive in an assay for Activators of ClpP, which cleaves peptides in various proteins in a process that requires ATP hydrolysis and has a limited peptidase activity in the absence of ATP-binding subunits.",
    404: "This molecule is tested positive in an assay for activators involved in the transport of proteins between the endosomes and the trans Golgi network.",
    405: "This molecule is an inhibitor of a protein that prevents the establishment of the cellular antiviral state by inhibiting ubiquitination that triggers antiviral transduction signal and inhibits post-transcriptional processing of cellular pre-mRNA.",
    406: "This molecule is tested positive in the high throughput screening assay to identify inhibitors of the SARS coronavirus 3C-like Protease, which cleaves the C-terminus of replicase polyprotein at 11 sites.",

    501: "This molecule is more soluble in water than in oil.",
    502: "This molecule is less soluble in water than in oil.",
    503: "This molecule has higher logP.",
    504: "This molecule has lower logP.",
    505: "This molecule has lower TPSA.",
    506: "This molecule has higher TPSA.",
    507: "This molecule has less hydrogen bond acceptors.",
    508: "This molecule has less hydrogen bond donors.",
    509: "This molecule has high molecular weight.",
    510: "This molecule has low molecular weight.",

    601: "This molecule has lower logP.",
    602: "This molecule has higher logP.",
    603: "This molecule has higher QED.",
    604: "This molecule has lower QED.",
    605: "This molecule has lower TPSA.",
    606: "This molecule has higher TPSA.",
    607: "This molecule has more HBA.",
    608: "This molecule has more HBD.",

    601: "This molecule has lower logP and more HBA.",
    602: "This molecule has higher logP and more HBA.",
    603: "This molecule has lower logP and more HBD.",
    604: "This molecule has higher logP and more HBD.",
    605: "This molecule has lower logP and lower TPSA.",
    606: "This molecule has lower logP and higher TPSA.",

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


def get_edit_SMILES_list(args):
    SMILES_list = []
    if args.test:
        if args.edit_SMILES is not None:
            SMILES_list.append(args.edit_SMILES)
    else:
        f = open(args.edit_SMILES_filepath, 'r')
        lines = f.readlines()
        for line in lines:
            SMILES = line.strip()
            if len(SMILES) > 0:
                SMILES_list.append(SMILES)
    return SMILES_list


def get_edit_prompt(args):
    if args.test:
        if args.edit_prompt is not None:
            return args.input_description
    else:
        if args.edit_task_id not in DESCRIPTION_DICT.keys():
            raise ValueError
        else:
            print("Use {} descrition.".format(args.edit_task_id))
            return DESCRIPTION_DICT[args.edit_task_id]


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


def load_space_projector(args):
    gen2joint_projector = MLP(args.gen_emb_dim, [args.SSL_emb_dim, args.SSL_emb_dim])
    joint2gen_projector = MLP(args.SSL_emb_dim, [args.gen_emb_dim, args.gen_emb_dim])

    if args.resume:
        if osp.exists(args.gen2joint_projector_path):
            print(f"Loading gen2joint_space_projector from {args.gen2joint_projector_path}")
            state_dict = torch.load(args.gen2joint_projector_path, map_location='cpu')
            gen2joint_projector.load_state_dict(state_dict)
        else:
            print(f"{args.gen2joint_projector_path} does not exist, random initialization.")
            
        if osp.exists(args.joint2gen_projector_path):
            print(f"Loading joint2gen_space_projector from {args.joint2gen_projector_path}")
            state_dict = torch.load(args.joint2gen_projector_path, map_location='cpu')
            joint2gen_projector.load_state_dict(state_dict)
        else:
            print(f"{args.joint2gen_projector_path} does not exist, random initialization.")
    
    return gen2joint_projector, joint2gen_projector


def get_molecule_similarity(mol_a, mol_b):
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024)
    sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    return sim


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



def evaluate_SMILES_success_rate(input_smi, output_smi, task_id):
    '''
    input_smi: str # input SMILES
    output_smi: str # output SMILES
    task_id: int # pre-defined task id in DESCRIPTION_DICT above
    '''
    # check smiles validation
    input_mol = Chem.MolFromSmiles(input_smi)
    output_mol = Chem.MolFromSmiles(output_smi)
    if input_mol is None and output_mol is None:
        return False
        # return False, "both invalid SMILES"
    elif input_mol is None and output_mol is not None:
        return False
        # return False, "invalid input SMILES"
    elif input_mol is not None and output_mol is None:
        return False
        # return False, "invalid output SMILES"

    success_flag = True
    # record = []
    # logP evaluation
    if task_id in [101, 102, 201, 202, 203, 204, 205, 206, 501, 502, 503, 504, 601]:
        input_prop = Descriptors.MolLogP(input_mol)
        output_prop = Descriptors.MolLogP(output_mol)
        if task_id in [101, 201, 203, 205, 206, 501, 504]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"logP input:{input_prop} <= output:{output_prop} failed")
        elif task_id in [102, 202, 204, 502, 503]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"logP input:{input_prop} >= output:{output_prop} failed")
    # QED evaluation
    if task_id in [103, 104]:
        input_prop = Descriptors.qed(input_mol)
        output_prop = Descriptors.qed(output_mol)
        if task_id in [103]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"QED input:{input_prop} >= output:{output_prop} failed")
        elif task_id in [104]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"QED input:{input_prop} <= output:{output_prop} failed")
    # TPSA evaluation
    if task_id in [105, 106, 205, 206, 505, 506]:
        input_prop = Descriptors.TPSA(input_mol)
        output_prop = Descriptors.TPSA(output_mol)
        if task_id in [105, 205, 506]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"TPSA input:{input_prop} <= output:{output_prop} failed")
        elif task_id in [106, 206, 505]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"TPSA input:{input_prop} >= output:{output_prop} failed")
    # HBA evaluation
    if task_id in [107, 201, 202, 507]:
        input_prop = Descriptors.NumHAcceptors(input_mol)
        output_prop = Descriptors.NumHAcceptors(output_mol)
        if task_id in [107, 201, 202]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"hydrogen bond acceptors input:{input_prop} >= output:{output_prop} failed")
        elif task_id in [507]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"hydrogen bond acceptors input:{input_prop} <= output:{output_prop} failed")
    # HBD evaluation
    if task_id in [108, 203, 204, 508]:
        input_prop = Descriptors.NumHDonors(input_mol)
        output_prop = Descriptors.NumHDonors(output_mol)
        if task_id in [108, 203, 204]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"hydrogen bond donors input:{input_prop} >= output:{output_prop} failed")
        elif task_id in [508]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"hydrogen bond donors input:{input_prop} <= output:{output_prop} failed")
    # Molecular Weight evaluation
    if task_id in [509, 510]:
        input_prop = Descriptors.MolWt(input_mol)
        output_prop = Descriptors.MolWt(output_mol)
        if task_id in [509]:
            if input_prop >= output_prop:
                success_flag = False
                # record.append(f"Molecular Weight input:{input_prop} >= output:{output_prop} failed")
        elif task_id in [510]:
            if input_prop <= output_prop:
                success_flag = False
                # record.append(f"Molecular Weight input:{input_prop} <= output:{output_prop} failed")
    if success_flag:
        return success_flag
        # return success_flag, "success"
    else:
        return success_flag
        # return success_flag, " and ".join(record)