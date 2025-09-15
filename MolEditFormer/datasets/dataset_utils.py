import random
import numpy as np
import networkx as nx

import torch

from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, allowable_features

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DESCRIPTION_MODE = ["full", "main", "expand"]
RETRIEVAL_MODE = ["random", "iterative"]
# v1: valued+prop_name, v2: valued+prop_explanation, v3: tagged+prop_name, v4: tagged+prop_explanation
VERSION = ["v1", "v2", "v3", "v4"]
TEXT_REQUIREMENTS_V1 = {
    "101": "decrease logP",
    "102": "increase logP",
    "103": "increase QED",
    "104": "decrease QED",
    "105": "decrease TPSA",
    "106": "increase TPSA",
    "107": "increase HBA",
    "108": "increase HBD",
    "201": "decrease logP and increase HBA",
    "202": "increase logP and increase HBA",
    "203": "decrease logP and increase HBD",
    "204": "increase logP and increase HBD",
    "205": "decrease logP and decrease TPSA",
    "206": "decrease logP and increase TPSA",
    "QED_constrained_optimization": "increase QED to at least 0.9",
    "PlogP_constrained_optimization": "increase PlogP as much as possible",
    "COX2": "increase the binding affinity for the target",
    "DRD2": "increase the binding affinity for the target",
    "EGFR": "increase the binding affinity for the target",
    "SARS_Cov_3C": "increase the binding affinity for the target",
    "2QBR_active2active": "maintain pIC50 for the target at least 6.5 with certain activity while ${task_requirement}",
    "2QBR_other": "increase pIC50 for the target ${task_requirement}",
}
TEXT_REQUIREMENTS_V2 = {
    "101": "more soluble in water",
    "102": "more insoluble in water",
    "103": "more like a drug",
    "104": "more unlike a drug",
    "105": "more permeable",
    "106": "less permeable",
    "107": "more hydrogen bond acceptors",
    "108": "more hydrogen bond donors",
    "201": "more soluble in water and more hydrogen bond acceptors",
    "202": "more insoluble in water and more hydrogen bond acceptors",
    "203": "more soluble in water and more hydrogen bond donors",
    "204": "more insoluble in water and more hydrogen bond donors",
    "205": "more soluble in water and more permeable",
    "206": "more soluble in water and less permeable",
    "QED_constrained_optimization": "more like a drug to at least 0.9",
    "PlogP_constrained_optimization": "more penalized insoluble in water as much as possible",
    "COX2": "make the binding with the target stronger and more stable",
    "DRD2": "make the binding with the target stronger and more stable",
    "EGFR": "make the binding with the target stronger and more stable",
    "SARS_Cov_3C": "make the binding with the target stronger and more stable",
}
TEXT_REQUIREMENTS_V3 = {
    "101": "decrease logP from ${input_level1} to ${output_level1}",
    "102": "increase logP from ${input_level1} to ${output_level1}",
    "103": "increase QED from ${input_level1} to ${output_level1}",
    "104": "decrease QED from ${input_level1} to ${output_level1}",
    "105": "decrease TPSA from ${input_level1} to ${output_level1}",
    "106": "increase TPSA from ${input_level1} to ${output_level1}",
    "107": "increase HBA from ${input_level1} to ${output_level1}",
    "108": "increase HBD from ${input_level1} to ${output_level1}",
    "201": "decrease logP from ${input_level1} to ${output_level1} and increase HBA from ${input_level2} to ${output_level2}",
    "202": "increase logP from ${input_level1} to ${output_level1} and increase HBA from ${input_level2} to ${output_level2}",
    "203": "decrease logP from ${input_level1} to ${output_level1} and increase HBD from ${input_level2} to ${output_level2}",
    "204": "increase logP from ${input_level1} to ${output_level1} and increase HBD from ${input_level2} to ${output_level2}",
    "205": "decrease logP from ${input_level1} to ${output_level1} and decrease TPSA from ${input_level2} to ${output_level2}",
    "206": "decrease logP from ${input_level1} to ${output_level1} and increase TPSA from ${input_level2} to ${output_level2}",
    "QED_constrained_optimization": "increase QED from high to extremely high",
    "PlogP_constrained_optimization": "increase PlogP as much as possible",
    "COX2": "increase the binding affinity for the target from ${input_level1} to ${output_level1}",
    "DRD2": "increase the binding affinity for the target from ${input_level1} to ${output_level1}",
    "EGFR": "increase the binding affinity for the target from ${input_level1} to ${output_level1}",
    "SARS_Cov_3C": "increase the binding affinity for the target from ${input_level1} to ${output_level1}",
    "2QBR_active2active": "maintain pIC50 for the target at least slightly high pIC50 with certain activity while ${task_requirement}",
    "2QBR_other": "increase pIC50 for the target from ${input_level1} to ${output_level1}",
}
TEXT_REQUIREMENTS_V4 = {
    "101": "",
    "102": "",
    "103": "",
    "104": "",
    "105": "",
    "106": "",
    "107": "",
    "108": "",
    "201": "",
    "202": "",
    "203": "",
    "204": "",
    "205": "",
    "206": "",
    "QED_constrained_optimization": "",
    "PlogP_constrained_optimization": "",
    "COX2": "",
    "DRD2": "",
    "EGFR": "",
    "SARS_Cov_3C": "",
}
LEVEL_LABELS_PROP_NAME = {
    "QED": {
        (0, 0.1): "almost no QED",
        (0.1, 0.2): "extremely Low QED",
        (0.2, 0.3): "very Low QED",
        (0.3, 0.4): "low QED",
        (0.4, 0.5): "slightly Low QED",
        (0.5, 0.6): "moderate QED",
        (0.6, 0.7): "slightly high QED",
        (0.7, 0.8): "high QED",
        (0.8, 0.9): "very high QED",
        (0.9, 1): "extremely high QED",
    },
    "logP": {
        ("-inf", -2): "very low logP",
        (-2, 0): "low logP",
        (0, 2): "moderate logP",
        (2, 4): "slightly high logP",
        (4, 6): "high logP",
        (6, "+inf"): "very high logP",
    },
    "TPSA": {
        (0, 30): "extremely low TPSA",
        (30, 60): "low TPSA",
        (60, 80): "slightly low TPSA",
        (80, 100): "moderate TPSA",
        (100, 120): "slightly high TPSA",
        (120, 140): "high TPSA",
        (140, 200): "very high TPSA",
        (200, "+inf"): "extremely high TPSA",
    },
    "HBA": {
        (0, 2): "extremely low HBA",
        (2, 4): "low HBA",
        (4, 6): "slightly low HBA",
        (6, 8): "moderate HBA",
        (8, 10): "slightly high HBA",
        (10, 13): "high HBA",
        (13, "+inf"): "extremely high HBA",
    },
    "HBD": {
        (0, 2): "extremely low HBD",
        (2, 4): "low HBD",
        (4, 6): "slightly low HBD",
        (6, 8): "moderate HBD",
        (8, 10): "slightly high HBD",
        (10, 13): "high HBD",
        (13, "+inf"): "extremely high HBD",
    },
    "SA": {
        (1, 2): "very low SA",
        (2, 4): "low SA",
        (4, 6): "moderate SA",
        (6, 8): "high SA",
        (8, 10): "very high SA",
    },
    "PlogP": {
        ("-inf", -2): "very low PlogP",
        (-2, 0): "low PlogP",
        (0, 2): "moderate PlogP",
        (2, 4): "slightly high PlogP",
        (4, 6): "high PlogP",
        (6, "+inf"): "very high PlogP",
    },
    "vina_affinity": {
        ("-inf", -11): "very high binding affinity",
        (-11, -9): "high binding affinity",
        (-9, -7): "moderate binding affinity",
        (-7, -5): "low binding affinity",
        (-5, "+inf"): "almost no binding affinity",
    },
    "pIC50": {
        ("-inf", 4.5): "very low pIC50 with certain inactivity",
        (4.5, 5): "low pIC50 with certain inactivity",
        (5, 5.5): "slightly low pIC50 with certain inactivity",
        (5.5, 6): "moderate pIC50 with activity uncertain but leaning toward inactive",
        (6, 6.5): "moderate pIC50 with activity uncertain but leaning toward active",
        (6.5, 7): "slightly high pIC50 with certain activity",
        (7, 7.5): "high pIC50 with certain activity",
        (7.5, "+inf"): "very high pIC50 with certain activity",
    },
}
LEVEL_LABELS_PROP_EXPLANATION = {
    "QED": {
        (0, 0.1): "not a drug",
        (0.1, 0.2): "extremely not like a drug",
        (0.2, 0.3): "very not like a drug",
        (0.3, 0.4): "not like a drug",
        (0.4, 0.5): "slightly not like a drug",
        (0.5, 0.6): "moderate drug-likeness",
        (0.6, 0.7): "slightly like a drug",
        (0.7, 0.8): "like a drug",
        (0.8, 0.9): "very like a drug",
        (0.9, 1): "extremely like a drug",
    },
    "logP": {
        ("-inf", -2): "extremely soluble in water",
        (-2, 0): "soluble in water",
        (0, 2): "moderate between soluble and insoluble",
        (2, 4): "slightly insoluble in water",
        (4, 6): "insoluble in water",
        (6, "+inf"): "practically insoluble in water",
    },
    "TPSA": {
        (0, 30): "extremely high permeability",
        (30, 60): "high permeability",
        (60, 80): "slightly high permeability",
        (80, 100): "moderate permeability",
        (100, 120): "slightly low permeability",
        (120, 140): "low permeability",
        (140, 200): "very low permeability",
        (200, "+inf"): "almost no permeability",
    },
    "HBA": {
        (0, 2): "extremely low Hydrogen Bond Acceptors",
        (2, 4): "low Hydrogen Bond Acceptors",
        (4, 6): "slightly low Hydrogen Bond Acceptors",
        (6, 8): "extremely low Hydrogen Bond Acceptors",
        (8, 10): "slightly high Hydrogen Bond Acceptors",
        (10, 13): "high Hydrogen Bond Acceptors",
        (13, "+inf"): "extremely high Hydrogen Bond Acceptors",
    },
    "HBD": {
        (0, 2): "extremely low Hydrogen Bond Donors",
        (2, 4): "low Hydrogen Bond Donors",
        (4, 6): "slightly low Hydrogen Bond Donors",
        (6, 8): "extremely low Hydrogen Bond Donors",
        (8, 10): "slightly high Hydrogen Bond Donors",
        (10, 13): "high Hydrogen Bond Donors",
        (13, "+inf"): "extremely high Hydrogen Bond Donors",
    },
    "SA": {
        (1, 2): "very easy to synthesis",
        (2, 4): "easy to synthesis",
        (4, 6): "moderate to synthesis",
        (6, 8): "hard to synthesis",
        (8, 10): "very hard to synthesis",
    },
    "PlogP": {
        ("-inf", -2): "extremely penalized soluble in water",
        (-2, 0): "penalized soluble in water",
        (0, 2): "moderate between penalized soluble and penalized insoluble",
        (2, 4): "slightly penalized insoluble in water",
        (4, 6): "penalized insoluble in water",
        (6, "+inf"): "practically penalized insoluble in water",
    },
    "vina_affinity": {
        ("-inf", -11): "very high binding to the target",
        (-11, -9): "high binding to the target",
        (-9, -7): "moderate binding to the target",
        (-7, -5): "low binding to the target",
        (-5, "+inf"): "almost no binding to the target",
    },
    "pIC50": {
        ("-inf", 4.5): "very low binding affinity with certain inactivity",
        (4.5, 5): "low binding affinity with certain inactivity",
        (5, 5.5): "slightly low binding affinity with certain inactivity",
        (5.5, 6): "moderate binding affinity with activity uncertain but leaning toward inactive",
        (6, 6.5): "moderate binding affinity with activity uncertain but leaning toward active",
        (6.5, 7): "slightly high binding affinity with certain activity",
        (7, 7.5): "high binding affinity with certain activity",
        (7.5, "+inf"): "very high binding affinity with certain activity",
    },
}
TASK_REFERENCE = {
    "101": {"input_level1": "logP"},
    "102": {"input_level1": "logP"},
    "103": {"input_level1": "QED"},
    "104": {"input_level1": "QED"},
    "105": {"input_level1": "TPSA"},
    "106": {"input_level1": "TPSA"},
    "107": {"input_level1": "HBA"},
    "108": {"input_level1": "HBD"},
    "201": {"input_level1": "logP", "input_level2": "HBA"},
    "202": {"input_level1": "logP", "input_level2": "HBA"},
    "203": {"input_level1": "logP", "input_level2": "HBD"},
    "204": {"input_level1": "logP", "input_level2": "HBD"},
    "205": {"input_level1": "logP", "input_level2": "TPSA"},
    "206": {"input_level1": "logP", "input_level2": "TPSA"},
    "QED_constrained_optimization": {"input_level1": "QED"},
    "PlogP_constrained_optimization": {"input_level1": "PlogP"},
    "COX2": {"input_level1": "binding_affinity"},
    "DRD2": {"input_level1": "binding_affinity"},
    "EGFR": {"input_level1": "binding_affinity"},
    "SARS_Cov_3C": {"input_level1": "binding_affinity"},
    "2QBR": {"input_level1": "pIC50"},
}


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeNetGraphDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def shuffle_atom_order(smiles):
    """ shuffle atom order to get un-canonical mol object 
    :param smiles: str object of SMILES
    :return: rdkit mol object """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    atom_order = list(range(mol.GetNumAtoms()))
    random.shuffle(atom_order)
    
    mol_copy = Chem.RenumberAtoms(mol, atom_order)

    shuffled_smiles = Chem.MolToSmiles(mol_copy, canonical=False)

    return shuffled_smiles


def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)
    return mol


def graph_data_obj_to_nx_simple(data):
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx, bond_dir_idx=bond_dir_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def murcko_scaffold_with_attachments(smiles: str, generic: bool=False, isomeric: bool=True):
    """
    Input:
        smiles: input molecule SMILES
        generic: whether to use MakeScaffoldGeneric to generalize scaffold atoms/bonds
        isomeric: whether to keep stereochemistry information in output SMILES
    Output:
        scaffold_smi_with_star: Murcko scaffold SMILES with attachment points [*]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES")

        # 1) Generate the Murcko scaffold (framework of rings and linkers)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            # If no scaffold is found, return ""
            return ""

        if generic:
            # Convert scaffold into a generic form (all atoms → C, all bonds → single)
            core = MurckoScaffold.MakeScaffoldGeneric(core)

        # 2) Map the scaffold back onto the original molecule
        #    substructure match gives mapping: core atom index → original mol atom index
        matches = mol.GetSubstructMatches(core, useChirality=isomeric)
        if not matches:
            # If mapping fails, just return the scaffold itself
            return Chem.MolToSmiles(core, isomericSmiles=isomeric)
        match = matches[0]
        coreAtomInMol = set(match)  # set of atom indices in the original mol that belong to scaffold

        # Build mapping: original mol atom index → core atom index
        molIdx_to_coreIdx = {mol_idx: core_idx for core_idx, mol_idx in enumerate(match)}

        # 3) Identify bonds between scaffold atoms and non-scaffold atoms in the original molecule
        attach_counts = {}  # record how many attachment points each core atom should have
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            a_in = a in coreAtomInMol
            b_in = b in coreAtomInMol
            if a_in ^ b_in:  # XOR: one atom is in scaffold, the other is not
                core_mol_idx = a if a_in else b
                core_atom_idx = molIdx_to_coreIdx.get(core_mol_idx, None)
                if core_atom_idx is not None:
                    attach_counts[core_atom_idx] = attach_counts.get(core_atom_idx, 0) + 1

        # 4) For each connection, add a dummy atom [*] onto the scaffold atom
        rw = Chem.RWMol(core)
        for core_atom_idx, nstar in attach_counts.items():
            for _ in range(nstar):
                star = Chem.Atom(0)  # atomic number 0 = dummy atom [*]
                star_idx = rw.AddAtom(star)
                rw.AddBond(core_atom_idx, star_idx, order=Chem.BondType.SINGLE)

        # 5) Convert back to SMILES
        mol_with_star = rw.GetMol()
        Chem.SanitizeMol(mol_with_star, catchErrors=True)  # clean up valence/aromaticity issues
        scaffold_smi_with_star = Chem.MolToSmiles(mol_with_star, isomericSmiles=isomeric)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        scaffold_smi_with_star = ""

    return scaffold_smi_with_star


def brics_scaffold_with_attachments(smiles: str, isomeric: bool=True):
    """
    Input: 
        - smiles: SMILES string
        - isomeric: Whether to consider stereochemistry in the output SMILES
    Output:
        - scaffold_smi: BRICS scaffold SMILES with dummy atoms [*] at cleavage points
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        # 1) Identify BRICS bonds to be cleaved
        brics_bonds = BRICS.FindBRICSBonds(mol)
        bond_indices = []
        dummy_labels = []

        for (a_idx, b_idx), (a_lab, b_lab) in brics_bonds:
            bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
            bond_indices.append(bond.GetIdx())
            # For each cleaved bond, add a pair of dummy atoms with BRICS labels
            dummy_labels.append((int(a_lab), int(b_lab)))

        # 2) If no BRICS bonds are found, return ""
        if not bond_indices:
            scaffold_smi = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
            brics_frags = sorted(BRICS.BRICSDecompose(mol))
            return ""

        # 3) Cleave the bonds and insert dummy atoms [*] with BRICS labels
        broken = Chem.FragmentOnBonds(
            mol,
            bondIndices=bond_indices,
            addDummies=True,
            dummyLabels=dummy_labels
        )

        # Scaffold SMILES: molecule broken at BRICS bonds with [*] dummy atoms
        scaffold_smi = Chem.MolToSmiles(broken, isomericSmiles=isomeric)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        scaffold_smi = ""

    return scaffold_smi
