import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, allowable_features


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
    "qed_test": "increase QED to at least 0.9",
    "plogp_test": "increase PlogP as much as possible",
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
    "qed_test": "more like a drug to at least 0.9",
    "plogp_test": "more penalized insoluble in water as much as possible",
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
    "qed_test": "increase QED from high to extremely high",
    "plogp_test": "increase PlogP as much as possible",
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
    "qed_test": "",
    "plogp_test": "",
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
    "qed_test": {"input_level1": "QED"},
    "plogp_test": {"input_level1": "PlogP"},
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
