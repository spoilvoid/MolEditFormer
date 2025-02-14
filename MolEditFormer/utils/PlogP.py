from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed as computer_qed_score
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
from . import sascorer
import networkx as nx


def calculateScore(mol):
    logp = Descriptors.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    # plogp(m) = logP(m) − SA(m) − cycle(m) where cycle(m) counts the number of rings that have more than six atoms.
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    plog = logp - sa - cycle_length

    return plog

if __name__=="__main__":
    print(calculateScore('COc1ccc(C(=O)N(C)[C@@H](C)C/C(N)=N/O)cc1O'))