import os
import os.path as osp
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer

from models import CLIP
from datasets import PubChemEdit, MolPair_SingleGraph , DataHelper, MolGraphDataset

from MolEditFormer.MolEditFormer.utils.basic_utils import get_local_time, freeze_network, seed_all, Logger
from MolEditFormer.models import MegaMolBART


def main(args):
    dataset = PubChemEdit(args.data_dir, mode=args.dataset_mode, can_smiles=args.can_smiles)
    MegaMolBART_wrapper = MegaMolBART(vocab_path=args.smiles_vocab_path, input_dir="ckpt/MegaMolBART", output_dir=None)
    
    latent_code_init, pad_mask_init = MegaMolBART_wrapper.smileslist2embedding([dataset[0][0]])  # [pad, B, d], 
    # print(pad_mask_init)
    
    regenerated_mol = MegaMolBART_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)[0]
    print(dataset[0][0], regenerated_mol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--data_source", type=str, default="PubChemEdit", choices=["PubChemEdit", "MolPair"])
    parser.add_argument("--data_dir", type=str, default="data/PubChemEdit/version_0")
    parser.add_argument("--dataset_mode", type=str, default="full", choices=["full", "main", "expand"])
    parser.add_argument("--can_smiles", action="store_true")
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    # inference config
    parser.add_argument("--sampling_alg", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--model_mode", type=str, default="inference", choices=["pretrain", "finetune", "inference"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    # model config
    parser.add_argument("--molecule_type", type=str, default="SMILES", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.set_defaults(repr_frozen=False)
    parser.add_argument("--mol_branch", dest='mol_branch', action='store_true')
    parser.add_argument('--no_mol_branch', dest='mol_branch', action='store_false')
    parser.set_defaults(mol_branch=True)
    parser.add_argument("--text_branch", dest='text_branch', action='store_true')
    parser.add_argument('--no_text_branch', dest='text_branch', action='store_false')
    parser.set_defaults(text_branch=False)
    # text branch config
    parser.add_argument("--text_emb_dim", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # smiles branch config
    parser.add_argument('--smiles_model_type', type=str, default="MegaMolBART", choices=["MegaMolBART"])
    parser.add_argument("--smiles_vocab_path", type=str, default="ckpt/MegaMolBART/bart_vocab.txt")
    parser.add_argument("--smiles_emb_dim", type=int, default=256)
    # graph branch config
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument('--graph_pooling', type=str, default='mean')
    # load config
    parser.add_argument('--text_tokenizer_dir', type=str, default='ckpt/SciBERT')
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--mol_model_path', type=str, default='ckpt/MegaMolBART/model_weight.pth')
    parser.add_argument('--text_projector_path', type=str, default=None)
    parser.add_argument('--mol_projector_path', type=str, default=None)
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/inference")
    parser.add_argument("--dir_name", type=str, default="")

    args = parser.parse_args()

    start = time.perf_counter()
    main(args)
    

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
