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

from MolEditFormer.basic_utils import seed_all, Logger
from MolEditFormer.models import CLIP
from MolEditFormer.datasets import PubChemEdit, MolPair_SingleGraph, MolGraphDataset


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.dir_name == "":
        model_save_dir = osp.join(args.store_dir, f"{args.data_source}_reconstruct")
    else:
        model_save_dir = osp.join(args.store_dir, args.dir_name)
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if args.molecule_type in ["2DGraph", "all"]:
        mol_args = {
            "molecule_type": args.molecule_type,
            "gnn_type": args.gnn_type,
            "num_layer": args.num_layer,
            "gnn_emb_dim": args.gnn_emb_dim,
            "JK": args.JK,
            "dropout_ratio": args.dropout_ratio,
            "graph_pooling": args.graph_pooling,
            "model_path": args.mol_model_path,
        }
    if args.molecule_type in ["3DGraph", "all"]:
        pass
    if args.molecule_type in ["SMILES", "all"]:
        mol_args = {
            "molecule_type": args.molecule_type,
            "smiles_emb_dim": args.smiles_emb_dim, 
            "vocab_path" : args.smiles_vocab_path, 
            "model_path": args.mol_model_path,
        }

    model = CLIP(
        mol_branch=args.mol_branch,
        text_branch=args.text_branch,
        mode=args.model_mode,
        device=device,
        mol_args=mol_args,
        text_args=None,
        CL_args=None,
    ).to(device)
    model.eval()

    if args.data_source == "PubChemEdit":
        test_set = PubChemEdit(args.data_dir, mode=args.dataset_mode, can_smiles=args.can_smiles)
    elif args.data_source == "MolPair":
        test_set = MolPair_SingleGraph(args.data_dir)
    test_loader = pyg_DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    original_smiles_list, result_smiles_list = [], []
    for i_batch, sample_batched in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
        # load data from dataloader
        if args.molecule_type not in ["2DGraph", "3DGraph", "SMILES", "all"]:
            raise ValueError("Invalid molecule type")
        
        if args.molecule_type == "2DGraph":
            molecule_batched = sample_batched[2].to(device)
        elif args.molecule_type == "3DGraph":
            pass
        elif args.molecule_type == "SMILES":
            molecule_batched = sample_batched[0]
        elif args.molecule_type == "all":
            pass
        original_smiles_list.extend(molecule_batched)
        token_ids, pad_mask = model.prepare_smiles_tokens(molecule_batched)
        batch_input = {'encoder_input': token_ids, 'encoder_pad_mask': pad_mask}
        output_mol_strs, _ = model.sample_molecules(batch_input, sampling_alg=args.sampling_alg)
        result_smiles_list.extend(output_mol_strs)
    
    df = pd.DataFrame({
        'original_smiles': original_smiles_list,
        'result_smiles': result_smiles_list
    })
    df.to_csv(osp.join(model_save_dir, 'smiles_results.csv'), index=False)

    # Calculate reconstruction accuracy
    accurate_num = 0
    for original, result in zip(original_smiles_list, result_smiles_list):
        if original == result:
            accurate_num += 1
    accuracy = accurate_num / len(original_smiles_list)
    print(f"Reconstruction accuracy: {accuracy:.2%}")


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
