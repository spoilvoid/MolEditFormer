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

from transformers import AutoModel, AutoTokenizer

from MolEditFormer.basic_utils import get_local_time, seed_all, Logger
from MolEditFormer.models import CLIP
from MolEditFormer.datasets import MolPair_PairSmiles_ZeroShot_Test
from molecule_edit_utils import evaluate_molecular_edit_result


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    result_save_dir = osp.join(args.store_dir, f"{args.data_source}_{args.molecule_type}")

    fuse_args = {
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "model_path": args.fuser_path,
    }
    mol_args = {
        "molecule_type": args.molecule_type,
        "smiles_emb_dim": args.smiles_emb_dim, 
        "vocab_path" : args.smiles_vocab_path, 
        "model_path": args.mol_model_path,
    }
    text_args = {
        "text_emb_dim": args.text_emb_dim,
        "max_seq_len": args.max_seq_len,
        "tokenizer_dir": args.text_tokenizer_dir,
        "model_path": args.text_model_path,
    }

    model = CLIP(
        mol_branch=args.mol_branch,
        text_branch=args.text_branch,
        mode=args.model_mode,
        device=device,
        mol_args=mol_args,
        text_args=text_args,
        fuse_args=fuse_args,
    ).to(device)
    model.eval()

    test_set = MolPair_PairSmiles_ZeroShot_Test(args.data_dir, task_id=args.task_id)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    original_smiles_list, result_smiles_list = [], []
    for i_batch, sample_batched in tqdm(enumerate(test_loader), disable=False, total=len(test_loader)):
        input_molecule_batched = sample_batched[0]
        description_batched = sample_batched[1]

        edited_molecule_batched = model.edit_molecules(batch_input_molecule=input_molecule_batched, batch_input_text=description_batched, sampling_alg=args.sampling_alg)

        original_smiles_list.extend(input_molecule_batched)
        result_smiles_list.extend(edited_molecule_batched)

    result_list, reason_list = [], []
    for input_smi, output_smi in zip(original_smiles_list, result_smiles_list):
        result, reason = evaluate_molecular_edit_result(input_smi, output_smi, task_id=args.task_id)
        result_list.append(result)
        reason_list.append(reason)

    df = pd.DataFrame({
        'original_smiles': original_smiles_list,
        'result_smiles': result_smiles_list,
        'result': result_list,
        'reason': reason_list,
    })
    df.to_csv(osp.join(result_save_dir, f"task_{args.task_id}_results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--data_dir", type=str, default="data/MolPair/mol_pair")
    parser.add_argument("--task_id", type=str, default=101, choices=list(range(101, 109))+list(range(201, 207)))
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # inference config
    parser.add_argument("--sampling_alg", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--model_mode", type=str, default="edit", choices=["pretrain", "finetune", "reconstruct", "edit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    # model config
    parser.set_defaults(repr_frozen=False)
    parser.add_argument("--mol_branch", dest='mol_branch', action='store_true')
    parser.add_argument('--no_mol_branch', dest='mol_branch', action='store_false')
    parser.set_defaults(mol_branch=True)
    parser.add_argument("--text_branch", dest='text_branch', action='store_true')
    parser.add_argument('--no_text_branch', dest='text_branch', action='store_false')
    parser.set_defaults(text_branch=True)
    # text branch config
    parser.add_argument("--text_emb_dim", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # smiles branch config
    parser.add_argument('--smiles_model_type', type=str, default="MegaMolBART", choices=["MegaMolBART"])
    parser.add_argument("--smiles_vocab_path", type=str, default="ckpt/MegaMolBART/bart_vocab.txt")
    parser.add_argument("--smiles_emb_dim", type=int, default=256)
    # load config
    parser.add_argument('--text_tokenizer_dir', type=str, default='ckpt/SciBERT')
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--mol_model_path', type=str, default='ckpt/MegaMolBART/model_weight.pth')
    parser.add_argument('--fuser_path', type=str, default=None)
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolEditFormer/inference/edit")
    # fuser config
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    args.molecule_type = "SMILES"
    args.data_source = "MolPair"

    start = time.perf_counter()
    main(args)
    

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
