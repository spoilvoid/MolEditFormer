import os
import os.path as osp
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer

from models import CLIP, tokenize
from datasets import DrugBank_retrieval_Graph, DrugBank_ATC_Graph

from basic_utils import Logger, seed_all, get_local_time, default_dump


def do_CL_eval(X, Y, neg_Y):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0) # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]
    labels = torch.zeros(B).long().to(logits.device)  # B*1

    criterion = nn.CrossEntropyLoss()

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    confidence = logits
    CL_conf = confidence.max(dim=1)[0]
    CL_conf = CL_conf.cpu().numpy()

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_loss, CL_conf, CL_acc


@torch.no_grad()
def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if not osp.exists(args.store_dir):
        os.mkdir(args.store_dir)

    T_max = max(args.T_list) - 1
    if args.task == "molecule_description_removed_PubChem":
        template = "SMILES_description_removed_from_PubChem_{}.txt"
        dataset = DrugBank_retrieval_Graph(args.data_dir, args.split, neg_sample_size=T_max, processed_dir_prefix=args.task, template=template)
    elif args.task == "molecule_pharmacodynamics_removed_PubChem":
        template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        dataset = DrugBank_retrieval_Graph(args.data_dir, args.split, neg_sample_size=T_max, processed_dir_prefix=args.task, template=template)
    elif args.task == "molecule_ATC":
        prompt_template = "This molecule is for {}."
        full_file_name = "SMILES_ATC_{}_full.txt".format(args.ATC_level)
        full_processed_dir_prefix = "ATC_full_{}".format(args.ATC_level)
        dataset = DrugBank_ATC_Graph(args.data_dir, full_file_name, full_processed_dir_prefix, neg_sample_size=T_max, prompt_template=prompt_template)
    dataloader = pyg_DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = CLIP(args).to(device)
    model.eval()
    
    accum_acc_list = [0 for _ in args.T_list]   
    for batch in tqdm(dataloader):
        text = batch[0]
        molecule_data = batch[1]
        neg_text = batch[2]
        neg_molecule_data = batch[3]

        text2joint_repr = model.encode_text_from_pretrain_model(text, device)
        mol2joint_repr = model.encode_graph(molecule_data.to(device))

        if args.test_mode == "given_text":
            neg_mol2joint_repr = [model.encode_graph(neg_molecule_data[idx].to(device)) for idx in range(T_max)]
            neg_mol2joint_repr = torch.stack(neg_mol2joint_repr)

            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(text2joint_repr, mol2joint_repr, neg_mol2joint_repr[:T-1])
                accum_acc_list[T_idx] += acc
        elif args.test_mode == "given_molecule":
            neg_text2joint_repr = [model.encode_text_from_pretrain_model(neg_text[idx], device) for idx in range(T_max)]
            neg_text2joint_repr = torch.stack(neg_text2joint_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(mol2joint_repr, text2joint_repr, neg_text2joint_repr[:T-1])
                accum_acc_list[T_idx] += acc
        else:
            raise Exception
    
    accum_acc_list = np.array(accum_acc_list)
    accum_acc_list /= len(dataloader)

    print('Initial', accum_acc_list)

    row = ", ".join(["{:.4f}".format(x * 100) for x in accum_acc_list])
    print("initial results,", row)

    result_dict = {}
    for T, acc in zip(args.T_list, accum_acc_list):
        result_dict[T] = acc
    if args.store_dir is not None:
        save_filename = f"{args.task}.json"
        json_str = json.dumps(result_dict, ensure_ascii=False, default=default_dump)
        with open(os.path.join(args.store_dir, save_filename), 'w', encoding='utf-8') as file:
            file.write(json_str)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--data_dir", type=str, default="data/DrugBank/retrieval")
    parser.add_argument("--split", type=str, default="full", choices=["full", "train", "val", "test"])
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # globe config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    # model config
    parser.add_argument("--molecule_type", type=str, default="2DGraph", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.add_argument("--mol_branch", dest='mol_branch', action='store_true')
    parser.add_argument('--no_mol_branch', dest='mol_branch', action='store_false')
    parser.set_defaults(mol_branch=True)
    parser.add_argument("--text_branch", dest='text_branch', action='store_true')
    parser.add_argument('--no_text_branch', dest='text_branch', action='store_false')
    parser.set_defaults(text_branch=True)
    # text branch config
    parser.add_argument("--text_emb_dim", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # graph branch config
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G", "GraphMVP_C"])
    # projector config
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    # load config
    parser.add_argument('--text_pretrain_dir', type=str, default='ckpt/SciBERT')
    parser.add_argument('--mol_pretrain_dir', type=str, default='ckpt/GraphMVP')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_resume', dest='resume', action='store_false')
    parser.set_defaults(resume=False)
    parser.add_argument('--text_model_path', type=str, default='ckpt/mol_align/text_model.pth')
    parser.add_argument('--text_projector_path', type=str, default='ckpt/mol_align/text_projector.pth')
    parser.add_argument('--mol_model_path', type=str, default='ckpt/mol_align/mol_model.pth')
    parser.add_argument('--mol_projector_path', type=str, default='ckpt/mol_align/mol_projector.pth')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/DrugBank_retrieval")
    # eval config
    parser.add_argument("--task", type=str, default="molecule_description",
        choices=["molecule_description_removed_PubChem", "molecule_pharmacodynamics_removed_PubChem", "molecule_ATC"])
    parser.add_argument("--test_mode", type=str, default="given_text", choices=["given_text", "given_molecule"])
    parser.add_argument("--ATC_level", type=int, default=5, choices=[1, 3, 4, 5])
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])

    args = parser.parse_args()

    start = time.perf_counter()
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))