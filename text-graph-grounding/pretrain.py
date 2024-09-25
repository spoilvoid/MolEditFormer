import os
import os.path as osp
import sys
import random
from random import sample
import math
import time
import argparse
import json
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer

from models import CLIP, tokenize
from datasets import PubChemEdit, DataHelper, MolGraphDataset

from basic_utils import get_local_time, freeze_network, seed_all, Logger


def cycle_index(num, shift):
    '''
    num, shift: int
    num > shift > 0
    return [shift, shift+1, ..., num-1, 0, 1, ..., shift-1]
    '''
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


# def cal_cl_loss(s_features, t_features, labels):
#     logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
#     logits = logit_scale * s_features @ t_features.t()
#     loss_i = F.cross_entropy(logits, labels)
#     loss_t = F.cross_entropy(logits.T, labels)
#     ret_loss = (loss_i + loss_t) / 2
#     return ret_loss


def cl_loss(s_features, t_features, args):
    '''
    s_features [batch_size, SSL_emb_dim]: molecular features 
    t_features [batch_size, SSL_emb_dim]: description text features 
    '''
    if args.normalize:
        X = F.normalize(s_features, dim=-1)
        Y = F.normalize(t_features, dim=-1)

    if args.SSL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        # use cycle_index to form k negative samples
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        # calculate the cosine similarity for each sample
        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        # calculate the contrastive learning loss according to the weighted sum
        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        # calculate the contrastive learning accuracy(pred_pos > 0 and pred_neg < 0)
        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        # suppose data in mini_batch should own different labels
        B = X.size()[0]
        # calculate logits by integrating text and structure features for each sample
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_save_dir = osp.join(args.store_dir, f"{args.molecule_type}_{args.gnn_type}-{get_local_time()}")
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logger = Logger(osp.join(model_save_dir, "log"), args.time_log)
    writer = SummaryWriter(osp.join(model_save_dir, "tensorboard"))
    

    model = CLIP(args).to(device)
    model.train()
    # dataset = DataHelper(arr_edge_index, args)
    # in_g = Data(x=node_f, edge_index=edge_index).to(device)
    # dataset = MolGraphDataset(args.graph_root)

    train_set = PubChemEdit(args.data_dir)
    train_loader = pyg_DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.repr_frozen:
        freeze_network(model.text_model)
        freeze_network(model.molecule_model)
        model_param_group = [
            {"params": model.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
            {"params": model.mol2latent.parameters(), "lr": args.graph_lr * args.graph_lr_scale},
        ]
        save_config = {
            "text_model": False,
            "molecule_model": False,
            "text2latent": True,
            "mol2latent": True,
        }
    else:
        model_param_group = [
            {"params": model.text_model.parameters(), "lr": args.text_lr},
            {"params": model.molecule_model.parameters(), "lr": args.graph_lr},
            {"params": model.text2latent.parameters(), "lr": args.text_lr * args.text_lr_scale},
            {"params": model.mol2latent.parameters(), "lr": args.graph_lr * args.graph_lr_scale},
        ]
        save_config = {
            "text_model": True,
            "molecule_model": True,
            "text2latent": True,
            "mol2latent": True,
        }
    optimizer = optim.Adam(model_param_group, weight_decay=args.weight_decay)


    optimal_loss = sys.maxsize
    for epoch_id in range(args.epoch_num):
        epoch_loss = 0.0
        for i_batch, sample_batched in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            # load data from dataloader
            # SMILES_batched = sample_batched[0]
            description_batched = sample_batched[1]
            graph_batched = sample_batched[2].to(device)

            # s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
            # s_n_arr = s_n.numpy()  # .reshape((1, -1))
            # t_n_arr = t_n.numpy().reshape(-1)
            # s_n_text, t_n_text = [new_dict[i] for i in s_n_arr], [new_dict[j] for j in t_n_arr]
            # s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(
            #     t_n_text, context_length=args.context_length
            # ).to(device)
            # s_n, t_n = s_n.long().to(device), t_n.long().to(device)
            
            # forward and backward
            image_features, text_features = model(graph_batched, description_batched, device)
            # for contrastive learning loss isn't symmetric, we need to average the loss
            loss_01, acc_01 = cl_loss(text_features, image_features, args)
            loss_02, acc_02 = cl_loss(image_features, text_features, args)
            all_loss = (loss_01 + loss_02) / 2
            all_acc = (acc_01 + acc_02) / 2
            # node_loss = cal_cl_loss(s_image_features, s_text_features, labels)
            # gt_loss = cal_cl_loss(s_image_features, t_text_features, labels)
            # tt_loss = cal_cl_loss(s_text_features, t_text_features, labels)

            # all_loss = node_loss + args.edge_coef * gt_loss + args.edge_coef * tt_loss

            optimizer.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            optimizer.step()

            # information record and save model
            loss = round((all_loss.detach().clone()).cpu().item(), 4)
            if (epoch_id * len(train_loader) + i_batch) % args.log_freq == 0:
                logger.log("{} epoch {}th batch loss in :{}".format(epoch_id + 1, i_batch, loss / args.batch_size))
                writer.add_scalar("Train_Loss/batch", loss / args.batch_size, epoch_id * len(train_loader) + i_batch)
            if (epoch_id * len(train_loader) + i_batch) % args.save_freq == 0:
                model.save_model(model_save_dir, f"epoch{epoch_id}_batch{i_batch}", save_config)
            epoch_loss += loss / len(train_loader)

        logger.log("{}th epoch mean loss:{}".format(epoch_id + 1, epoch_loss))
        writer.add_scalar("Train_Loss/epoch", epoch_loss, epoch_id + 1)
        model.save_model(model_save_dir, f"epoch{epoch_id}", save_config)
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            if optimal_loss < args.loss_threshold:
                model.save_model(model_save_dir, "best", save_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # log config
    parser.add_argument("--time_log", type=bool, default=True)
    parser.add_argument("--log_freq", type=int, default=1000)
    # dataset config
    parser.add_argument("--data_dir", type=str, default="data/PubChemEdit")
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
    parser.add_argument("--text_lr", type=float, default=2e-5)
    parser.add_argument("--graph_lr", type=float, default=2e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1)
    parser.add_argument("--graph_lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    # model config
    parser.add_argument("--molecule_type", type=str, default="2DGraph", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.add_argument("--repr_frozen", dest='repr_frozen', action='store_true')
    parser.add_argument('--no_repr_frozen', dest='repr_frozen', action='store_false')
    parser.set_defaults(repr_frozen=False)
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
    parser.add_argument('--graph_pretrain_dir', type=str, default='ckpt/GraphMVP')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/mol_align")
    parser.add_argument("--loss_threshold", type=float, default=sys.maxsize)
    parser.add_argument("--save_freq", type=int, default=4000)
    # contrastive SSL config
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)


    # parser.add_argument("--edge_coef", type=float, default=10)
    # parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")


    # parser.add_argument("--neigh_num", type=int, default=3)
    # parser.add_argument("--context_length", type=int, default=128)
    # parser.add_argument("--embed_dim", type=int, default=128)
    # parser.add_argument("--transformer_heads", type=int, default=8)
    # parser.add_argument("--transformer_layers", type=int, default=12)
    # parser.add_argument("--transformer_width", type=int, default=512)
    # parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    # parser.add_argument("--num_nodes", type=int, default=1)
    # parser.add_argument("--gt_layers", type=int, default=3)
    # parser.add_argument("--att_d_model", type=int, default=128)
    # parser.add_argument("--att_norm", type=bool, default=True)
    # parser.add_argument("--head", type=int, default=8)
    # parser.add_argument("--if_pos", type=bool, default=False)

    args = parser.parse_args()

    start = time.perf_counter()
    main(args)
    

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
