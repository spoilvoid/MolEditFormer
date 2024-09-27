
import os
import os.path as osp
import sys
import argparse
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer

from models import CLIP, MegaMolBART, MLP
from molecule_edit_utils import load_space_projector
from datasets import ZINC250K_Graph

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


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


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

    elif args.SSL_loss == 'MSELoss':
        criterion = nn.MSELoss()
        CL_loss = criterion(X, Y)
        CL_acc = 0

    else:
        raise Exception

    return CL_loss, CL_acc


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def save_model(save_dir, prefix="", gen2joint_projector=None, joint2gen_projector=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    if gen2joint_projector is not None:
        torch.save(gen2joint_projector.state_dict(), osp.join(save_dir, f"{prefix}_gen2joint_projector.pth"))
    if joint2gen_projector is not None:
        torch.save(joint2gen_projector.state_dict(), osp.join(save_dir, f"{prefix}_joint2gen_projector.pth"))


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_save_dir = osp.join(args.store_dir, f"{args.molecule_type}_{args.gnn_type}_lr{args.gen2joint_lr}-{get_local_time()}")
    logger = Logger(osp.join(model_save_dir, "log"), args.time_log)
    writer = SummaryWriter(osp.join(model_save_dir, "tensorboard"))
    # load model
    if args.gen_model == "MegaMolBART":
        gen_model_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.gen_model_dir, output_dir=None)
        print(f"Loading pretrained MegaMolBART from {args.gen_model_dir}.")
    else:
        raise NotImplementedError
    mol_branch_model = CLIP(args)
    gen2joint_projector, joint2gen_projector= load_space_projector(args)

    gen_model_wrapper.model = gen_model_wrapper.model.to(device)
    mol_branch_model = mol_branch_model.to(device)
    gen2joint_projector = gen2joint_projector.to(device)
    joint2gen_projector = joint2gen_projector.to(device)
    freeze_network(gen_model_wrapper.model)
    freeze_network(mol_branch_model)
    gen_model_wrapper.model.eval()
    mol_branch_model.eval()
    gen2joint_projector.train()
    joint2gen_projector.train()

    # load dataset
    if args.molecule_type not in ["2DGraph", "3DGraph", "SMILES", "all"]:
        raise ValueError("Invalid molecule type")
    trainset = ZINC250K_Graph(args.data_dir)
    train_loader = pyg_DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    logger.log(f"gen2joint_lr: {args.gen2joint_lr}, joint2gen_lr: {args.joint2gen_lr}")
    model_param_group = [
        {"params": gen2joint_projector.parameters(), "lr": args.gen2joint_lr},
        {"params": joint2gen_projector.parameters(), "lr": args.joint2gen_lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.weight_decay)

    optimal_loss = sys.maxsize
    for epoch_id in range(args.epoch_num):
        epoch_loss = 0.0
        for i_batch, sample_batched in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            # forward the generation model to get the molecule representation in multi-modality model's joint space
            SMILES_batched = sample_batched[0]
            gen_model_embedding, gen_model_pad_mask = gen_model_wrapper.smileslist2embedding(SMILES_batched)
            gen_repr = mean_pooling(gen_model_embedding, gen_model_pad_mask)
            gen2joint_repr = gen2joint_projector(gen_repr)
            # forward the molecule branch to get the molecule representation in multi-modality model's joint space
            if args.molecule_type == "2DGraph" or args.molecule_type == "all":
                graph_batched = sample_batched[1].to(device)
                joint_repr = mol_branch_model.encode_graph(graph_batched)
            if args.molecule_type == "3DGraph" or args.molecule_type == "all":
                pass
            if args.molecule_type == "SMILES" or args.molecule_type == "all":
                joint_repr = mol_branch_model.encode_graph(SMILES_batched)
            joint2gen_repr = joint2gen_projector(joint_repr)

            loss_1, _ = cl_loss(joint_repr, gen2joint_repr, args)
            loss_2, _ = cl_loss(gen_repr, joint2gen_repr, args)

            loss = (loss_1 + loss_2) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch_id * len(train_loader) + i_batch) % args.log_freq == 0:
                logger.log("{} epoch {}th batch loss in :{}".format(epoch_id + 1, i_batch, loss))
                writer.add_scalar("Train_Loss/batch", loss, epoch_id * len(train_loader) + i_batch)
            # if (epoch_id * len(train_loader) + i_batch) % args.save_freq == 0:
            #     save_model(model_save_dir, f"epoch{epoch_id}_batch{i_batch}", gen2joint_projector, joint2gen_projector)
            epoch_loss += loss / len(train_loader)

        logger.log("{}th epoch mean loss:{}".format(epoch_id + 1, epoch_loss))
        writer.add_scalar("Train_Loss/epoch", epoch_loss, epoch_id + 1)
        if (epoch_id + 1) % args.save_freq == 0:
            save_model(model_save_dir, f"epoch{epoch_id}", gen2joint_projector, joint2gen_projector)
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            save_model(model_save_dir, "best", gen2joint_projector, joint2gen_projector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # log config
    parser.add_argument("--time_log", type=bool, default=True)
    parser.add_argument("--log_freq", type=int, default=1000)
    # dataset config
    parser.add_argument("--data_dir", type=str, default="data/ZINC250k")
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epoch_num", type=int, default=100, help="epoch number")
    parser.add_argument("--gen2joint_lr", type=float, default=1e-3)
    parser.add_argument("--joint2gen_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    # model config
    parser.add_argument("--molecule_type", type=str, default="2DGraph", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.add_argument("--mol_branch", dest='mol_branch', action='store_true')
    parser.add_argument('--no_mol_branch', dest='mol_branch', action='store_false')
    parser.set_defaults(mol_branch=True)
    parser.add_argument("--text_branch", dest='text_branch', action='store_true')
    parser.add_argument('--no_text_branch', dest='text_branch', action='store_false')
    parser.set_defaults(text_branch=False)
    # fixed generation model config
    parser.add_argument('--gen_model', type=str, default="MegaMolBART", choices=["MegaMolBART"])
    parser.add_argument("--vocab_path", type=str, default="bart_vocab.txt")
    parser.add_argument("--gen_emb_dim", type=int, default=256)
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
    parser.add_argument("--gen_model_dir", type=str, default="ckpt/MegaMolBART/checkpoints")
    parser.add_argument("--resume", dest='resume', action='store_true')
    parser.add_argument('--no_resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    parser.add_argument('--mol_pretrain_dir', type=str, default='ckpt/GraphMVP')
    parser.add_argument('--mol_model_path', type=str, default='ckpt/mol_align/mol_model.pth')
    parser.add_argument('--mol_projector_path', type=str, default='ckpt/mol_align/mol_projector.pth')
    parser.add_argument('--gen2joint_projector_path', type=str, default='ckpt/mol_align/gen2joint_projector.pth')
    parser.add_argument('--joint2gen_projector_path', type=str, default='ckpt/mol_align/joint2gen_projector.pth')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/edit_1st_step")
    parser.add_argument("--save_freq", type=int, default=10, help="according to epoch")
    # contrastive SSL config
    parser.add_argument("--SSL_loss", type=str, default="MSELoss", choices=["EBM_NCE", "InfoNCE", "MSELoss"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    start = time.perf_counter()
    main(args)
    
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
