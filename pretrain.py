import os
import os.path as osp
import sys
import math
import time
import argparse
import numpy as np
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

from models import CLIP, tokenize
from datasets import PubChemEdit, MolPair_SingleGraph , DataHelper, MolGraphDataset

from basic_utils import get_local_time, freeze_network, seed_all, Logger


class epoch_based_WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(epoch_based_WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火学习率
            return [
                base_lr * 0.5 * (1 + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                )) for base_lr in self.base_lrs
            ]
        

class batch_based_WarmupCosineLR_step(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(batch_based_WarmupCosineLR_step, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1

        if current_step <= self.warmup_steps:
            # 线性增加学习率
            return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # 余弦退火学习率
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]


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
        # neg_X [args.CL_neg_samples * batch_size, SSL_emb_dim]: negative molecular features
        # neg_Y [args.CL_neg_samples * batch_size, SSL_emb_dim]: negative description text features 
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        # calculate the cosine similarity for each sample
        # 这里由于组播的原理这里是逐项相乘，这里sum后得到对应分子-文本对的余弦相似度，再除以温度参数
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
        # 在InfoNCE中，假设每个样本都有一个唯一的标签
        # 认为文本特征向量与分子特征向量的点积如果来自于同一化学分子，那么结果应为1，否则为0，将一个模态A向量与所有模态B向量点积后，经过归一化，可以称作logits
        # 这里的温度系数在进行归一化后应该没有任何作用
        criterion = nn.CrossEntropyLoss()
        # B: batch_size
        B = X.size()[0]
        # calculate logits by integrating text and structure features for each sample
        logits = torch.mm(X, Y.transpose(1, 0))  # Batch_size * Batch_size
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
    if args.dir_name == "":
        if args.warmup_choice == "epoch":
            model_save_dir = osp.join(args.store_dir, f"{args.data_source}_{args.molecule_type}_{args.gnn_type}_warmup_{args.warmup_choice}{args.warmup_epoch}")
        else:
            model_save_dir = osp.join(args.store_dir, f"{args.data_source}_{args.molecule_type}_{args.gnn_type}_warmup_{args.warmup_choice}{args.warmup_batch}")
    else:
        model_save_dir = osp.join(args.store_dir, args.dir_name)
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logger = Logger(osp.join(model_save_dir, "log"), args.time_log)
    writer = SummaryWriter(osp.join(model_save_dir, "tensorboard"))

    model = CLIP(args).to(device)
    model.train()
    # dataset = DataHelper(arr_edge_index, args)
    # in_g = Data(x=node_f, edge_index=edge_index).to(device)
    # dataset = MolGraphDataset(args.graph_root)

    if args.data_source == "PubChemEdit":
        train_set = PubChemEdit(args.data_dir, mode=args.mode, can_smiles=args.can_smiles)
    elif args.data_source == "MolPair":
        train_set = MolPair_SingleGraph(args.data_dir)
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
    if args.warmup_choice == "no":
        pass
    elif args.warmup_choice == "epoch":
        scheduler = epoch_based_WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epoch, total_epochs=args.epoch_num)
    elif args.warmup_choice == "batch":
        scheduler = batch_based_WarmupCosineLR_step(optimizer, warmup_steps=args.warmup_batch, total_steps=args.epoch_num * len(train_loader))
    else:
        raise ValueError("Invalid warmup choice")

    optimal_loss = args.loss_threshold
    for epoch_id in range(args.start_epoch, args.epoch_num):
        epoch_loss = 0.0
        for i_batch, sample_batched in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
            # load data from dataloader
            if args.molecule_type not in ["2DGraph", "3DGraph", "SMILES", "all"]:
                raise ValueError("Invalid molecule type")
            
            if args.molecule_type == "2DGraph" or args.molecule_type == "all":
                graph_batched = sample_batched[2].to(device)
            if args.molecule_type == "3DGraph" or args.molecule_type == "all":
                pass
            if args.molecule_type == "SMILES" or args.molecule_type == "all":
                SMILES_batched = sample_batched[0]

            description_batched = sample_batched[1]

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
            if args.warmup_choice == "batch":
                scheduler.step()

            # information record and save model
            loss = round((all_loss.detach().clone()).cpu().item(), 4)
            if (epoch_id * len(train_loader) + i_batch + 1) % args.log_freq == 0:
                logger.log("{} epoch {}th batch loss in :{}".format(epoch_id + 1, i_batch + 1, loss))
                writer.add_scalar("Train_Loss/batch", loss, epoch_id * len(train_loader) + i_batch + 1)
                if loss < optimal_loss:
                    model.save_model(model_save_dir, f"epoch{epoch_id}_batch{i_batch+1}", save_config)
            epoch_loss += loss / len(train_loader)
        
        if args.warmup_choice == "epoch":
            scheduler.step()

        logger.log("{}th epoch mean loss:{}".format(epoch_id + 1, epoch_loss))
        writer.add_scalar("Train_Loss/epoch", epoch_loss, epoch_id + 1)
        model.save_model(model_save_dir, f"epoch{epoch_id}", save_config)
        if epoch_loss < optimal_loss:
            optimal_loss = epoch_loss
            model.save_model(model_save_dir, "best", save_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # log config
    parser.add_argument("--time_log", type=bool, default=True)
    parser.add_argument("--log_freq", type=int, default=1000)
    # dataset config
    parser.add_argument("--data_source", type=str, default="PubChemEdit", choices=["PubChemEdit", "MolPair"])
    parser.add_argument("--data_dir", type=str, default="data/PubChemEdit/version_0")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "main", "expand"])
    parser.add_argument("--can_smiles", action="store_true")
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--warmup_choice", type=str, default="no", choices=["no", "epoch", "batch"])
    parser.add_argument("--warmup_epoch", type=int, default=10, help="epoch start to warmup")
    parser.add_argument("--warmup_batch", type=int, default=5000, help="batch start to warmup")
    parser.add_argument("--epoch_num", type=int, default=32, help="epoch number")
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--graph_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1)
    parser.add_argument("--graph_lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    # model config
    parser.add_argument("--molecule_type", type=str, default="2DGraph", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.add_argument("--repr_frozen", dest='repr_frozen', action='store_true')
    parser.add_argument('--no_repr_frozen', dest='repr_frozen', action='store_false')
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
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/pretrain")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--save_freq", type=int, default=4000)
    parser.add_argument("--loss_threshold", type=float, default=sys.maxsize)
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
