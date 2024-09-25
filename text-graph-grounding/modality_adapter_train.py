import argparse
import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from basic_utils import get_mol_to_joint_latent, freeze_network
from models import MegaMolBART, MLP
from molecule_edit_utils import load_CLIP_graph_branch
from datasets import ZINC250K_Graph

from basic_utils import Logger, seed_all


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

    elif args.SSL_loss == 'RR':
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


def get_molecule_repr_generation(molecule_data, molecule_model, molecule_type="MegaMolBART", MegaMolBART_wrapper=None):
    if molecule_type == "MegaMolBART":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model, molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = mean_pooling(embedding, pad_mask)
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    return molecule_repr


def save_model(save_best, epoch=None):
    if args.output_model_dir is not None:
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        saved_file_path = os.path.join(args.output_model_dir, "generation2MoleculeSTM_{}".format(model_file))
        torch.save(generation2MoleculeSTM.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(args.output_model_dir, "MoleculeSTM2generation_{}".format(model_file))
        torch.save(MoleculeSTM2generation.state_dict(), saved_file_path)
    return


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load model
    if args.generation_model == "MegaMolBART":
        MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.MegaMolBART_generation_model_dir, output_dir=None)
        print("Loading from pretrained MegaMolBART ({}).".format(args.MegaMolBART_generation_model_dir))
        generation_model_dim = 256
    else:
        raise NotImplementedError
    
    graph_branch, graph_projector = load_CLIP_graph_branch(args)
    graph_branch_dim = args.gnn_emb_dim

    MegaMolBART_wrapper = MegaMolBART_wrapper.to(device)
    graph_branch = graph_branch.to(device)
    graph_projector = graph_projector.to(device)
    freeze_network(MegaMolBART_wrapper.model)
    freeze_network(graph_branch)
    freeze_network(graph_projector)
    MegaMolBART_wrapper.model.eval()
    graph_branch.eval()
    graph_projector.eval()

    # load dataset
    dataset = ZINC250K_Graph(args.data_dir)
    dataloader = pyg_DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    graph_branch_dim = args.gnn_emb_dim
    gen2joint_projector = MLP(generation_model_dim, [graph_branch_dim, graph_branch_dim]).to(device)
    joint2gen_projector = MLP(graph_branch_dim, [generation_model_dim, generation_model_dim]).to(device)

    model_param_group = [
        {"params": gen2joint_projector.parameters(), "lr": args.gen2joint_lr},
        {"params": joint2gen_projector.parameters(), "lr": args.joint2gen_lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10
    
    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        train(e)


    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    for batch in L:
        if args.MoleculeSTM_molecule_type == "SMILES":
            SMILES_list = batch
        else:
            SMILES_list, graph = batch
            graph = graph.to(device)

        if args.MoleculeSTM_molecule_type == "SMILES":
            molecule_repr_MoleculeSTM = get_mol_to_joint_latent(
                SMILES_list, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
            )
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

        else:
            molecule_repr_MoleculeSTM = get_mol_to_joint_latent(
                graph, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=None
            )
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

        if args.generation_model == "MegaMolBART":
            molecule_repr_generation = get_molecule_repr_generation(
                SMILES_list, molecule_model=molecule_model_generation,
                molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
            )
        else:  # for HierVAE
            hiervae_data_list = MolGraph.tensorize(SMILES_list, vocab, avocab)
            molecule_repr_generation = molecule_model_generation.forward_MoleculeSTM(hiervae_data_list)
        molecule_repr_generation2MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

        loss_01, acc_01 = do_CL(molecule_repr_generation, molecule_repr_MoleculeSTM2generation, args)
        loss_02, acc_02 = do_CL(molecule_repr_MoleculeSTM, molecule_repr_generation2MoleculeSTM, args)
        loss = (loss_01 + loss_02) / 2
        acc = (acc_01 + acc_02) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc

    accum_loss /= len(L)
    accum_acc /= len(L)
    
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True, epoch=epoch)
    print("SSL Loss: {:.5f}\tSSL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return


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
    parser.add_argument("--gen2joint_lr", type=float, default=1e-4)
    parser.add_argument("--joint2gen_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    # model config
    parser.add_argument("--molecule_type", type=str, default="2DGraph", choices=["2DGraph", "3DGraph", "SMILES", "all"])
    parser.add_argument("--repr_frozen", dest='repr_frozen', action='store_true')
    parser.add_argument('--no_repr_frozen', dest='repr_frozen', action='store_false')
    parser.set_defaults(repr_frozen=False)
    # fixed generation model config
    parser.add_argument('--generation_model', type=str, default="MegaMolBART", choices=["MegaMolBART"])
    parser.add_argument("--vocab_path", type=str, default="bart_vocab.txt")
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
    parser.add_argument("--generation_model_dir", type=str, default="ckpt/MegaMolBART/checkpoints")
    parser.add_argument('--graph_model_path', type=str, default='ckpt/GraphMVP')
    parser.add_argument('--graph_projector_path', type=str, default='ckpt/GraphMVP')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/mol_align")
    parser.add_argument("--loss_threshold", type=float, default=sys.maxsize)
    parser.add_argument("--save_freq", type=int, default=4000)
    # contrastive SSL config
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE", "MSELoss"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    print(args)

    start = time.perf_counter()
    main(args)
    
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
