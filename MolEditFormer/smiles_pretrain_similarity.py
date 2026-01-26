import os
import os.path as osp
import sys
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import random_split, DataLoader
from tensorboardX import SummaryWriter

from transformers import AutoModel, AutoTokenizer

from MolEditFormer.utils.basic_utils import get_local_time, seed_all, Logger
from MolEditFormer.utils.molecule_edit_utils import get_can_smiles
from MolEditFormer.models import MolEditFormer_pretrain
# from MolEditFormer.datasets import PubChemEdit, PubChemEdit_ZINC250k
from MolEditFormer.datasets import PubChemEdit_ZINC250k


# class epoch_based_WarmupCosineLR(_LRScheduler):
#     def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
#         self.warmup_epochs = warmup_epochs
#         self.total_epochs = total_epochs
#         super(epoch_based_WarmupCosineLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.warmup_epochs:
#             # 线性增加学习率
#             return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
#         else:
#             # 余弦退火学习率
#             return [
#                 base_lr * 0.5 * (1 + math.cos(
#                     math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
#                 )) for base_lr in self.base_lrs
#             ]
        

# class batch_based_WarmupCosineLR_step(_LRScheduler):
#     def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         super(batch_based_WarmupCosineLR_step, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         current_step = self.last_epoch + 1

#         if current_step <= self.warmup_steps:
#             # 线性增加学习率
#             return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
#         else:
#             # 余弦退火学习率
#             progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
#             return [base_lr * cosine_decay for base_lr in self.base_lrs]


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.dir_name == "":
        model_save_dir = osp.join(args.store_dir, f"{args.data_source}_{args.molecule_type}")
    else:
        model_save_dir = osp.join(args.store_dir, args.dir_name)
    os.makedirs(model_save_dir, exist_ok=True)
    logger = Logger(osp.join(model_save_dir, "log"), args.time_log)
    writer = SummaryWriter(osp.join(model_save_dir, "tensorboard"))

    CL_args = {
        "CL_emb_dim": args.SSL_emb_dim,
        "CL_loss": args.SSL_loss,
        "CL_neg_samples": args.CL_neg_samples,
        "T": args.T,
        "normalize": args.normalize,
        "mol2latent_path": None, 
        "text2latent_path": None,
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

    model = MolEditFormer_pretrain(
        mol_branch=args.mol_branch,
        text_branch=args.text_branch,
        mode=args.model_mode,
        device=device,
        mol_args=mol_args,
        text_args=text_args,
        CL_args=CL_args,
    ).to(device)

    model.eval()
    if args.mixed:
        mixed_config = {
            "can2can_ratio": args.can2can_ratio,
            "non2can_ratio": args.non2can_ratio,
            "non2non_ratio": args.non2non_ratio,
        }
        if mixed_config["non2can_ratio"] + mixed_config["non2non_ratio"]  + mixed_config["can2can_ratio"] != 1.0:
            raise ValueError("Invalid mixed config")
    else:
        mixed_config = None

    if args.validation_ratio == 0:
        train_set = PubChemEdit_ZINC250k(args.data_dir, value_type=args.value_type, property_type=args.property_type, mode=args.dataset_mode, mixed=args.mixed, mixed_config=mixed_config, scaffold_hint=args.scaffold_hint)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    elif 0 < args.validation_ratio < 1:
        dataset = PubChemEdit_ZINC250k(args.data_dir, value_type=args.value_type, property_type=args.property_type, mode=args.dataset_mode, mixed=args.mixed, mixed_config=mixed_config, scaffold_hint=args.scaffold_hint)
        dataset_size = len(dataset)
        val_size = int(dataset_size * args.validation_ratio)
        train_size = dataset_size - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise ValueError("Invalid validation ratio")

    corr_sim_list, non_corr_sim_list = [], []

    for i_batch, sample_batched in tqdm(enumerate(val_loader), disable=False, total=len(val_loader)):
        encoder_input_molecule_batched = sample_batched[0]
        decoder_input_molecule_batched = sample_batched[1]
        description_batched = sample_batched[2]

        batch_mol_latent = model.sample_mol_latent(encoder_input_molecule_batched)
        batch_text_latent = model.sample_text_latent(description_batched)

        # Calculate cosine similarity between corresponding and non-corresponding pairs
        batch_size = batch_mol_latent.shape[0]
        corresponding_similarities = []
        non_corresponding_similarities = []

        for i in range(batch_size):
            mol_latent = batch_mol_latent[i].unsqueeze(0)  # [1, dim]
            text_latent = batch_text_latent[i].unsqueeze(0)  # [1, dim]
            
            # Cosine similarity for corresponding pair
            sim_corresponding = F.cosine_similarity(mol_latent, text_latent).item()
            corresponding_similarities.append(sim_corresponding)
            
            # Cosine similarity for non-corresponding pairs
            for j in range(batch_size):
                if i != j:
                    sim_non_corresponding = F.cosine_similarity(mol_latent, batch_text_latent[j].unsqueeze(0)).item()
                    non_corresponding_similarities.append(sim_non_corresponding)
        
        corr_sim_list.extend(corresponding_similarities)
        non_corr_sim_list.extend(non_corresponding_similarities)

        if i_batch > 100:
            break


    # Calculate means
    corr_mean = np.mean(corr_sim_list)
    non_corr_mean = np.mean(non_corr_sim_list)
    
    print(f"Corresponding similarities mean: {corr_mean:.4f}")
    print(f"Non-corresponding similarities mean: {non_corr_mean:.4f}")
    
    plt.tight_layout()
    
    # Save corresponding pairs plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(corr_sim_list, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax1.set_title(f'aa (mean={corr_mean:.4f})')
    ax1.set_xlabel('aa')
    ax1.set_ylabel('aa')
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(osp.join("/data4/zengyu/AI4C/MolEditFormer/MolEditFormer", 'corresponding_similarity_distribution.pdf'), dpi=100)
    plt.close(fig1)
    
    # Save non-corresponding pairs plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(non_corr_sim_list, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_title(f'aa (mean={non_corr_mean:.4f})')
    ax2.set_xlabel('aa')
    ax2.set_ylabel('aa')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(osp.join("/data4/zengyu/AI4C/MolEditFormer/MolEditFormer", 'non_corresponding_similarity_distribution.pdf'), dpi=100)
    plt.close(fig2)


    # if args.warmup_choice == "no":
    #     pass
    # elif args.warmup_choice == "epoch":
    #     scheduler = epoch_based_WarmupCosineLR(optimizer, warmup_epochs=args.warmup_epoch, total_epochs=args.epoch_num)
    # elif args.warmup_choice == "batch":
    #     scheduler = batch_based_WarmupCosineLR_step(optimizer, warmup_steps=args.warmup_batch, total_steps=args.epoch_num * len(train_loader))
    # else:
    #     raise ValueError("Invalid warmup choice")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # log config
    parser.add_argument("--time_log", type=bool, default=True)
    parser.add_argument("--log_freq", type=int, default=1000)
    # dataset config
    parser.add_argument("--data_dir", type=str, default="data/PubChemEdit_ZINC250k")
    parser.add_argument("--dataset_mode", type=str, default="main", choices=["full", "main", "expand"])
    parser.add_argument("--value_type", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--property_type", type=str, default="name", choices=["name", "explanation"])
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--scaffold_hint", action="store_true")
    parser.add_argument("--can2can_ratio", type=float, default=1.0)
    parser.add_argument("--non2can_ratio", type=float, default=0.0)
    parser.add_argument("--non2non_ratio", type=float, default=0.0)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    # dataloader config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # train config
    parser.add_argument("--model_mode", type=str, default="pretrain", choices=["pretrain", "finetune", "reconstruct", "edit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    # parser.add_argument("--warmup_choice", type=str, default="no", choices=["no", "epoch", "batch"])
    parser.add_argument("--warmup_epoch", type=int, default=10, help="epoch start to warmup")
    parser.add_argument("--warmup_batch", type=int, default=5000, help="batch start to warmup")
    parser.add_argument("--epoch_num", type=int, default=32, help="epoch number")
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--graph_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1)
    parser.add_argument("--graph_lr_scale", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
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
    # projector config
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    # load config
    parser.add_argument('--text_tokenizer_dir', type=str, default='ckpt/SciBERT')
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--mol_model_path', type=str, default='ckpt/MegaMolBART/model_weight.pth')
    parser.add_argument('--text_projector_path', type=str, default=None)
    parser.add_argument('--mol_projector_path', type=str, default=None)
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolEditFormer/pretrain")
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
    # loss config
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()
    args.molecule_type = "SMILES"
    args.data_source = "PubChemEdit_ZINC250k"

    start = time.perf_counter()
    main(args)
    

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))