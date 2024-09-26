import os
import os.path as osp
import sys
import math
import numpy as np
import argparse
from tqdm import tqdm
import time
import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import CLIP, MegaMolBART, MLP
from molecule_edit_utils import load_space_projector, get_edit_SMILES_list, get_edit_prompt_list, evaluate_SMILES_list
from basic_utils import get_local_time, freeze_network, seed_all, Logger


def clip_loss_for_edit(molecule_repr, text_repr):
    molecule_repr = F.normalize(molecule_repr, dim=-1)
    text_repr = F.normalize(text_repr, dim=-1)

    similarity = -torch.mm(molecule_repr, text_repr.transpose(0, 1))[0]
    return similarity


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask

    
def check_edit(SMILES, text, l2_lambda_list, gen_model_wrapper, text_branch_model, gen2joint_projector, device):
    # 首先将输入的SMILES转化为文本-分子联合latent
    text_list = [text]
    text2joint_repr = text_branch_model.encode_text_from_pretrain_model(text_list, device)

    # 将输入SMILES在MegaMolBART中的latent作为被解码的latent
    latent_code_init, pad_mask_init = gen_model_wrapper.smileslist2embedding([SMILES])  # [pad, B, d], [pad, B]
    regenerated_mol = gen_model_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)[0]

    result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
    
    if args.use_noise_for_init:
        print("Use random noise for init")
        random_noise = torch.randn(latent_code_init.size()).to(device)
    
    for l2_lambda in l2_lambda_list:
        print("l2 lambda: {}".format(l2_lambda))
        # 记录优化历程中的SMILES，第一个为输入SMILES，第二个为未经过latent optimization直接解码的SMILES
        current_SMILES_list = [SMILES, regenerated_mol]
        if args.use_noise_for_init:
            print("Use random noise for init")
            latent = latent_code_init.detach().clone() + random_noise
        else:
            print("No random noise for init")
            latent = latent_code_init.detach().clone()
        pad_mask = pad_mask_init.detach().clone()
        latent.requires_grad = True
        optimizer = optim.Adam([latent], lr=args.lr)

        for i in tqdm(range(args.epochs)):
            t = i / args.epochs
            # 学习率在前段不变，后段呈现余弦退火
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            molecule_repr_generation = mean_pooling(latent, pad_mask) # [B, d]
            if args.normalize:
                molecule_repr_generation = F.normalize(molecule_repr_generation, dim=-1)
            gen2joint_repr = gen2joint_projector(molecule_repr_generation)

            clip_loss_ = clip_loss_for_edit(gen2joint_repr, text2joint_repr)
            l2_loss_ =  l2_lambda * ((latent_code_init - latent) ** 2).mean()

            loss = clip_loss_ + l2_loss_

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print("clip loss: {:.5f}\tL2 loss: {:.5f}".format(clip_loss_.item(), l2_loss_.item()))

        generated_mols = gen_model_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
        current_SMILES_list.append(generated_mols[0])
        result_SMILES_list_one_pair.append([text] + current_SMILES_list + ['{}'.format(l2_lambda)])

        current_result_list = evaluate_SMILES_list(current_SMILES_list, text)
        result_eval_list_one_pair.append(current_result_list)
        print()
    
    result_eval_list_one_pair = np.array(result_eval_list_one_pair)
    result_eval_list_one_pair = np.any(result_eval_list_one_pair, axis=0, keepdims=True)
    print("result_eval_list_one_pair\n", result_eval_list_one_pair)
    return result_SMILES_list_one_pair, result_eval_list_one_pair


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    logger = Logger(osp.join(args.store_dir, "log"), args.time_log)
    writer = SummaryWriter(osp.join(args.store_dir, "tensorboard"))

    # load model
    if args.gen_model == "MegaMolBART":
        gen_model_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.gen_model_dir, output_dir=None)
        print(f"Loading pretrained MegaMolBART from {args.gen_model_dir}.")
    else:
        raise NotImplementedError
    text_branch_model = CLIP(args)
    gen2joint_projector, joint2gen_projector= load_space_projector(args)

    gen_model_wrapper.model = gen_model_wrapper.model.to(device)
    text_branch_model = text_branch_model.to(device)
    gen2joint_projector.to(device)
    # joint2gen_projector.to(device)
    gen_model_wrapper.model.eval()
    text_branch_model.eval()
    gen2joint_projector.eval()
    # joint2gen_projector.eval()
    
    print("\n\n\nstart editing\n\n\n")

    source_SMILES_list = get_edit_SMILES_list(args)
    description_list = get_edit_prompt_list(args)

    for description in description_list:
        print(f"edit task description: {description}")
        result_SMILES_list, result_acc_list = [], []
        for SMILES in source_SMILES_list:
            print(f"edit input: {SMILES}")
            text_list = [description]
            text2joint_repr = text_branch_model.encode_text_from_pretrain_model(text_list, device)

            # 将输入SMILES在MegaMolBART中的latent作为被解码的latent
            latent_code_init, pad_mask_init = gen_model_wrapper.smileslist2embedding([SMILES])  # [pad, B, d], [pad, B]
            regenerated_mol = gen_model_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)[0]

            result_SMILES_list_one_pair, result_eval_list_one_pair = [], []
            
            for l2_lambda in args.l2_lambda_list:
                print("l2 lambda: {}".format(l2_lambda))
                # 记录优化历程中的SMILES，第一个为输入SMILES，第二个为未经过latent optimization直接解码的SMILES
                current_SMILES_list = [SMILES, regenerated_mol]

                latent = latent_code_init.detach().clone()
                if args.use_noise_for_init:
                    print("Use random noise for init")
                    random_noise = torch.randn(latent_code_init.size()).to(device)
                    latent += random_noise
                else:
                    print("No random noise for init")
                latent.requires_grad = True

                pad_mask = pad_mask_init.detach().clone()

                optimizer = optim.Adam([latent], lr=args.lr)

                for epoch_id in tqdm(range(args.epochs)):
                    # 学习率在前段不变，后段呈现余弦退火
                    t = epoch_id / args.epochs
                    lr = get_lr(t, args.lr)
                    optimizer.param_groups[0]["lr"] = lr

                    latent2gen_repr = mean_pooling(latent, pad_mask) # [B, d]
                    if args.normalize:
                        latent2gen_repr = F.normalize(latent2gen_repr, dim=-1)
                    gen2joint_repr = gen2joint_projector(latent2gen_repr)

                    clip_loss_ = clip_loss_for_edit(gen2joint_repr, text2joint_repr)
                    loss = clip_loss_ + l2_lambda * nn.MSELoss(latent_code_init, latent)
                    # l2_loss_ =  l2_lambda * ((latent_code_init - latent) ** 2).mean()
                    # loss = clip_loss_ + l2_loss_

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                print(F"final loss: {loss.item()}")

                generated_mols = gen_model_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
                current_SMILES_list.append(generated_mols[0])

                result_SMILES_list_one_pair.append([description] + current_SMILES_list + ['{}'.format(l2_lambda)])

                current_result_list = evaluate_SMILES_list(current_SMILES_list, description)
                result_eval_list_one_pair.append(current_result_list)
            
            result_eval_list_one_pair = np.array(result_eval_list_one_pair)
            result_eval_list_one_pair = np.any(result_eval_list_one_pair, axis=0, keepdims=True)
            print("result_eval_list_one_pair\n", result_eval_list_one_pair)


            result_SMILES_list_, result_acc_list_ = check_edit(SMILES, description, args.l2_lambda_list, gen_model_wrapper, text_branch_model, gen2joint_projector, device)
            result_SMILES_list.extend(result_SMILES_list_)
            result_acc_list.append(result_acc_list_)
            print("\n\n\n")
        
        result_acc_list = np.concatenate(result_acc_list, axis=0)
        result_acc_list = np.sum(result_acc_list, axis=0)
        result_acc_list = 100. * result_acc_list / len(source_SMILES_list)
        result_acc_row = '\t'.join(['{}'.format(x) for x in result_acc_list])
        print("===== Accuracy =====\t{}".format(result_acc_row))

        if args.store_dir is not None:
            saver_file = os.path.join(args.output_model_dir, "edited_SMILES.tsv")
            with open(saver_file, 'a') as f:
                for row in result_SMILES_list:
                    row = "\t".join(row)
                    print(row, file=f)

            saver_file = os.path.join(args.output_model_dir, "accuracy")
            np.savez(saver_file, result_acc_list)



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
    parser.add_argument("--weight_decay", type=float, default=0)
    # fixed generation model config
    parser.add_argument('--gen_model', type=str, default="MegaMolBART", choices=["MegaMolBART"])
    parser.add_argument("--vocab_path", type=str, default="bart_vocab.txt")
    parser.add_argument("--gen_emb_dim", type=int, default=256)
    # text branch config
    parser.add_argument("--text_emb_dim", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)
    # projector config
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    # load config
    parser.add_argument("--gen_model_dir", type=str, default="ckpt/MegaMolBART/checkpoints")
    parser.add_argument('--text_model_path', type=str, default='ckpt/mol_align/text_model.pth')
    parser.add_argument('--text_projector_path', type=str, default='ckpt/mol_align/text_projector.pth')
    parser.add_argument('--gen2joint_projector_path', type=str, default='ckpt/mol_align/gen2joint_projector.pth')
    parser.add_argument('--joint2gen_projector_path', type=str, default='ckpt/mol_align/joint2gen_projector.pth')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/edit_2nd_step")
    parser.add_argument("--save_freq", type=int, default=4000)
    # molecular edit task config
    parser.add_argument('--l2_lambda_list', nargs='+', type=float, default=[1e1, 1e0, 1e-1, 1e-2, 1e-3])
    parser.add_argument("--edit_task_id", type=int, default=None)
    parser.add_argument("--edit_SMILES_filepath", type=str, default="data/EditBenchmark/edit_SMILES.txt")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--input_SMILES", type=str, default=None)
    parser.add_argument("--input_description", type=str, default=None)
    parser.add_argument("--init_noise", dest="init_noise", action="store_true")
    parser.add_argument("--no_init_noise", dest="init_noise", action="store_false")
    parser.set_defaults(init_noise=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()

    main(args)