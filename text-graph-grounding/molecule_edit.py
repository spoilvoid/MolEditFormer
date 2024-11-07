import os
import os.path as osp
import sys
import math
import numpy as np
import argparse
from tqdm import tqdm
import time
import json

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from models import CLIP, MegaMolBART, MLP
from molecule_edit_utils import load_space_projector, get_edit_SMILES_list, get_edit_prompt, evaluate_SMILES_list
from basic_utils import get_local_time, seed_all, default_dump, Logger


# molecule_repr: [batch_size, d_model_joint], text_repr: [batch_size, d_model_joint]
# 这里batch_size为1，没有问题，否则可能产生问题，需要修改
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


# 这里为什么要mask反转，这样所有有效token不是mask被置为0了吗
# 结论，因为其mask表示的是0为有效，1为无效
# token_embeddings: [pad_len, batch_size, d_model_gen], attention_mask: [pad_len, batch_size]
def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    # input_mask_expanded: [batch_size, pad_len, d_model_gen]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)
    logger = Logger(osp.join(args.store_dir, "log"), time_log=False, log_name=f"{args.edit_task_id}")

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

    edit_SMILES_list = get_edit_SMILES_list(args)
    prompt = get_edit_prompt(args)
    result_dict = {}
    
    logger.log(f"edit task id: {args.edit_task_id}")
    logger.log(f"edit task description: {prompt}")
    result_dict[prompt] = {}
    print(f"edit task description: {prompt}")
    success_count = 0
    for smi in edit_SMILES_list:
        result_dict[prompt][smi] = {}
        print(f"edit input: {smi}")
        text_list = [prompt]
        text2joint_repr = text_branch_model.encode_text_from_pretrain_model(text_list, device)

        # 将输入SMILES在MegaMolBART中的latent作为被解码的latent
        # latent_code_init: [pad_len, batch_size, d_model_gen], pad_mask_init: [pad_len, batch_size]
        latent_code_init, pad_mask_init = gen_model_wrapper.smileslist2embedding([smi])  # [pad, B, d], 
        print(pad_mask_init)
        
        regenerated_mol = gen_model_wrapper.inverse_transform([latent_code_init], pad_mask_init.bool().cuda(), k=1, sanitize=True)[0]
        success_flag = False
        for l2_lambda in args.l2_lambda_list:
            print("l2 lambda: {}".format(l2_lambda))
            # 记录优化历程中的SMILES，第一个为输入SMILES，第二个为未经过latent optimization直接解码的SMILES，后面的为不同l2_lambda下进行学习后解码得到的SMILES
            current_SMILES_list = [smi, regenerated_mol]

            latent = latent_code_init.detach().clone()
            if args.init_noise:
                print("Use random noise for init")
                random_noise = torch.randn(latent_code_init.size()).to(device)
                latent += random_noise
            latent.requires_grad = True

            pad_mask = pad_mask_init.detach().clone()

            optimizer = optim.Adam([latent], lr=args.lr)

            for epoch_id in tqdm(range(args.epoch_num)):
                # 学习率在前段不变，后段呈现余弦退火
                t = epoch_id / args.epoch_num
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr

                latent2gen_repr = mean_pooling(latent, pad_mask) # [B, d]
                if args.normalize:
                    latent2gen_repr = F.normalize(latent2gen_repr, dim=-1)
                gen2joint_repr = gen2joint_projector(latent2gen_repr)

                clip_loss = clip_loss_for_edit(gen2joint_repr, text2joint_repr)
                loss = clip_loss + l2_lambda * nn.MSELoss()(latent_code_init, latent)
                # l2_loss_ =  l2_lambda * ((latent_code_init - latent) ** 2).mean()
                # loss = clip_loss_ + l2_loss_

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(F"final MSELoss: {loss.item()}")

            generated_mols = gen_model_wrapper.inverse_transform([latent], pad_mask.bool().cuda(), k=1, sanitize=True)
            current_SMILES_list.append(generated_mols[0])
            # evaluate_SMILES_list输入一个SMILES列表，返回一个bool列表，表示每个SMILES是否符合prompt的要求，输出是[True]或[False]
            current_result_list = evaluate_SMILES_list(current_SMILES_list, prompt)
            if current_result_list[0]:
                success_flag = True
            logger.log(f"input: {smi}, l2_lambda: {l2_lambda}, output: {current_SMILES_list[2]}, result: {current_result_list[0]}")
            result_dict[prompt][smi][l2_lambda] = {
                "output": current_SMILES_list[2],
                "result": current_result_list[0]
            }
            result_dict[prompt][smi][l2_lambda] = {
                "output": current_SMILES_list[2],
                "result": current_result_list[0]
            }
        if success_flag:
            success_count += 1
    result_dict[prompt]["success_count"] = success_count
    result_dict[prompt]["success_rate"] = success_count / len(edit_SMILES_list)

    if args.store_dir is not None:
        save_filename = f"{args.edit_task_id}_result.json"
        json_str = json.dumps(result_dict, ensure_ascii=False, default=default_dump)
        with open(os.path.join(args.store_dir, save_filename), 'w', encoding='utf-8') as file:
            file.write(json_str)
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # edit data config
    parser.add_argument('--l2_lambda_list', nargs='+', type=float, default=[1e1, 1e0, 1e-1, 1e-2, 1e-3])
    parser.add_argument("--edit_task_id", type=int, default=None)
    parser.add_argument("--edit_SMILES_filepath", type=str, default="data/EditBenchmark/edit_SMILES.txt")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--edit_SMILES", type=str, default=None)
    parser.add_argument("--edit_prompt", type=str, default=None)
    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    # model config
    parser.add_argument("--mol_branch", dest='mol_branch', action='store_true')
    parser.add_argument('--no_mol_branch', dest='mol_branch', action='store_false')
    parser.set_defaults(mol_branch=False)
    parser.add_argument("--text_branch", dest='text_branch', action='store_true')
    parser.add_argument('--no_text_branch', dest='text_branch', action='store_false')
    parser.set_defaults(text_branch=True)
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
    parser.add_argument("--resume", dest='resume', action='store_true')
    parser.add_argument('--no_resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    parser.add_argument('--text_pretrain_dir', type=str, default='ckpt/SciBERT')
    parser.add_argument('--text_model_path', type=str, default='ckpt/mol_align/text_model.pth')
    parser.add_argument('--text_projector_path', type=str, default='ckpt/mol_align/text_projector.pth')
    parser.add_argument('--gen2joint_projector_path', type=str, default='ckpt/mol_align/gen2joint_projector.pth')
    parser.add_argument('--joint2gen_projector_path', type=str, default='ckpt/mol_align/joint2gen_projector.pth')
    # save config
    parser.add_argument("--store_dir", type=str, default="ckpt/MolAlign/edit_2nd_step")
    # molecular edit task config
    parser.add_argument("--init_noise", dest="init_noise", action="store_true")
    parser.add_argument("--no_init_noise", dest="init_noise", action="store_false")
    parser.set_defaults(init_noise=False)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch_num", type=int, default=100)

    args = parser.parse_args()

    start = time.perf_counter()
    main(args)
    
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))