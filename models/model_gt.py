from collections import OrderedDict
from typing import Tuple, Union

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union, List
from transformers import AutoModel, AutoTokenizer

from graph_model import GNN, GNN_graphpred
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from model_utils import pad_array_1d, LayerNorm, QuickGELU

_tokenizer = _Tokenizer()


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("c_fc", nn.Linear(d_model, d_model * 4)),
#                     ("gelu", QuickGELU()),
#                     ("c_proj", nn.Linear(d_model * 4, d_model)),
#                 ]
#             )
#         )
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x


# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()

        # self.context_length = args.context_length
        self.args = args

        if not args.mol_branch and not args.text_branch:
            raise ValueError("At least one of the branches should be enabled")
        
        # load molecule branch
        if args.mol_branch:
            if args.molecule_type not in ["2DGraph", "3DGraph", "SMILES", "all"]:
                raise ValueError("Invalid molecule type")
            
            if args.molecule_type == "2DGraph" or args.molecule_type == "all":
                self.molecule_dim = args.gnn_emb_dim

                self.molecule_node_model = GNN(
                    num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
                    JK=args.JK, drop_ratio=args.dropout_ratio,
                    gnn_type=args.gnn_type)
                self.molecule_model = GNN_graphpred(
                    num_layer=args.num_layer,
                    emb_dim=args.gnn_emb_dim,
                    JK=args.JK,
                    graph_pooling=args.graph_pooling,
                    num_tasks=1,
                    molecule_node_model=self.molecule_node_model)
            if args.molecule_type == "3DGraph" or args.molecule_type == "all":
                pass
            if args.molecule_type == "SMILES" or args.molecule_type == "all":
                pass
            # load molecule projector
            self.mol2latent = nn.Linear(self.molecule_dim, args.SSL_emb_dim)
            # load molecule branch weight
            if args.resume:
                state_dict = torch.load(args.mol_model_path, map_location='cpu')
                self.molecule_model.load_state_dict(state_dict)
                state_dict = torch.load(args.mol_projector_path, map_location='cpu')
                self.mol2latent.load_state_dict(state_dict)
            else:
                pretrained_graph_path = osp.join(args.mol_pretrain_dir, args.pretrain_gnn_mode, "model.pth")
                self.molecule_model.from_pretrained(pretrained_graph_path)

        # load text branch
        if args.text_branch:
            self.max_seq_len = args.max_seq_len
            self.text_dim = args.text_emb_dim
            self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_pretrain_dir)
            self.text_model = AutoModel.from_pretrained(args.text_pretrain_dir)
            # load text projector
            self.text2latent = nn.Linear(self.text_dim, args.SSL_emb_dim)
            # load text branch weight
            if args.resume:
                state_dict = torch.load(args.text_model_path, map_location='cpu')
                self.text_model.load_state_dict(state_dict)
                state_dict = torch.load(args.text_projector_path, map_location='cpu')
                self.text2latent.load_state_dict(state_dict)

    def preprocess_each_sentence(self, sentence, tokenizer, max_seq_len):
        text_input = tokenizer(
            sentence, truncation=True, max_length=max_seq_len,
            padding='max_length', return_tensors='np')
        # print(text_input)
        input_ids = text_input['input_ids'].squeeze()
        attention_mask = text_input['attention_mask'].squeeze()

        sentence_tokens_ids = pad_array_1d(input_ids, max_seq_len)
        sentence_masks = pad_array_1d(attention_mask, max_seq_len)
        return [sentence_tokens_ids, sentence_masks]

    def prepare_text_tokens(self, device, description, tokenizer, max_seq_len):
        B = len(description)
        tokens_outputs = [self.preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
        tokens_ids = [o[0] for o in tokens_outputs]
        masks = [o[1] for o in tokens_outputs]
        tokens_ids = torch.Tensor(tokens_ids).long().to(device)
        masks = torch.Tensor(masks).bool().to(device)
        return tokens_ids, masks

    def encode_graph(self, molecule_data):
        if not self.args.mol_branch:
            raise ValueError("molecule branch should be enabled")
        molecule_repr, _ = self.molecule_model(molecule_data)
        molecule_repr = self.mol2latent(molecule_repr)
        return molecule_repr

    def encode_text_from_pretrain_model(self, text, device):
        if not self.args.text_branch:
            raise ValueError("text branch should be enabled")
        description_tokens_ids, description_masks = self.prepare_text_tokens(
            device,
            description=text,
            tokenizer=self.text_tokenizer,
            max_seq_len=self.max_seq_len
        )
        description_output = self.text_model(input_ids=description_tokens_ids, attention_mask=description_masks)
        description_repr = description_output["pooler_output"]
        description_repr = self.text2latent(description_repr)
        return description_repr

    def forward(self, molecule_data, text, device):  # g, s_n, t_n, s_n_text, t_n_text
        if not (self.args.mol_branch and self.args.text_branch):
            raise ValueError("text branch and molecule branch should both be enabled")
        elif self.args.mol_branch and not self.args.text_branch:
            raise ValueError("text branch should be enabled")
        elif not self.args.mol_branch and self.args.text_branch:
            raise ValueError("molecule branch should be enabled")
        
        s_image_features = self.encode_graph(molecule_data)

        text_features = self.encode_text_from_pretrain_model(text, device)

        return s_image_features, text_features

        # return s_image_features, s_text_features, t_text_features, labels

    def save_model(self, save_dir, prefix="", config=None):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if config is None or not isinstance(config, dict):
            print("Please provide the config file for saving the model")
            return
        for key, value in config.items():
            if value:
                model_branch = getattr(self, key, None)
                if model_branch is None:
                    print(f"Model branch {key} does not exist")
                    continue
                torch.save(model_branch.state_dict(), osp.join(save_dir, f"{prefix}_{key}.pth"))


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
