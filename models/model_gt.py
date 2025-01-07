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
from . import graph_transformer
from . import GNN, GNN_graphpred
from . import SimpleTokenizer as _Tokenizer
from .model_utils import cycle_index, mean_pooling
from .mega_molbart.mega_mol_bart import MegaMolBART

_tokenizer = _Tokenizer()


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=value)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()

        # self.context_length = args.context_length
        self.args = args
        # self.edge_coef = args.edge_coef
        # self.text_pretrain_folder = args.text_pretrain_folder

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
                self.molecule_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=args.mol_pretrain_dir, output_dir=None)
                self.molecule_dim = args.smiles_emb_dim
                self.molecule_model = self.molecule_wrapper.model
                self.molecule_tokenizer = self.molecule_wrapper.tokenizer
            # load molecule projector
            self.mol2latent = nn.Linear(self.molecule_dim, args.SSL_emb_dim)
            # load molecule branch weight
            if args.resume:
                state_dict = torch.load(args.mol_model_path, map_location='cpu')
                self.molecule_model.load_state_dict(state_dict)
                state_dict = torch.load(args.mol_projector_path, map_location='cpu')
                self.mol2latent.load_state_dict(state_dict)
            else:
                if args.molecule_type == "2DGraph" or args.molecule_type == "all":
                    pretrained_graph_path = osp.join(args.mol_pretrain_dir, args.pretrain_gnn_mode, "model.pth")
                    self.molecule_model.from_pretrained(pretrained_graph_path)
                if args.molecule_type == "3DGraph" or args.molecule_type == "all":
                    pass
                if args.molecule_type == "SMILES" or args.molecule_type == "all":
                    pass

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

    def prepare_text_tokens(self, batch_text, device):
        text_input = self.text_tokenizer(batch_text, truncation=True, max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
        tokens_ids = text_input['input_ids'].long().to(device)
        pad_mask = text_input['attention_mask'].bool().to(device)
        return tokens_ids, pad_mask

    def prepare_smiles_tokens(self, batch_smiles, device):
        smiles_input = self.molecule_tokenizer.tokenize(batch_smiles, pad=True)
        token_ids = torch.tensor(self.molecule_tokenizer.convert_tokens_to_ids(smiles_input['original_tokens'])).long().to(device).T
        pad_mask = torch.tensor(smiles_input['masked_pad_masks']).bool().to(device).T
        token_ids = token_ids[:self.molecule_wrapper.max_model_position_embeddings]
        pad_mask = pad_mask[:self.molecule_wrapper.max_model_position_embeddings]
        return token_ids, pad_mask

    def encode_graph(self, batch_graph):
        if not self.args.mol_branch:
            raise ValueError("molecule branch should be enabled")
        molecule_embedding, _ = self.molecule_model(batch_graph)
        return molecule_embedding

    def encode_smiles(self, smiles_token_ids, smiles_mask):
        encode_input = {"encoder_input": smiles_token_ids, "encoder_pad_mask": smiles_mask}
        molecule_embedding = self.molecule_model.encode(encode_input)
        return molecule_embedding

    def encode_text_from_pretrain_model(self, text_token_ids, text_mask):
        if not self.args.text_branch:
            raise ValueError("text branch should be enabled")
        description_output = self.text_model(input_ids=text_token_ids, attention_mask=text_mask)
        description_embedding = description_output["last_hidden_state"]
        description_pooled_latent = description_output["pooler_output"]
        return description_embedding, description_pooled_latent

    def decode_smiles(self, decoder_input, decoder_mask, memory):
        decode_input = {"decoder_input": decoder_input, "decoder_pad_mask": decoder_mask, "memory_input": memory,"memory_pad_mask": decoder_mask}
        decode_output = self.molecule_model.decode(decode_input) # logits
        return decode_output

    def forward(self, batch_molecule, batch_text, device):
        if not (self.args.mol_branch and self.args.text_branch):
            raise ValueError("text branch and molecule branch should both be enabled")
        elif self.args.mol_branch and not self.args.text_branch:
            raise ValueError("text branch should be enabled")
        elif not self.args.mol_branch and self.args.text_branch:
            raise ValueError("molecule branch should be enabled")
        # text_token_ids, text_mask: [batch_size, text_max_seq_len]
        text_token_ids, text_mask = self.prepare_text_tokens(batch_text, device)
        # text_embedding: [batch_size, text_max_seq_len, text_d_model]
        # text_pooled_embedding: [batch_size, text_d_model]
        text_embedding, text_pooled_embedding = self.encode_text_from_pretrain_model(text_token_ids, text_mask)  
        # text_repr: [batch_size, text_d_model]
        text_repr = mean_pooling(text_embedding.transpose(0, 1), text_mask.transpose(0, 1))
        # text_latent: [batch_size, SSL_emb_dim]
        text_latent = self.text2latent(text_repr)

        if self.args.molecule_type == "SMILES":
            # molecule_token_ids, molecule_mask: [mol_max_seq_len, batch_size]
            molecule_token_ids, molecule_mask = self.prepare_smiles_tokens(batch_molecule, device)
            # molecule_embedding: [mol_max_seq_len, batch_size, mol_d_model]
            molecule_embedding = self.encode_smiles(molecule_token_ids, molecule_mask)
            # molecule_repr: [batch_size, mol_d_model]
            molecule_repr = mean_pooling(molecule_embedding, ~molecule_mask)
            # decode_logits: [mol_max_seq_len, batch_size, vocab_size]
            decode_logits = self.decode_smiles(molecule_token_ids, molecule_mask, molecule_embedding)
        elif self.args.molecule_type == "2DGraph":
            molecule_repr = self.encode_graph(batch_molecule)
        # molecule_latent: [batch_size, SSL_emb_dim]
        molecule_latent = self.mol2latent(molecule_repr)

        cl_loss, mask_loss = self._calc_loss(molecule_latent, text_latent, molecule_token_ids, molecule_mask, decode_logits)
        return cl_loss, mask_loss

    def _calc_loss(self, molecule_latent, text_latent, smiles_token_ids=None, smiles_mask=None, decoder_output=None):
        cl_loss = (self._calc_cl_loss(molecule_latent, text_latent) + self._calc_cl_loss(text_latent, molecule_latent)) / 2
        if smiles_token_ids is not None and smiles_mask is not None and decoder_output is not None:
            mask_loss = self._calc_mask_loss(smiles_token_ids, smiles_mask, decoder_output)
            return cl_loss, mask_loss
        else:
            return cl_loss, None
    
    def _calc_cl_loss(self, latent_1, latent_2):
        '''
        latent_1 [batch_size, SSL_emb_dim]: molecular features or text features 
        latent_2 [batch_size, SSL_emb_dim]: molecular features or text features 
        '''
        if self.args.normalize:
            latent_1 = F.normalize(latent_1, dim=-1)
            latent_2 = F.normalize(latent_2, dim=-1)

        if self.args.SSL_loss == 'EBM_NCE':
            criterion = nn.BCEWithLogitsLoss()
            # use cycle_index to form k negative samples
            # neg_latent_1 [args.CL_neg_samples * batch_size, SSL_emb_dim]: negative molecular features or text features 
            # neg_latent_2 [args.CL_neg_samples * batch_size, SSL_emb_dim]: negative molecular features or text features 
            neg_latent_2 = torch.cat([latent_2[cycle_index(len(latent_2), i + 1)] for i in range(self.args.CL_neg_samples)], dim=0)
            neg_latent_1 = latent_1.repeat((self.args.CL_neg_samples, 1))
            # calculate the cosine similarity for each sample
            # 这里由于组播的原理这里是逐项相乘，这里sum后得到对应分子-文本对的余弦相似度，再除以温度参数
            pred_pos = torch.sum(latent_1 * latent_2, dim=1) / self.args.T
            pred_neg = torch.sum(neg_latent_1 * neg_latent_2, dim=1) / self.args.T
            # calculate the contrastive learning loss according to the weighted sum
            loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
            loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
            CL_loss = (loss_pos + self.args.CL_neg_samples * loss_neg) / (1 + self.args.CL_neg_samples)
        elif self.args.SSL_loss == 'InfoNCE':
            criterion = nn.CrossEntropyLoss()
            # suppose data in mini_batch should own different labels
            B = latent_1.size()[0]
            # calculate logits by integrating text and structure features for each sample
            logits = torch.mm(latent_1, latent_2.transpose(1, 0))  # B*B
            logits = torch.div(logits, self.args.T)
            labels = torch.arange(B).long().to(logits.device)  # B*1
            CL_loss = criterion(logits, labels)
        else:
            raise Exception
        return CL_loss

    def _calc_mask_loss(self, smiles_token_ids, smiles_mask, decoder_output):
        """ Calculate the loss for the token prediction task

        Args:
            smiles_token_ids (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokenizer
            smiles_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens
            decoder_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """  
        pad_token_idx = self.molecule_wrapper.tokenizer.vocab[self.molecule_wrapper.tokenizer.pad_token]
        mask_loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_idx)

        (seq_len, batch_size) = tuple(smiles_token_ids.size())
        token_pred = decoder_output.reshape((seq_len * batch_size,
                -1)).float()
        loss = mask_loss_func(token_pred, smiles_token_ids.reshape(-1)).reshape((seq_len, batch_size))
        inv_target_mask = ~(smiles_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model, should contain batch['encoder_input] meaning Smiles token_ids and batch['encoder_pad_mask'] meaning Smiles pad_mask
            sampling_alg (str): Algorithm to use to sample SMILES strings from model, choice = ['greedy', 'beam']

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """
        if self.args.molecule_type == "2DGraph" or self.args.molecule_type == "all":
            pass
        if self.args.molecule_type == "3DGraph" or self.args.molecule_type == "all":
            pass
        if self.args.molecule_type == "SMILES" or self.args.molecule_type == "all":
            mol_strs, log_lhs = self.molecule_model.sample_molecules(batch_input=batch_input, sampling_alg=sampling_alg)
        return mol_strs, log_lhs

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
