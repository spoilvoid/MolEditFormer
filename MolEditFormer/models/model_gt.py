from collections import OrderedDict
from typing import Tuple, Union

import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Any, Union, List
from transformers import AutoModel, AutoTokenizer

from MolEditFormer.models.model_utils import cycle_index, pad_tensor, mean_pooling, load_mega_mol_bart, ArgsContainer
from MolEditFormer.models import GNN, GNN_graphpred


class MolTextFuser(nn.Module):
    def __init__(
        self, 
        mol_dim: int, 
        text_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            mol_dim: Dimension of molecule embedding
            text_dim: Dimension of text embedding
            num_layers: Number of stacked cross-attention layers and LayerNorm
            num_heads: Number of attention heads
            dropout: Dropout probability applied within MultiheadAttention
        """
        super(MolTextFuser, self).__init__()
        self.mol_dim = mol_dim
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.text2mol_adapter = nn.Linear(text_dim, mol_dim)
        self.cross_attn_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=mol_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(mol_dim) for _ in range(num_layers)])

    def forward(self, mol_embedding, text_embedding, text_mask):
        """
        Args:
            mol_embedding: [mol_max_seq_len, batch_size, mol_dim]
            text_embedding: [text_max_seq_len, batch_size, text_dim]
            text_mask: [batch_size, text_max_seq_len]

        Returns:
            mol_embedding: [mol_max_seq_len, batch_size, mol_dim]
        """
        # adapted_text_embedding: [text_max_seq_len, batch_size, mol_dim]
        adapted_text_embedding = self.text2mol_adapter(text_embedding)
        for i in range(self.num_layers):
            out, _ = self.cross_attn_layers[i](query=mol_embedding, key=adapted_text_embedding, value=adapted_text_embedding, key_padding_mask=text_mask)
            mol_embedding = self.norm_layers[i](mol_embedding + out)

        return mol_embedding


class CLIP(nn.Module):
    MOLECULE_TYPE_RANGE = ["2DGraph", "3DGraph", "SMILES", "all"]
    GRAPH2D_MOL_ARGS_RANGE = ["molecule_type", "gnn_emb_dim", "num_layer", "JK", "dropout_ratio", "gnn_type", "graph_pooling", "model_path"]
    GRAPH3D_MOL_ARGS_RANGE = ["molecule_type", "model_path"]
    SMILES_MOL_ARGS_RANGE = ["molecule_type", "smiles_emb_dim", "vocab_path", "model_path"]
    TEXT_ARGS_RANGE = ["text_emb_dim", "max_seq_len", "tokenizer_dir", "model_path"]
    CL_ARGS_RANGE = ["CL_emb_dim", "CL_loss", "CL_neg_samples", "T", "normalize", "mol2latent_path", "text2latent_path"]
    FUSE_ARGS_RANGE = ["num_layers", "num_heads", "dropout", "model_path"]

    def __init__(
        self, 
        mol_branch: bool,
        text_branch: bool,
        mode: str,
        device: torch.device,
        mol_args: dict = None,
        text_args: dict = None,
        CL_args: dict = None,
        fuse_args: dict = None,
    ):
        super().__init__()

        self.mol_branch = mol_branch
        self.mol_args = ArgsContainer(**(mol_args or {}))
        self.text_branch = text_branch
        self.text_args = ArgsContainer(**(text_args or {}))
        self.mode = mode
        self.device = device
        self.CL_args = ArgsContainer(**(CL_args or {}))
        self.fuse_args = ArgsContainer(**(fuse_args or {}))

        # check args validity
        self._args_check()
        
        # load molecule branch
        if self.mol_branch:
            if self.mol_args.molecule_type in ["2DGraph", "all"]:
                self.molecule_dim = self.mol_args.gnn_emb_dim
                self.molecule_node_model = GNN(
                    num_layer=self.mol_args.num_layer, emb_dim=self.mol_args.gnn_emb_dim,
                    JK=self.mol_args.JK, drop_ratio=self.mol_args.dropout_ratio,
                    gnn_type=self.mol_args.gnn_type)
                self.molecule_model = GNN_graphpred(
                    num_layer=self.mol_args.num_layer,
                    emb_dim=self.mol_args.gnn_emb_dim,
                    JK=self.mol_args.JK,
                    graph_pooling=self.mol_args.graph_pooling,
                    num_tasks=1,
                    molecule_node_model=self.molecule_node_model)
                if self.mol_args.model_path is not None:
                    state_dict = torch.load(self.mol_args.model_path, map_location='cpu')
                    self.molecule_model.load_state_dict(state_dict)
            if self.mol_args.molecule_type in ["3DGraph", "all"]:
                pass
            if self.mol_args.molecule_type in ["SMILES", "all"]:
                self.molecule_dim = self.mol_args.smiles_emb_dim
                self.molecule_model, self.molecule_tokenizer = load_mega_mol_bart(self.mol_args.model_path, self.mol_args.vocab_path)

        # load text branch
        if self.text_branch:
            self.max_seq_len = self.text_args.max_seq_len
            self.text_dim = self.text_args.text_emb_dim
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_args.tokenizer_dir)
            self.text_model = AutoModel.from_pretrained(self.text_args.tokenizer_dir)
            if self.text_args.model_path is not None:
                state_dict = torch.load(self.text_args.model_path, map_location='cpu')
                self.text_model.load_state_dict(state_dict)

        # load projector
        if self.mode == "pretrain":
            self.mol2latent = nn.Linear(self.molecule_dim, self.CL_args.CL_emb_dim)
            self.text2latent = nn.Linear(self.text_dim, self.CL_args.CL_emb_dim)
            if self.CL_args.mol2latent_path is not None:
                state_dict = torch.load(self.CL_args.mol2latent_path, map_location='cpu')
                self.mol2latent.load_state_dict(state_dict)
            if self.CL_args.text2latent_path is not None:
                state_dict = torch.load(self.CL_args.text2latent_path, map_location='cpu')
                self.text2latent.load_state_dict(state_dict)
        elif self.mode == "finetune":
            self.modality_fuser = MolTextFuser(
                mol_dim=self.molecule_dim, text_dim=self.text_dim, 
                num_layers=self.fuse_args.num_layers, num_heads=self.fuse_args.num_heads,
                dropout=self.fuse_args.dropout)
            if self.fuse_args.model_path is not None:
                state_dict = torch.load(self.fuse_args.model_path, map_location='cpu')
                self.modality_fuser.load_state_dict(state_dict)

        # change model's device
        self.to(self.device)

    def _args_check(self):
        # args check
        if self.mode == 'pretrain':
            if not self.mol_branch:
                raise ValueError("molecule branch should be enabled")
            elif self.mol_args is None:
                raise ValueError("mol_args should be provided")
            elif "molecule_type" not in self.mol_args.keys() or self.mol_args.molecule_type not in self.MOLECULE_TYPE_RANGE:
                raise ValueError("mol_args should contain valid molecule_type")
            elif self.mol_args.molecule_type in ["2DGraph", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.GRAPH2D_MOL_ARGS_RANGE):
                raise ValueError(f"2DGraph mol_args should at least contain {self.GRAPH2D_MOL_ARGS_RANGE}")
            elif self.mol_args.molecule_type in ["3DGraph", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.GRAPH3D_MOL_ARGS_RANGE):
                raise ValueError(f"3DGraph mol_args should at least contain {self.GRAPH3D_MOL_ARGS_RANGE}")
            elif self.mol_args.molecule_type in ["SMILES", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.SMILES_MOL_ARGS_RANGE):
                raise ValueError(f"SMILES mol_args should at least contain {self.SMILES_MOL_ARGS_RANGE}")
            
            if not self.text_branch:
                raise ValueError("text branch should be enabled")
            elif self.text_args is None:
                raise ValueError("text_args should be provided")
            elif any(arg_name not in self.text_args.keys() for arg_name in self.TEXT_ARGS_RANGE):
                raise ValueError(f"text_args should at least contain {self.TEXT_ARGS_RANGE}")
            
            if self.CL_args is None:
                raise ValueError("CL_args should be provided")
            elif any(arg_name not in self.CL_args.keys() for arg_name in self.CL_ARGS_RANGE):
                raise ValueError(f"CL_args should at least contain {self.CL_ARGS_RANGE}")
            
        elif self.mode == 'finetune':
            if not self.mol_branch:
                raise ValueError("molecule branch should be enabled")
            elif self.mol_args is None:
                raise ValueError("mol_args should be provided")
            elif "molecule_type" not in self.mol_args.keys() or self.mol_args.molecule_type not in self.MOLECULE_TYPE_RANGE:
                raise ValueError("mol_args should contain valid molecule_type")
            elif self.mol_args.molecule_type in ["2DGraph", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.GRAPH2D_MOL_ARGS_RANGE):
                raise ValueError(f"2DGraph mol_args should at least contain {self.GRAPH2D_MOL_ARGS_RANGE}")
            elif self.mol_args.molecule_type in ["3DGraph", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.GRAPH3D_MOL_ARGS_RANGE):
                raise ValueError(f"3DGraph mol_args should at least contain {self.GRAPH3D_MOL_ARGS_RANGE}")
            elif self.mol_args.molecule_type in ["SMILES", "all"] and any(arg_name not in self.mol_args.keys() for arg_name in self.SMILES_MOL_ARGS_RANGE):
                raise ValueError(f"SMILES mol_args should at least contain {self.SMILES_MOL_ARGS_RANGE}")
            
            if not self.text_branch:
                raise ValueError("text branch should be enabled")
            elif self.text_args is None:
                raise ValueError("text_args should be provided")
            elif any(arg_name not in self.text_args.keys() for arg_name in self.TEXT_ARGS_RANGE):
                raise ValueError(f"text_args should at least contain {self.TEXT_ARGS_RANGE}")
            
            if self.fuse_args is None:
                raise ValueError("fuse_args should be provided")
            elif any(arg_name not in self.fuse_args.keys() for arg_name in self.FUSE_ARGS_RANGE):
                raise ValueError(f"fuse_args should at least contain {self.FUSE_ARGS_RANGE}")
            
        elif self.mode == 'inference':
            pass

        if not self.mol_branch and not self.text_branch:
            raise ValueError("At least one of the branches should be enabled")
        
    def prepare_text_tokens(self, batch_text):
        text_input = self.text_tokenizer(batch_text, truncation=True, max_length=self.text_args.max_seq_len, padding='max_length', return_tensors='pt')
        tokens_ids = text_input['input_ids'].long().to(self.device)
        pad_mask = text_input['attention_mask'].bool().to(self.device)
        return tokens_ids, pad_mask

    def prepare_smiles_tokens(self, batch_smiles):
        smiles_input = self.molecule_tokenizer.tokenize(batch_smiles, pad=True)
        token_ids = torch.tensor(self.molecule_tokenizer.convert_tokens_to_ids(smiles_input['original_tokens'])).long().to(self.device).T
        pad_mask = torch.tensor(smiles_input['masked_pad_masks']).bool().to(self.device).T

        token_ids = pad_tensor(token_ids, dim=0, pad_size=self.molecule_model.sampler.max_seq_len, pad_value=self.molecule_tokenizer.vocab[self.molecule_tokenizer.pad_token])
        pad_mask = pad_tensor(pad_mask, dim=0, pad_size=self.molecule_model.sampler.max_seq_len, pad_value=True)
        token_ids = token_ids[:self.molecule_model.sampler.max_seq_len]
        pad_mask = pad_mask[:self.molecule_model.sampler.max_seq_len]
        return token_ids, pad_mask

    def encode_graph(self, batch_graph):
        if not self.mol_branch:
            raise ValueError("molecule branch should be enabled")
        molecule_embedding, _ = self.molecule_model(batch_graph)
        return molecule_embedding

    def encode_smiles(self, smiles_token_ids, smiles_mask):
        encode_input = {"encoder_input": smiles_token_ids, "encoder_pad_mask": smiles_mask}
        molecule_embedding = self.molecule_model.encode(encode_input)
        return molecule_embedding

    def encode_text_from_pretrain_model(self, text_token_ids, text_mask):
        if not self.text_branch:
            raise ValueError("text branch should be enabled")
        description_output = self.text_model(input_ids=text_token_ids, attention_mask=text_mask)
        description_embedding = description_output["last_hidden_state"]
        description_pooled_latent = description_output["pooler_output"]
        return description_embedding, description_pooled_latent

    def decode_smiles(self, decoder_input, decoder_mask, memory):
        decode_input = {"decoder_input": decoder_input, "decoder_pad_mask": decoder_mask, "memory_input": memory,"memory_pad_mask": decoder_mask}
        decode_output = self.molecule_model.decode(decode_input) # logits
        return decode_output

    def forward(self, batch_input_molecule, batch_input_text, batch_output_molecule=None, batch_output_text=None):
        if not (self.mol_branch and self.text_branch):
            missing = (["molecule"] if not self.mol_branch else []) + (["text"] if not self.text_branch else [])
            raise ValueError(f"{' and '.join(missing)} branch{'es' if len(missing) > 1 else ''} should be enabled")
        elif self.mode == "finetune" and batch_output_molecule is None:
            raise ValueError("batch_output_molecule should be provided in finetune mode")
        
        # input_text_token_ids, input_text_mask: [batch_size, text_max_seq_len]
        input_text_token_ids, input_text_mask = self.prepare_text_tokens(batch_input_text)
        # text_embedding: [batch_size, text_max_seq_len, text_d_model]
        # text_pooled_embedding: [batch_size, text_d_model]
        text_embedding, text_pooled_embedding = self.encode_text_from_pretrain_model(input_text_token_ids, input_text_mask)

        if self.mol_args.molecule_type in ["SMILES", "all"]:
            # input_molecule_token_ids, molecule_mask: [mol_max_seq_len, batch_size]
            input_molecule_token_ids, input_molecule_mask = self.prepare_smiles_tokens(batch_input_molecule)
            # molecule_embedding: [mol_max_seq_len, batch_size, mol_d_model]
            molecule_embedding = self.encode_smiles(input_molecule_token_ids, input_molecule_mask)
        elif self.mol_args.molecule_type in ["2DGraph", "all"]:
            # molecule_repr = self.encode_graph(batch_molecule)
            pass
        elif self.mol_args.molecule_type in ["3DGraph", "all"]:
            pass
        
        if self.mode == "pretrain":
            output_molecule_token_ids, output_molecule_mask = input_molecule_token_ids, input_molecule_mask
            # text_repr: [batch_size, text_d_model]
            text_repr = mean_pooling(text_embedding.transpose(0, 1), input_text_mask.transpose(0, 1))
            # text_latent: [batch_size, SSL_emb_dim]
            text_latent = self.text2latent(text_repr)
            # molecule_repr: [batch_size, mol_d_model]
            molecule_repr = mean_pooling(molecule_embedding, ~input_molecule_mask)
            # molecule_latent: [batch_size, SSL_emb_dim]
            molecule_latent = self.mol2latent(molecule_repr)
            if self.mol_args.molecule_type in ["SMILES", "all"]:
                # decode_logits: [mol_max_seq_len, batch_size, vocab_size]
                decode_logits = self.decode_smiles(input_molecule_token_ids, input_molecule_mask, molecule_embedding)
            elif self.mol_args.molecule_type in ["2DGraph", "all"]:
                # molecule_repr = self.encode_graph(batch_molecule)
                pass
            elif self.mol_args.molecule_type in ["3DGraph", "all"]:
                pass
            cl_loss, mask_loss = self._calc_pretrain_loss(molecule_latent, text_latent, output_molecule_token_ids, output_molecule_mask, decode_logits)

        elif self.mode == "finetune":
            fused_molecule_embedding = self.modality_fuser(molecule_embedding, text_embedding.transpose(0, 1), input_text_mask)
            if self.mol_args.molecule_type in ["SMILES", "all"]:
                # output_molecule_token_ids, output_molecule_mask: [mol_max_seq_len, batch_size]
                output_molecule_token_ids, output_molecule_mask = self.prepare_smiles_tokens(batch_output_molecule)
                # decode_logits: [mol_max_seq_len, batch_size, vocab_size]
                decode_logits = self.decode_smiles(output_molecule_token_ids, output_molecule_mask, fused_molecule_embedding)
            elif self.mol_args.molecule_type in ["2DGraph", "all"]:
                # molecule_repr = self.encode_graph(batch_molecule)
                pass
            elif self.mol_args.molecule_type in ["3DGraph", "all"]:
                pass
            mask_loss = self._calc_finetune_loss(output_molecule_token_ids, output_molecule_mask, decode_logits)
            cl_loss = None
        return cl_loss, mask_loss

    def _calc_pretrain_loss(self, molecule_latent, text_latent, smiles_token_ids, smiles_mask, decoder_output):
        cl_loss = (self._calc_cl_loss(molecule_latent, text_latent) + self._calc_cl_loss(text_latent, molecule_latent)) / 2
        mask_loss = self._calc_mask_loss(smiles_token_ids, smiles_mask, decoder_output)
        return cl_loss, mask_loss

    def _calc_finetune_loss(self, smiles_token_ids, smiles_mask, decoder_output):
        mask_loss = self._calc_mask_loss(smiles_token_ids, smiles_mask, decoder_output)
        return mask_loss

    def _calc_cl_loss(self, latent_1, latent_2):
        '''
        latent_1 [batch_size, SSL_emb_dim]: molecular features or text features 
        latent_2 [batch_size, SSL_emb_dim]: molecular features or text features 
        '''
        if self.CL_args.normalize:
            latent_1 = F.normalize(latent_1, dim=-1)
            latent_2 = F.normalize(latent_2, dim=-1)

        if self.CL_args.CL_loss == 'EBM_NCE':
            criterion = nn.BCEWithLogitsLoss()
            # use cycle_index to form k negative samples
            # neg_latent_1 [CL_neg_samples * batch_size, SSL_emb_dim]: negative molecular features or text features 
            # neg_latent_2 [CL_neg_samples * batch_size, SSL_emb_dim]: negative molecular features or text features 
            neg_latent_2 = torch.cat([latent_2[cycle_index(len(latent_2), i + 1)] for i in range(self.CL_args.CL_neg_samples)], dim=0)
            neg_latent_1 = latent_1.repeat((self.CL_args.CL_neg_samples, 1))
            # calculate the cosine similarity for each sample
            # 这里由于组播的原理这里是逐项相乘，这里sum后得到对应分子-文本对的余弦相似度，再除以温度参数
            pred_pos = torch.sum(latent_1 * latent_2, dim=1) / self.CL_args.T
            pred_neg = torch.sum(neg_latent_1 * neg_latent_2, dim=1) / self.CL_args.T
            # calculate the contrastive learning loss according to the weighted sum
            loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
            loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
            CL_loss = (loss_pos + self.CL_args.CL_neg_samples * loss_neg) / (1 + self.CL_args.CL_neg_samples)
        elif self.CL_args.CL_loss == 'InfoNCE':
            criterion = nn.CrossEntropyLoss()
            # suppose data in mini_batch should own different labels
            B = latent_1.size()[0]
            # calculate logits by integrating text and structure features for each sample
            logits = torch.mm(latent_1, latent_2.transpose(1, 0))  # B*B
            logits = torch.div(logits, self.CL_args.T)
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
        pad_token_idx = self.molecule_tokenizer.vocab[self.molecule_tokenizer.pad_token]
        mask_loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_idx)

        (seq_len, batch_size) = tuple(smiles_token_ids.size())
        token_pred = decoder_output.reshape((seq_len * batch_size, -1)).float()
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
        if self.mol_args.molecule_type in ["2DGraph", "all"]:
            pass
        if self.mol_args.molecule_type in ["3DGraph", "all"]:
            pass
        if self.mol_args.molecule_type in ["SMILES", "all"]:
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