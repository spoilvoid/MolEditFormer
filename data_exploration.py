from collections import OrderedDict
from typing import Tuple, Union
from typing import Any, Union, List

import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as pyg_DataLoader

from transformers import AutoModel, AutoTokenizer

from datasets import PubChemEdit


data_dir = "data/PubChemEdit/version_0"
batch_size = 64
num_workers = 8
max_seq_len = 512
text_dim = 768
text_pretrain_dir = "ckpt/SciBERT"


def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(sentence, truncation=False, padding=False, return_tensors='np')
    # print(text_input)
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()
    return [input_ids, attention_mask]


def prepare_text_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    return tokens_ids, masks


dataset = PubChemEdit(data_dir)
dataloader = pyg_DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
text_tokenizer = AutoTokenizer.from_pretrained(text_pretrain_dir)


token_num_list = []
for batch in tqdm(dataloader):
    text = batch[1]
    description_tokens_ids, description_masks = prepare_text_tokens(description=text, tokenizer=text_tokenizer, max_seq_len=max_seq_len)
    for item in description_tokens_ids:
        token_num_list.append(len(item))
np.save("token_num_list.npy", token_num_list)