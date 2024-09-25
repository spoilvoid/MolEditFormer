import os
import os.path as osp
import logging
import datetime
import sys
import random
import numpy as np
import torch


# for language model
def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


# for language model
def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


# for language model
def prepare_text_tokens(device, description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(tokens_ids).long().to(device)
    masks = torch.Tensor(masks).bool().to(device)
    return tokens_ids, masks


def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# def get_num_task_and_type(dataset):
#     if dataset in ["esol", "freesolv", "lipophilicity"]:
#         return 1, "regression"
#     elif dataset in ["hiv", "bace", "bbbp"]:
#         return 1, "classification"
#     elif dataset == "tox21":
#         return 12, "classification"
#     elif dataset == "pcba":
#         return 92, "classification"
#     elif dataset == "muv":
#         return 17, "classification"
#     elif dataset == "toxcast":
#         return 617, "classification"
#     elif dataset == "sider":
#         return 27, "classification"
#     elif dataset == "clintox":
#         return 2, "classification"
#     raise ValueError("Invalid dataset name.")


def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')


class Logger(object):
    def __init__(self, save_dir, time_log, log_name='train_logger'):
        print(f"initialize Logger file in {save_dir}")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)

        if time_log:
            log_file = logging.FileHandler(osp.join(save_dir, f"{log_name}_{get_local_time()}.txt"))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            log_file.setFormatter(formatter)
            self.logger.addHandler(log_file)
            
        self.log(f"PID: {os.getpid()}")

        # log command that runs the code
        s = ""
        for arg in sys.argv:
            s = s + arg + " "
        self.log(osp.basename(sys.executable) + " " + s)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)