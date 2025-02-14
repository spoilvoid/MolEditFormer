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


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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