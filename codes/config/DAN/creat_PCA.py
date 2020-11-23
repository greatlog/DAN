import os
import math
import argparse
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import options.options as option
from models import create_model
from IPython import embed

import sys

sys.path.insert(0, "../../")
from data.data_sampler import DistIterSampler
from utils import util
from data import create_dataloader, create_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, help="Path to option YMAL file of SFTMD_Net.")
parser.add_argument(
    "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)

# convert to NoneDict, which returns None for missing keys
opt = option.dict_to_nonedict(opt)


batch_ker = util.random_batch_kernel(
    batch=30000, l=21, sig_min=0.6, sig_max=5, rate_iso=0, scaling=3, tensor=False, random_disturb=True
)
print("batch kernel shape: {}".format(batch_ker.shape))
b = np.size(batch_ker, 0)
batch_ker = batch_ker.reshape((b, -1))
pca_matrix = util.PCA(batch_ker, k=opt["code_length"]).float()
print("PCA matrix shape: {}".format(pca_matrix.shape))
torch.save(pca_matrix, "./pca_matrix.pth")
print("Save PCA matrix at: ./pca_matrix.pth")
