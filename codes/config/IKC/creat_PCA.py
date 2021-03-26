import argparse
import sys

import numpy as np
import torch

import options as option
from utils import util

parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/train/SFTMD/train_SFTMD_x4.yml",
    help="Path to options YMAL file.",
)
args = parser.parse_args()
opt_F = option.parse(args.opt, is_train=True)

batch_ker = util.random_batch_kernel(
    batch=30000,
    l=opt_F["kernel_size"],
    sig_min=opt_F["sig_min"],
    sig_max=opt_F["sig_max"],
    rate_iso=1.0,
    tensor=False,
)
print("batch kernel shape: {}".format(batch_ker.shape))
b = np.size(batch_ker, 0)
batch_ker = batch_ker.reshape((b, -1))
pca_matrix = util.PCA(batch_ker, k=opt_F["code_length"]).float()

torch.save(pca_matrix, "pca_matrix.pth")
print("Save PCA matrix at: pca_matrix.pth")
