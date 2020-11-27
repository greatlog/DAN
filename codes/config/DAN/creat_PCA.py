import numpy as np
import torch
import sys

sys.path.insert(0, "../../")
import utils as util

batch_ker = util.random_batch_kernel(
    batch=30000,
    l=21,
    sig_min=0.6,
    sig_max=5,
    rate_iso=0,
    scaling=3,
    tensor=False,
    random_disturb=True,
)
print("batch kernel shape: {}".format(batch_ker.shape))
b = np.size(batch_ker, 0)
batch_ker = batch_ker.reshape((b, -1))
pca_matrix = util.PCA(batch_ker, k=opt["code_length"]).float()
print("PCA matrix shape: {}".format(pca_matrix.shape))
torch.save(pca_matrix, "./pca_matrix.pth")
print("Save PCA matrix at: ./pca_matrix.pth")
