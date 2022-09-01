# Source:   https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
#           https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
# Modified by Yulin Shen.
# Modification highlights:
#   - Inputs flattening
#   - Inputs clipping using tanh()
#   - Avoid division by 0 in calculation of kernel_val in gaussian_kernel()

import torch


__all__ = [
    "MMD",
    "MMD_single_batch",
]

def _gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.tensor(0.0).to(a.device)
    a = a.view(a.shape[0], -1)
    b = b.view(b.shape[0], -1)
    return _gaussian_kernel(a, a).mean() + _gaussian_kernel(b, b).mean() - 2*_gaussian_kernel(a, b).mean()

#################################################################

def _gaussian_kernel_single_batch(a, b):
    assert a.shape==b.shape
    depth = a.shape[1]
    numerator = (a - b).pow(2).mean(1)/depth
    return torch.exp(-numerator)


def MMD_single_batch(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.tensor(0.0).to(a.device)
    a = a.view(a.shape[0], -1)
    b = b.view(b.shape[0], -1)
    return _gaussian_kernel_single_batch(a, a).mean() + _gaussian_kernel_single_batch(b, b).mean() - 2*_gaussian_kernel_single_batch(a, b).mean()
