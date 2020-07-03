import numpy as np
from random import choice
from random import shuffle
from pykeops.torch import generic_argkmin
import torch.nn as nn
import torch
import math
from collections import deque
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def move_i_first(index, i):
    if (index == i).nonzero()[..., 0].shape[0]:
        inx = (index == i).nonzero()[0][0]
    else:
        index[-1] = i
        inx = (index == i).nonzero()[0][0]
    if inx > 1:
        index[1:inx+1], index[0] = index[0:inx].clone(), index[inx].clone() 
    else:
        index[inx], index[0] = index[0].clone(), index[inx].clone() 
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx, is_L=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sp.csr_matrix(sparse_mx)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    if is_L:
        sparse_mx = rescale_L(sparse_mx, lmax=2)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adj(A):
    
    kernal_size = 9
    A_temp = []
    for x in A:
        x.data = np.ones(x.data.shape)
        # build symmetric adjacency matrix
        x = x + x.T.multiply(x.T > x) - x.multiply(x.T > x)
        #x = x + sp.eye(x.shape[0])
        A_temp.append(x.astype('float32'))
    A_temp = [normalize(x) for x in A_temp]
    A = [sparse_mx_to_torch_sparse_tensor(x) for x in A_temp]

    Adj = []
    for adj in A:
        index_list = []
        for i in range(adj.shape[0]): #
            index = (adj._indices()[0] == i).nonzero().squeeze()
            if index.dim() == 0:
                index = index.unsqueeze(0)
            index1 = torch.index_select(adj._indices()[1], 0, index[:kernal_size-1])
            #index1 = move_i_first(index1, i)
            index_list.append(index1)
        index_list.append(torch.zeros(kernal_size-1, dtype=torch.int64)-1)
        index_list = torch.stack([torch.cat([i, i.new_zeros(
            kernal_size - 1 - i.size(0))-1], 0) for inx, i in enumerate(index_list)], 0)
        Adj.append(index_list)
    return Adj
