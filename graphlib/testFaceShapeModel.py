#%%
import torch
import tensorflow as tf
import numpy as np
import scipy.sparse
from lib import graph
from torch.nn.parameter import Parameter


def PoolwT(x, L):
    Mp = L.shape[0]
    N, M, Fin = x.shape
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))

    i = torch.LongTensor(np.transpose(indices))
    v = torch.FloatTensor(L.data)
    L = torch.sparse.FloatTensor(i, v, L.shape)

    x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
    x = x.view(M, Fin * N)  # M x Fin*N

    x = torch.spmm(L, x)  # Mp x Fin*N
    x = x.view(Mp, Fin, N)  # Mp x Fin x N
    x = x.permute(2, 0, 1).contiguous()  # N x Mp x Fin
    return x


def TensorPoolwT(x, L):
    
    Mp = L.shape[0]
    N, M, Fin = x.shape
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = L.tocoo()
    
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)

    x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
    x = tf.reshape(x, [M, Fin * N])  # M x Fin*N

    x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
    x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
    x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

    return x


def Chebyshev5(x, L, Fout, K):
    N, M, Fin = x.shape
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L, lmax=2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))

    i = torch.LongTensor(np.transpose(indices))
    v = torch.FloatTensor(L.data)
    L = torch.sparse.FloatTensor(i, v, L.shape)

    # Transform to Chebyshev basis
    x0 = x.permute(1, 2, 0)  # M x Fin x N
    x0 = x0.contiguous()
    x0 = x0.view(M, Fin * N)  # M x Fin*N
    x = x0.unsqueeze(0)  # 1 x M x Fin*N

    def concat(x, x_):
        x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
        return torch.cat((x, x_), 0)  # K x M x Fin*N

    if K > 1:
        x1 = torch.spmm(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * torch.spmm(L, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = x.view(K, M, Fin, N)  # K x M x Fin x N
    x = x.permute(3, 1, 2, 0)  # N x M x Fin x K
    x = x.contiguous()
    x = x.view(N * M, Fin * K)  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    # W = self._weight_variable([Fin * K, Fout], regularization=False)
    #W = Parameter(torch.FloatTensor(Fin * K, Fout))
    W = torch.tensor([[-6.2733e-09, 3.0666e-41, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, -0.6, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [0.11, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, -0.5, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, 0.4, 0.0000e+00]])
    x = torch.mm(x, W)  # N*M x Fout
    return x.view(N, M, Fout)  # N x M x Fout


def TensorChebyshev5(x, L, Fout, K):
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L, lmax=2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)
    # Transform to Chebyshev basis
    x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
    x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
        return tf.concat([x, x_], axis=0)  # K x M x Fin*N

    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
    x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
    x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    #W = weight_variable([Fin * K, Fout], regularization=False)
    W = torch.tensor([[-6.2733e-09, 3.0666e-41, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, -0.6, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [0.11, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, -0.5, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, 0.4, 0.0000e+00]])
    W = tf.convert_to_tensor(W.numpy())
    x = tf.matmul(x, W)  # N*M x Fout
    return tf.reshape(x, [N, M, Fout])  # N x M x Fout

#if __name__ == "__main__":
#%%
x = torch.tensor([[[23., 1., 4.], [6., 24., 6.], [2, 3, 5]],
                [[4., 8., 4.], [6., 24., 6.], [2, 3, 5]],
                [[7., 8., 9.], [4., 34., 86.], [2, 3, 5]]])
L = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 0., 3.]])
L = L.numpy()
Fout = 4
K = 2
#%%
xp = Chebyshev5(x, L, Fout, K)
xp = PoolwT(xp, L)
print(xp, xp.shape)

#%%
xt = x.numpy()
xt = tf.convert_to_tensor(xt)
xt = TensorChebyshev5(xt, L, Fout, K)
xt = TensorPoolwT(xt, L)
with tf.Session().as_default():
    print(xt.eval(), tf.shape(xt))

#%%
import torch
import numpy as np
from scipy.sparse import coo_matrix
w = np.array([0.4, 0.6])
coo = coo_matrix(([3.,4.,5., 6.], ([0,1,1, 2], [2,0,2,1])), shape=(3,3))
print(coo)

values = coo.data
print(coo.todense())
for i in range(coo.shape[0]):
    index = np.where(coo.row == i)[0]
    print(index)
    coo.data[index] = np.multiply(coo.data[index], w[:len(index)])
print(coo.todense())
#print(np.where(coo.row == 0), coo.col)
indices = np.vstack((coo.row, coo.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = coo.shape

torch.sparse.FloatTensor(i, v, torch.Size(shape))._indices()[0]
#%%
import torch
import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix

coo = coo_matrix(([3.,4.,5., 6.], ([0,1,1, 2], [2,0,2,1])), shape=(3,3))
adj = scipy.sparse.csr_matrix(coo)
print(adj)
adj = adj.tocoo()
indices = np.column_stack((adj.row, adj.col))

i = torch.LongTensor(np.transpose(indices))
v = torch.FloatTensor(adj.data)
adj = torch.sparse.FloatTensor(i, v, adj.shape)
adj2 = torch.ones((3, 3))
print(adj.shape)

#%%
import pickle
M, A, D, U = pickle.load(open("./LFW_ROOT/template_small.pkl", 'rb')) 
print(U[1].todense())