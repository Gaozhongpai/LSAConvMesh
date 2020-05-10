import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        N, M, Fin = x.shape
        #print(adj.to_dense())
        x = x.permute(1, 2, 0).contiguous()  # M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N
        x = torch.spmm(adj, x)  # M x Fin*N
        x = x.view(M, Fin, N)  # M x Fin x N

        x = x.permute(2, 0, 1).contiguous()   # N x M x Fin
        x = x.view(N * M, Fin)  # N*M x Fin
        x = torch.mm(x, self.weight) + self.bias  # N*M x Fout
        return x.view(N, M, -1) 

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class chebyshevConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, kernal_size, bias=True):
        super(chebyshevConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = kernal_size
        self.weight = Parameter(torch.FloatTensor(in_features * kernal_size, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, L):
        N, M, Fin = x.shape
        # Transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # M x Fin x N
        x0 = x0.view(M, Fin * N)  # M x Fin*N
        x = x0.unsqueeze(0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
            return torch.cat((x, x_), 0)  # K x M x Fin*N

        if self.K > 1:
            x1 = torch.spmm(L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * torch.spmm(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.view(self.K, M, Fin, N)  # K x M x Fin x N
        x = x.permute(3, 1, 2, 0).contiguous()  # N x M x Fin x K
        x = x.view(N * M, Fin * self.K)  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        # W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = torch.mm(x, self.weight) + self.bias  # N*M x Fout
        return x.view(N, M, -1)  # N x M x Fout

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphFullConvolution(Module):
    """
    Full GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, adj, kernal_size=7, bias=True):
        super(GraphFullConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernal_size = kernal_size
        #self.adj = adj
        self.weight = Parameter(torch.cuda.FloatTensor(kernal_size, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.cuda.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        index_list = []
        index_value = []
        for i in range(adj.shape[0]): #
            index = (adj._indices()[0] == i).nonzero().squeeze()
            if index.dim() == 0:
                index = index.unsqueeze(0)
            index_list.append(torch.index_select(adj._indices()[1], 0, index[:self.kernal_size]))
            index_value.append(torch.index_select(adj._values(), 0, index[:self.kernal_size]))
        index_list = torch.cat([torch.cat([i, i.new_zeros(
            self.kernal_size - i.size(0))], 0) for i in index_list], 0)
        index_value = torch.stack([torch.cat([i, i.new_zeros(
            self.kernal_size - i.size(0))], 0) for i in index_value], 0)
        self.register_buffer("index_list", index_list)
        self.register_buffer("index_value", index_value)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        N, M, Fin = x.shape
        Fout = self.weight.shape[2]
        #print(adj.to_dense())
        x = x.permute(1, 0, 2).contiguous()  # M x N x Fin
        x = torch.stack(Fout*[x], 3) # M x N x Fin x Fout
        weight = torch.stack(M*[torch.stack(N*[self.weight], 1)], 0)
        select_x = torch.index_select(x, 0, self.index_list).view(
            M, self.kernal_size, N, Fin, Fout) # M x K x N x Fin x Fou
        x_elements = torch.mul(weight, select_x) # M x K x N x Fin x Fou
        x = torch.matmul(self.index_value.unsqueeze(1), x_elements.view(M, -1, N*Fin*Fout))
        x = x.view(-1, N, Fin, Fout)
        x = torch.sum(x, dim=2) # M x N x Fout
        x = x.permute(1, 0, 2).contiguous()  # N x M x Fout
        x = x.view(N*M, -1) # N*M x Fout
        x = x + self.bias  # N*M x Fout
        return x.view(N, M, -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionImprove(Module):
    """
    Full GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, adj, kernal_size=9, bias=True):
        super(GraphConvolutionImprove, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernal_size = kernal_size
        self.conv = torch.nn.Linear(in_features*kernal_size,out_features,bias=bias)
        self.activation = torch.nn.ELU()

        index_list = []
        index_value = []
        for i in range(adj.shape[0]): #
            index = (adj._indices()[0] == i).nonzero().squeeze()
            if index.dim() == 0:
                index = index.unsqueeze(0)
            index1 = torch.index_select(adj._indices()[1], 0, index[:self.kernal_size])
            if (index1 == i).nonzero()[..., 0].shape[0]:
                inx = (index1 == i).nonzero()[0][0]
            else:
                index1[-1] = i
                inx = (index1 == i).nonzero()[0][0]
            if inx > 1:
                index1[1:inx+1], index1[0] = index1[0:inx].clone(), index1[inx].clone() 
            else:
                index1[inx], index1[0] = index1[0].clone(), index1[inx].clone() 
            index_list.append(index1)
            index_value.append(torch.index_select(adj._values(), 0, index[:self.kernal_size]))
        index_list = torch.stack([torch.cat([i, i.new_zeros(
            self.kernal_size - i.size(0))-1], 0) for inx, i in enumerate(index_list)], 0)
        index_value = torch.stack([torch.cat([i, i.new_zeros(
            self.kernal_size - i.size(0))], 0) for i in index_value], 0)
        self.register_buffer("index_list", index_list)
        self.register_buffer("index_value", index_value)

    def forward(self, x):
        N, M, Fin = x.shape
        K, Fout = self.kernal_size, self.out_features
        x = F.pad(x.unsqueeze(1), (0, 0, 0, 1), "constant", 0).squeeze(1)
        index_list = self.index_list.repeat(N, 1, 1).view(N*M*K) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(N, device=x.device).view(-1,1).repeat([1,M*K]).view(-1).long() # [0*numpt,1*numpt,etc.]
        x = x[batch_index, index_list, :].view(N*M, K*Fin)  # [bsize*numpt, spiral*feats]
        # index_value = self.index_value.repeat(N, 1, 1, Fin).view(N*M, K*Fin)
        # x = x*index_value
        x = self.conv(x)
        x = self.activation(x)
        return x.view(N, M, Fout)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + 'X'   \
               + str(self.kernal_size) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionKnn(Module):
    """
    Full GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, knn_index, kernal_size=9, bias=True):
        super(GraphConvolutionKnn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernal_size = kernal_size
        self.conv = torch.nn.Linear(in_features*kernal_size,out_features,bias=bias)
        self.activation = torch.nn.ELU()
        self.knn_index = knn_index

    def forward(self, x):
        N, M, Fin = x.shape
        K, Fout = self.kernal_size, self.out_features
        x = F.pad(x.unsqueeze(1), (0, 0, 0, 1), "constant", 0).squeeze(1)
        index_list = self.knn_index.repeat(N, 1, 1).view(N*M*K) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(N, device=x.device).view(-1,1).repeat([1,M*K]).view(-1).long() # [0*numpt,1*numpt,etc.]
        x = x[batch_index, index_list, :].view(N*M, K*Fin)  # [bsize*numpt, spiral*feats]
        x = self.conv(x)
        x = self.activation(x)
        return x.view(N, M, Fout)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + 'X'   \
               + str(self.kernal_size) + ' -> ' \
               + str(self.out_features) + ')'