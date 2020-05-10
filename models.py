import torch
import torch.nn as nn
import math
import pdb
import copy
#from sparsemax import Sparsemax
from numpy import inf

class PaiConv(nn.Module):
    def __init__(self, num_pts, in_c, spiral_size, out_c,activation='elu',bias=True): # ,device=None):
        super(PaiConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        #self.device = device
        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)
        self.adjweight = nn.Parameter(torch.randn(num_pts, spiral_size, spiral_size), requires_grad=True)
        self.adjweight.data = torch.eye(spiral_size).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        #self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj, kernal_weight):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
        
        # kernal_weight = kernal_weight[None, :, None, None].to(x)
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() 
        spirals = x[batch_index,spirals_index,:].view(bsize, num_pts, spiral_size, feats)
        # spirals = (kernal_weight.expand_as(spirals)*spirals).permute(1, 0, 3, 2).contiguous()
        spirals = spirals.permute(1, 0, 3, 2).contiguous()
	    # spirals = spirals.view(num_pts, bsize*feats, spiral_size)     
        # weight = self.softmax(torch.bmm(torch.transpose(spirals, 1, 2), spirals))
        # spirals = torch.bmm(spirals, weight) #.view(num_pts, feats, spiral_size)
        spirals = torch.bmm(spirals.view(num_pts, bsize*feats, spiral_size), self.adjweight)  #self.sparsemax(self.adjweight))
        spirals = spirals.view(num_pts, bsize, feats, spiral_size).permute(1, 0, 3, 2).contiguous()
        spirals = self.activation(spirals.view(bsize*num_pts, spiral_size*feats))
        out_feat = self.activation(self.conv(spirals)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat


class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, activation = 'elu'):
        super(PaiAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = [torch.cat([torch.cat([torch.arange(x.shape[0]-1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in spirals]
        #self.spirals = [x.float().cuda() for x in spirals]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.U = [nn.Parameter(x, False) for x in U]
        self.U = nn.ParameterList(self.U)
        # self.spirals_const = [torch.zeros(len(x), self.spiral_sizes[i], dtype=torch.float32).cuda() - 1 for i, x in enumerate(self.spirals)]
        # self.index_weight = [nn.Parameter(torch.randn(len(x), self.spiral_sizes[i] - 1), requires_grad=True) for i, x in enumerate(self.spirals)]
        # self.index_weight = nn.ParameterList(self.index_weight)
        self.kernal_weight = [self.spiral_sizes[i] / (self.spiral_sizes[i] + 1 - (x == -1.).sum(1)).float() for i, x in enumerate(self.spirals)]
        self.kernal_weight = [x / torch.mean(x[:-1]) for x in self.kernal_weight]
        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(PaiConv(self.spirals[i].shape[0], input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation))
                input_size = filters_enc[1][i]

            self.conv.append(PaiConv(self.spirals[i].shape[0], input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(PaiConv(self.spirals[-2-i].shape[0], input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(PaiConv(self.spirals[-2-i].shape[0], input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(PaiConv(self.spirals[-2-i].shape[0], input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(PaiConv(self.spirals[-2-i].shape[0], input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity')) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(PaiConv(self.spirals[-2-i].shape[0], input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity'))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    # def reset_parameters(self):
    #     for x in self.index_weight:
    #         x.data.uniform_(0.0, 1.0)
    #         x.data = (x / x.sum(1, keepdim=True)).clamp(min=self.eps)
    #         x.data, _  = x.data.sort(descending=True)
    
    # def updateIndex(self):
    #     self.spirals_const = [x.detach().fill_(-1.) for x in self.spirals_const]
    #     for i, adj in enumerate(self.spirals_const):
    #         adj[:-1, 0] = torch.arange(adj.shape[0]-1)
    #         #self.index_weight[i].data = (self.index_weight[i] / \
    #         #    self.index_weight[i].sum(1, keepdim=True)).clamp(min=self.eps)
    #         #index_weight = nn.functional.softmax(self.index_weight[i], dim=1)
    #         weight, inx = torch.topk(self.index_weight[i], k=self.spiral_sizes[0]-1, dim=1)
    #         inx = ((weight - weight.detach())*0.0001 + 1)*inx.float()
    #         #inx = inx.float()
    #         y_inx = ((inx - inx.detach() + 1)*self.spirals[i])
    #         adj[:, 1:] = torch.full_like(y_inx, 0).scatter_(1, inx.long(), y_inx)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()   # N x Mp x Fin
        return x

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1), self.kernal_weight[i])
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1), self.kernal_weight[i])
                j+=1
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            #x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1-i])
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1), self.kernal_weight[-2-i])
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1), self.kernal_weight[-2-i])
                j+=1
        return x

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x

