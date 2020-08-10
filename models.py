import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf

class PaiConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='elu',bias=True): # ,device=None):
        super(PaiConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        self.adjweight = nn.Parameter(torch.randn(num_pts, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
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

    def forward(self,x,neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)
        x_neighbors = x_neighbors.permute(1, 0, 3, 2).contiguous()
        # x_neighbors = x_neighbors.view(num_pts, bsize*feats, num_neighbor)     
        # weight = self.softmax(torch.bmm(torch.transpose(x_neighbors, 1, 2), x_neighbors))
        # x_neighbors = torch.bmm(x_neighbors, weight) #.view(num_pts, feats, num_neighbor)
        x_neighbors = torch.bmm(x_neighbors.view(num_pts, bsize*feats, num_neighbor), self.adjweight)  #self.sparsemax(self.adjweight))
        x_neighbors = x_neighbors.view(num_pts, bsize, feats, num_neighbor).permute(1, 0, 3, 2).contiguous()
        x_neighbors = self.activation(x_neighbors.view(bsize*num_pts, num_neighbor*feats))
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat


class FeaStConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='elu',bias=True): # ,device=None):
        super(FeaStConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        self.mlp = nn.Conv1d(in_c, num_neighbor, kernel_size=1, bias=bias)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0
        self.softmax = nn.Softmax(dim=1)
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

    def forward(self,x,neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats).view(bsize*num_pts, num_neighbor, feats)
        x_neighbors = x_neighbors.permute(0, 2, 1).contiguous()

        #### relative position ####
        x_relative = x_neighbors - x_neighbors[:, :, 0:1]
        permatrix = self.softmax(self.mlp(x_relative))

        x_neighbors = torch.matmul(x_neighbors, permatrix) 
        x_neighbors = x_neighbors.view(bsize*num_pts, -1)
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)  
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat


class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, num_neighbors, x_neighbors, D, U, activation = 'elu'):
        super(PaiAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.x_neighbors = [torch.cat([torch.cat([torch.arange(x.shape[0]-1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in x_neighbors]
        #self.x_neighbors = [x.float().cuda() for x in x_neighbors]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.num_neighbors = num_neighbors
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.U = [nn.Parameter(x, False) for x in U]
        self.U = nn.ParameterList(self.U)

        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(num_neighbors)-1):
            if filters_enc[1][i]:
                self.conv.append(FeaStConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[1][i],
                                            activation=self.activation))
                input_size = filters_enc[1][i]

            self.conv.append(FeaStConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[0][i+1],
                                        activation=self.activation))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(num_neighbors)-1):
            if i != len(num_neighbors)-2:
                self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[0][i+1],
                                             activation=self.activation))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size,num_neighbors[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size,num_neighbors[-2-i], filters_dec[1][i+1],
                                                 activation='identity')) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(FeaStConv(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[0][i+1],
                                                 activation='identity'))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    # def reset_parameters(self):
    #     for x in self.index_weight:
    #         x.data.uniform_(0.0, 1.0)
    #         x.data = (x / x.sum(1, keepdim=True)).clamp(min=self.eps)
    #         x.data, _  = x.data.sort(descending=True)

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
        S = self.x_neighbors
        D = self.D
        
        j = 0
        for i in range(len(self.num_neighbors)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.num_neighbors)-1):
            #x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1-i])
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x

