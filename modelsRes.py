import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf
from sparsemax import Sparsemax
import torch.nn.functional as F
import math

class PaiConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu',bias=True): # ,device=None):
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
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()
        
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)
        # x_neighbors = x_neighbors.view(num_pts, bsize*feats, num_neighbor)     
        # weight = self.softmax(torch.bmm(torch.transpose(x_neighbors, 1, 2), x_neighbors))
        # x_neighbors = torch.bmm(x_neighbors, weight) #.view(num_pts, feats, num_neighbor)
        x_neighbors = torch.einsum('bnkf, bnkt->bntf', x_neighbors, self.adjweight[None].repeat(bsize, 1, 1, 1))   #self.sparsemax(self.adjweight))
        x_neighbors = F.elu(x_neighbors.contiguous().view(bsize*num_pts, num_neighbor*feats)) 
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat

class PaiConvTiny(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c,activation='relu',bias=True): # ,device=None):
        super(PaiConvTiny,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c*num_neighbor,out_c,bias=bias)
        # self.norm = nn.BatchNorm1d(in_c)
        # self.fc1 = nn.Linear(in_c, in_c)
        # self.fc2 = nn.Linear(out_c, out_c)
        mappingsize = 64
        self.num_base = 64
        self.num_neighbor = num_neighbor
        if self.num_base < num_pts:
            num_base = self.num_base
            self.temp_factor = 100
            self.tmptmlp = nn.Linear(mappingsize*2, 1)
            self.softmax = nn.Softmax(dim=1) # Sparsemax(dim=-1) # nn.Softmax(dim=1)
            self.mlp = nn.Linear(mappingsize*2, num_base)
        else:
            num_base = num_pts

        self.adjweight = nn.Parameter(torch.randn(num_base, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0,-1,0] = 0.0

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        neighbor_index = neighbor_index[:, :, :self.num_neighbor].contiguous()
        _, _, num_neighbor = neighbor_index.size()

        # x = self.activation(self.fc1(x.view(-1, feats))).view(bsize, num_pts, -1)
        # x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize*num_pts*num_neighbor) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*num_neighbor]).view(-1).long() 
        x_neighbors = x[batch_index,neighbor_index,:].view(bsize, num_pts, num_neighbor, feats)

        if self.num_base < num_pts:
            tmpt = torch.sigmoid(self.tmptmlp(t_vertex))*(0.1 - 1.0/self.temp_factor) + 1.0/self.temp_factor 
            adjweightBase = self.softmax(self.mlp(t_vertex)/tmpt)
            adjweight = torch.einsum('ns,skt->nkt', adjweightBase, self.adjweight)[None].repeat(bsize, 1, 1, 1)
        else:
            adjweight = self.adjweight[None].repeat(bsize, 1, 1, 1)
        x_neighbors = torch.einsum('bnkf,bnkt->bnft', x_neighbors, adjweight)
        x_neighbors = x_neighbors.view(bsize*num_pts, num_neighbor*feats)
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize,num_pts,self.out_c)
        # out_feat = self.activation(self.fc2(out_feat.view(-1, self.out_c))).view(bsize, num_pts, -1)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        return out_feat

class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, 
                 t_vertices, sizes, num_neighbors, x_neighbors, D, U, activation = 'elu'):
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

        mappingsize = 64
        self.B = nn.Parameter(torch.randn(7*9, mappingsize) , requires_grad=False)  
        self.t_vertices = [x[self.x_neighbors[i]].permute(0, 2, 1).contiguous() for i, x in enumerate(t_vertices)]
        t_vertices_repeat = [x[:, :, 0:1].expand_as(x) for x in self.t_vertices]
        self.t_vertices = [torch.cat([x - t_vertices_repeat[i], t_vertices_repeat[i], 
                            torch.norm(x - t_vertices_repeat[i], dim=1, keepdim=True)], dim=1)
                            for i, x in enumerate(self.t_vertices)]
        self.t_vertices = [2.*math.pi*x.view(x.shape[0], -1) @ (self.B.data).to(x) for x in self.t_vertices]
        self.t_vertices = [torch.cat([torch.sin(x), torch.cos(x)], dim=-1) for x in self.t_vertices]

        self.eps = 1e-7
        #self.reset_parameters()
        #self.device = device
        self.activation = activation
        self.conv = []
        self.mlp_conv = []
        input_size = filters_enc[0]
        for i in range(len(num_neighbors)-1):
            self.conv.append(PaiConvTiny(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[i+1],
                                        activation=self.activation))
            self.mlp_conv.append(nn.Linear(input_size, filters_enc[i+1], bias=False))
            input_size = filters_enc[i+1]

        self.conv = nn.ModuleList(self.conv)   
        self.mlp_conv = nn.ModuleList(self.mlp_conv)
        
        # self.fc_enc =nn.Linear(sum(filters_enc[1:]), input_size)
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0])
        # self.fc_dec =nn.Linear(sum(filters_dec[1:-1])//2, filters_dec[-2])
        
        self.dconv = []
        self.mlp_dconv = []
        # self.mlp_dec = []
        input_size = filters_dec[0]
        for i in range(len(num_neighbors)-1):
            self.dconv.append(PaiConvTiny(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[i+1],
                                            activation=self.activation))
            self.mlp_dconv.append(nn.Linear(input_size, filters_dec[i+1], bias=False))
            input_size = filters_dec[i+1]  
            # self.mlp_dec.append(nn.Linear(input_size, filters_dec[-2-i]//2, bias=False))
            if i == len(num_neighbors)-2:
                input_size = filters_dec[-2]
                self.dconv.append(PaiConvTiny(self.x_neighbors[-2-i].shape[0], input_size, num_neighbors[-2-i], filters_dec[-1],
                                                activation="identity"))
                    
        self.dconv = nn.ModuleList(self.dconv)
        self.mlp_dconv = nn.ModuleList(self.mlp_dconv)
        # self.mlp_dec = nn.ModuleList(self.mlp_dec)

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
        t_vertices = self.t_vertices
        x_list = []
        for i in range(len(self.num_neighbors)-1):
            x_res = self.poolwT(x, D[i])
            x = self.conv[i](x, t_vertices[i], S[i].repeat(bsize,1,1))
            x_res = self.mlp_conv[i](x_res.view(-1, x_res.shape[-1])).view(bsize, -1, x.shape[-1])
            #x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i]) + x_res
            # x_down = x
            # for j in range(i+1, len(self.num_neighbors)-1):
            #     x_down = self.poolwT(x_down, D[j])
            # x_list.append(x_down)
        ## x = self.conv[-1](x, t_vertices[-1], S[-1].repeat(bsize,1,1))
        ## x_list.append(x)
        # x = torch.cat(x_list, dim=-1)
        # x = F.elu(self.fc_enc(x.view(bsize*x.shape[1], -1)).view(bsize, x.shape[1], -1))
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U
        t_vertices = self.t_vertices

        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)

        x_list = []
        for i in range(len(self.num_neighbors)-1):
            #x = torch.matmul(U[-1-i],x)
            x_res = self.mlp_dconv[i](x.view(-1, x.shape[-1]))
            x_res = x_res.view(bsize, -1, x_res.shape[-1])
            x_res = self.poolwT(x_res, U[-1-i])
            x = self.poolwT(x, U[-1-i])
            x = self.dconv[i](x, t_vertices[-2-i], S[-2-i].repeat(bsize,1,1))
            x = x + x_res
            # x_up = x
            # for j in range(i+1, len(self.num_neighbors)-1):
            #     x_up = self.poolwT(x_up, U[-1-j])
            # x_up = self.mlp_dec[i](x_up.view(bsize*x_up.shape[1], -1)).view(bsize, x_up.shape[1], -1)
            # x_list.append(x_up)

        # x = torch.cat(x_list, dim=-1)
        # x = F.elu(self.fc_dec(x.view(bsize*x.shape[1], -1)).view(bsize, x.shape[1], -1))
        x = self.dconv[-1](x, t_vertices[0], S[0].repeat(bsize,1,1))
        return x

    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x
