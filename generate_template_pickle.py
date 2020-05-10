import torch
from torch import nn
import math
import scipy.sparse
import numpy as np
from torch.nn.parameter import Parameter
from device import device
from psbody.mesh import Mesh
from graphlib import graph, coarsening, utils, mesh_sampling
import pickle

reference_mesh_file = "./data/template.obj"
ds_factors = [4, 4, 4, 4]  
# Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
# the mesh 4 times. Each time the mesh is sampled by a factor of 4
reference_mesh = Mesh(filename=reference_mesh_file)
M, A, D, U = mesh_sampling.generate_transform_matrices(
    reference_mesh, ds_factors)
pickle.dump([M, A, D, U], open("./data/pai_template.pkl", 'wb'))   


#%%
from psbody.mesh import Mesh
import torch

reference_mesh_file = "./data/template.obj"
reference_mesh = Mesh(filename=reference_mesh_file)
mean = torch.load('./data/Processed/sliced/mean.tch')
reference_mesh.v = mean.numpy()
reference_mesh.show()

#%%
