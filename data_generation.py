from tqdm import tqdm
import numpy as np
import os, argparse
from psbody.mesh import Mesh
import torch


# parser = argparse.ArgumentParser(description='Arguments for dataset split')
# parser.add_argument('-r','--root_dir', type=str,
#             help='Root data directory location, should be same as in neural3dmm.ipynb')
# parser.add_argument('-d','--dataset', type=str, 
#             help='Dataset name, Default is DFAUST')
# parser.add_argument('-v','--num_valid', type=int, default=100, 
#             help='Number of meshes in validation set, default 100')

# args = parser.parse_args()
def frontalize(vertices, canonical_vertices):
    # mesh = Mesh(filename="/home/user/3dfaceRe/center-loss_conv/scripts/template_fwh.obj")
    # canonical_vertices = mesh.v
    #canonical_vertices = np.load('Data/uv-data/canonical_vertices.npy')
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices

template = Mesh(filename='./data/template.obj')

nVal = 100        # args.num_valid
root_dir = './'   # args.root_dir
dataset = 'data'  # args.dataset 
name = 'sliced'

data = os.path.join(root_dir, dataset, 'Processed',name)

train = np.load(data+'/train.npy')
# train = torch.tensor(train.astype('float32'))
# mean = torch.mean(train, dim=0)
# std = torch.std(train, dim=0)
# torch.save(mean, './data/Processed/sliced/mean.tch')
# torch.save(std, './data/Processed/sliced/std.tch')

# for i in tqdm(range(len(train))):
#     train[i] = frontalize(train[i], template.v)
# np.save(data+'/train.npy', train)

if not os.path.exists(os.path.join(data,'points_train')):
    os.makedirs(os.path.join(data,'points_train'))

if not os.path.exists(os.path.join(data,'points_val')):
    os.makedirs(os.path.join(data,'points_val'))

if not os.path.exists(os.path.join(data,'points_test')):
    os.makedirs(os.path.join(data,'points_test'))


for i in tqdm(range(len(train)-nVal)):
    vertex = torch.tensor(train[i].astype('float32'))
    torch.save(vertex, os.path.join(data,'points_train','{0}.tch'.format(i)))
    #np.save(os.path.join(data,'points_train','{0}.npy'.format(i)),train[i])
for i in range(len(train)-nVal,len(train)):
    vertex = torch.tensor(train[i].astype('float32'))
    torch.save(vertex, os.path.join(data,'points_val','{0}.tch'.format(i)))
    #np.save(os.path.join(data,'points_val','{0}.npy'.format(i)),train[i])
    
test = np.load(data+'/test.npy')
# for i in tqdm(range(len(test))):
#     test[i] = frontalize(test[i], template.v)
# np.save(data+'/test.npy', test)

for i in range(len(test)):
    vertex = torch.tensor(test[i].astype('float32'))
    torch.save(vertex, os.path.join(data,'points_test','{0}.tch'.format(i)))
    #np.save(os.path.join(data,'points_test','{0}.npy'.format(i)),test[i])

files = []
for r, d, f in os.walk(os.path.join(data,'points_train')):
    for file in f:
        if '.tch' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_train.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_val')):
    for file in f:
        if '.tch' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_val.npy'),files)

files = []
for r, d, f in os.walk(os.path.join(data,'points_test')):
    for file in f:
        if '.tch' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data,'paths_test.npy'),files)

