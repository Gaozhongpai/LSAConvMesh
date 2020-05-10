import numpy as np
from random import choice
from random import shuffle
from pykeops.torch import generic_argkmin
import torch.nn as nn
import torch
import math
from collections import deque
import scipy.sparse as sp
## Note: Non-smooth surfaces or bad triangulations may lead to non-spiral orderings of the vertices.
## Common issue in badly triangulated surfaces is that there exist some edges that belong to more than two triangles. In this
## case the mathematical definition of the spiral is insufficient. In this case, in this version of the code, we randomly
## choose two triangles in order to continue the inductive assignment of the order to the rest of the vertices.

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

def get_adj_trigs(A, F, reference_mesh, meshpackage = 'mpi-mesh'):
    
    # Adj = []
    # for x in A:
    #     adj_x = []
    #     dx = x.todense()
    #     for i in range(x.shape[0]):
    #         adj_x.append(dx[i].nonzero()[1])
    #     Adj.append(adj_x)
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

    if meshpackage =='trimesh':
        mesh_faces = reference_mesh.faces
    elif meshpackage =='mpi-mesh':
        mesh_faces = reference_mesh.f
    # Create Triangles List

    trigs_full = [[] for i in range(len(Adj[0]))]
    for t in mesh_faces:
        u, v, w = t
        trigs_full[u].append((u,v,w))
        trigs_full[v].append((u,v,w))
        trigs_full[w].append((u,v,w))

    Trigs = [trigs_full]
    for i,T in enumerate(F):
        trigs_down = [[] for i in range(len(Adj[i+1]))]
        for u,v,w in T:
            trigs_down[u].append((u,v,w))
            trigs_down[v].append((u,v,w))
            trigs_down[w].append((u,v,w))
        Trigs.append(trigs_down)
    return Adj, Trigs

def generate_spirals(step_sizes, M, Adj, Trigs, 
                    reference_points, dilation=None, 
                    random=False, meshpackage = 'mpi-mesh', 
                    counter_clockwise = True, nb_stds = 2, 
                    is_shuffle=False):
    Adj_spirals = []
    for i in range(len(Adj)):
        if meshpackage =='trimesh':
            mesh_vertices  = M[i].vertices
        elif meshpackage =='mpi-mesh':
            mesh_vertices  = M[i].v
 
        sp = get_spirals(mesh_vertices, Adj[i],Trigs[i],reference_points[i], n_steps=step_sizes[i],\
                         padding='zero', counter_clockwise = counter_clockwise, random = random)
        Adj_spirals.append(sp)
        print('spiral generation for hierarchy %d (%d vertices) finished' %(i,len(Adj_spirals[-1])))

    
    ## Dilated convolution
    if dilation:
        for i in range(len(dilation)):
            dil = dilation[i]
            dil_spirals = []
            for j in range(len(Adj_spirals[i])):
                s = Adj_spirals[i][j][:1] + Adj_spirals[i][j][1::dil]
                dil_spirals.append(s)
            Adj_spirals[i] = dil_spirals
            
            
    # Calculate the lengths of spirals
    #   Use mean + 2 * std_dev, to capture 97% of data
    L = []
    for i in range(len(Adj_spirals)):
        L.append([])
        for j in range(len(Adj_spirals[i])):
            L[i].append(len(Adj_spirals[i][j]))
        L[i] = np.array(L[i])
    spiral_sizes = []
    for i in range(len(L)):
        sz = L[i].mean() + nb_stds*L[i].std()
        spiral_sizes.append(int(sz))
        print('spiral sizes for hierarchy %d:  %d' %(i,spiral_sizes[-1]))
    

    # 1) fill with -1 (index to the dummy vertex, i.e the zero padding) the spirals with length smaller than the chosen one
    # 2) Truncate larger spirals
    spirals_np = []
    for i in range(len(spiral_sizes)): #len(Adj_spirals)):
        S = np.zeros((1,len(Adj_spirals[i])+1,spiral_sizes[i])) - 1
        for j in range(len(Adj_spirals[i])):
            if is_shuffle:
                shuffle(Adj_spirals[i][j])
            S[0,j,:len(Adj_spirals[i][j])] = Adj_spirals[i][j][:spiral_sizes[i]]
        #spirals_np.append(np.repeat(S,args['batch_size'],axis=0))
        spirals_np.append(S)

    return spirals_np, spiral_sizes, Adj_spirals


def distance(v,w):
    return np.sqrt(np.sum(np.square(v-w)))

def single_source_shortest_path(V,E,source,dist=None,prev=None):
    import heapq
    if dist == None:
        dist = [None for i in range(len(V))]
        prev = [None for i in range(len(V))]
    q = []
    seen = set()
    heapq.heappush(q,(0,source,None))
    while len(q) > 0 and len(seen) < len(V):
        d_,v,p = heapq.heappop(q)
        if v in seen:
            continue
        seen.add(v)
        prev[v] = p
        dist[v] = d_
        for w in E[v]:
            if w in seen:
                continue
            dw = d_ + distance(V[v],V[w])
            heapq.heappush(q,(dw,w,v))
    
    return prev, dist
        
        
def get_spirals(mesh, adj, trig, reference_points, n_steps=1, padding='zero', counter_clockwise = True, random = False):
    spirals = []
    
    if not random:
        heat_path = None
        dist = None
        for reference_point in reference_points:
            heat_path,dist = single_source_shortest_path(mesh,adj,reference_point, dist, heat_path)
        heat_source = reference_points

    for i in range(mesh.shape[0]):
        seen = set(); seen.add(i)
        trig_central = list(trig[i]); A = adj[i]; spiral = [i]
        
        # 1) Frist degree of freedom - choose starting pooint:
        if not random:
            if i in heat_source: # choose closest neighbor
                shortest_dist = np.inf
                init_vert = None
                for neighbor in A:
                    d = np.sum(np.square(mesh[i] - mesh[neighbor]))
                    if d < shortest_dist:
                        shortest_dist = d
                        init_vert = neighbor

            else: #   on the shortest path to the reference point
                init_vert = heat_path[i]
        else:
            # choose starting point:
            #   random for first ring
            init_vert = choice(A)
            
    
        
        
        # first ring
        if init_vert is not None:
            ring = [init_vert]; seen.add(init_vert)
        else:
            ring = []
        while len(trig_central) > 0 and init_vert is not None:
            cur_v = ring[-1]
            cur_t = [t for t in trig_central if t in trig[cur_v]]
            if len(ring) == 1:
                orientation_0 = (cur_t[0][0]==i and cur_t[0][1]==cur_v) \
                                or (cur_t[0][1]==i and cur_t[0][2]==cur_v) \
                                or (cur_t[0][2]==i and cur_t[0][0]==cur_v)
                if not counter_clockwise:
                    orientation_0 = not orientation_0
                    
                # 2) Second degree of freedom - 2nd point/orientation ambiguity
                if len(cur_t) >=2:
                # Choose the triangle that will direct the spiral counter-clockwise
                    if orientation_0:
                    # Third point in the triangle - next vertex in the spiral
                        third = [p for p in cur_t[0] if p!=i and p!=cur_v][0]
                        trig_central.remove(cur_t[0])
                    else:
                        third = [p for p in cur_t[1] if p!=i and p!=cur_v][0]
                        trig_central.remove(cur_t[1])
                    ring.append(third)
                    seen.add(third)      
                # 3) Stop if the spiral hits the boundary in the first point
                elif len(cur_t) == 1:
                    break
            else:
                # 4) Unique ordering for the rest of the points (3rd onwards) 
                if len(cur_t) >= 1:
                    # Third point in the triangle - next vertex in the spiral
                    third = [p for p in cur_t[0] if p!= cur_v and p!=i][0]   
                    # Don't append the spiral if the vertex has been visited already 
                    # (happens when the first ring is completed and the spiral returns to the central vertex)
                    if third not in seen:
                        ring.append(third)
                        seen.add(third)
                    trig_central.remove(cur_t[0])
            # 4) Stop when the spiral hits the boundary (the already visited triangle is no longer in the list): First half of the spiral
                elif len(cur_t) == 0:
                    break
                


                
      
        rev_i = len(ring)
        if init_vert is not None:
            v = init_vert

            if orientation_0 and len(ring)==1:
                reverse_order = False
            else:
                reverse_order = True
        need_padding = False  
        
        # 5) If on the boundary: restart from the initial vertex towards the other direction, 
        # but put the vertices in reverse order: Second half of the spiral
        # One exception if the starting point is on the boundary +  2nd point towards the desired direction
        while len(trig_central) > 0 and init_vert is not None:
            cur_t = [t for t in trig_central if t in trig[v]]
            if len(cur_t) != 1:
                break
            else:
                need_padding = True
                
            third = [p for p in cur_t[0] if p!=v and p!=i][0]
            trig_central.remove(cur_t[0])
            if third not in seen:
                ring.insert(rev_i,third)
                seen.add(third)
                if not reverse_order:
                    rev_i = len(ring)
                v = third
            
        # Add a dummy vertex between the first half of the spiral and the second half - similar to zero padding in a 2d grid
        if need_padding:
            ring.insert(rev_i,-1)
            """
            ring_copy = list(ring[1:])
            rev_i = rev_i - 1
            for z in range(len(ring_copy)-2):
                if padding == 'zero':
                    ring.insert(rev_i,-1) # -1 is our sink node
                elif padding == 'mirror':
                    ring.insert(rev_i,ring_copy[rev_i-z-1])
            """
        spiral += ring
        
        
        # Next rings:
        for step in range(n_steps-1):
            next_ring = set([]); next_trigs = set([]); 
            if len(ring) == 0:
                break
            base_triangle = None
            init_vert = None
            
            # Find next hop neighbors
            for w in ring:
                if w!=-1:
                    for u in adj[w]:
                        if u not in seen:
                            next_ring.add(u)
            
            # Find triangles that contain two outer ring nodes. That way one can folllow the spiral ordering in the same way 
            # as done in the first ring: by simply discarding the already visited triangles+nodes.
            for u in next_ring:
                for tr in trig[u]:
                    if len([x for x in tr if x in seen]) == 1:
                        next_trigs.add(tr)
                    elif ring[0] in tr and ring[-1] in tr:
                        base_triangle = tr
            # Normal case: starting point in the second ring -> 
            # the 3rd point in the triangle that connects the 1st and the last point in the 1st ring with the 2nd ring
            if base_triangle is not None:
                init_vert = [x for x in base_triangle if x != ring[0] and x != ring[-1]]
                # Make sure that the the initial point is appropriate for starting the spiral,
                # i.e it is connected to at least one of the next candidate vertices
                if len(list(next_trigs.intersection(set(trig[init_vert[0]]))))==0:
                    init_vert = None


            # If no such triangle exists (one of the vertices is dummy, 
            # or both the first and the last vertex take part in a specific type of boundary)
            # or the init vertex is not connected with the rest of the ring -->
            # Find the relative point in the the triangle that connects the 1st point with the 2nd, or the 2nd with the 3rd
            # and so on and so forth. Note: This is a slight abuse of the spiral topology
            if init_vert is None:
                for r in range(len(ring)-1):
                    if ring[r] !=-1 and ring[r+1]!=-1:
                        tr = [t for t in trig[ring[r]] if t in trig[ring[r+1]]]
                        for t in tr:
                            init_vert = [v for v in t if v not in seen] 
                            # make sure that the next vertex is appropriate to start the spiral ordering in the next ring
                            if len(init_vert)>0 and len(list(next_trigs.intersection(set(trig[init_vert[0]]))))>0:
                                break
                            else:
                                init_vert = []
                        if len(init_vert)>0  and len(list(next_trigs.intersection(set(trig[init_vert[0]]))))>0:
                            break
                        else:
                            init_vert = []

            
            # The rest of the procedure is the same as the first ring
            if init_vert is None:
                init_vert = []
            if len(init_vert)>0:
                init_vert = init_vert[0]
                ring = [init_vert]
                seen.add(init_vert)
            else:
                init_vert = None
                ring = []

#             if i == 57:
#                 import pdb;pdb.set_trace()
            while len(next_trigs) > 0 and init_vert is not None:
                cur_v = ring[-1]
                cur_t = list(next_trigs.intersection(set(trig[cur_v])))
                    
                if len(ring) == 1:
                    try:
                        orientation_0 = (cur_t[0][0] in seen and cur_t[0][1]==cur_v) \
                                        or (cur_t[0][1] in seen and cur_t[0][2]==cur_v) \
                                        or (cur_t[0][2] in seen and cur_t[0][0]==cur_v)
                    except:
                        import pdb;pdb.set_trace()
                    if not counter_clockwise:
                        orientation_0 = not orientation_0

                    # 1) orientation ambiguity for the next ring
                    if len(cur_t) >=2:
                    # Choose the triangle that will direct the spiral counter-clockwise
                        if orientation_0:
                        # Third point in the triangle - next vertex in the spiral
                            third = [p for p in cur_t[0] if p not in seen and p!=cur_v][0]
                            next_trigs.remove(cur_t[0])
                        else:
                            third = [p for p in cur_t[1] if p not in seen and p!=cur_v][0]
                            next_trigs.remove(cur_t[1])
                        ring.append(third)
                        seen.add(third)      
                    # 2) Stop if the spiral hits the boundary in the first point
                    elif len(cur_t) == 1:
                        break
                else:
                    # 3) Unique ordering for the rest of the points
                    if len(cur_t) >= 1:
                        third = [p for p in cur_t[0] if p != v and p not in seen]
                        next_trigs.remove(cur_t[0])
                        if len(third)>0:
                            third = third[0]
                            if third not in seen:
                                ring.append(third)
                                seen.add(third)
                        else:
                            break
                    # 4) Stop when the spiral hits the boundary 
                    # (the already visited triangle is no longer in the list): First half of the spiral
                    elif len(cur_t) == 0:
                        break
                    
                       
            rev_i = len(ring)
            if init_vert is not None:
                v = init_vert

                if orientation_0 and len(ring)==1:
                    reverse_order = False
                else:
                    reverse_order = True
        
            need_padding = False
            
            while len(next_trigs) > 0 and init_vert is not None:
                cur_t = [t for t in next_trigs if t in trig[v]]
                if len(cur_t) != 1:
                    break
                else:
                    need_padding = True
                    
                third = [p for p in cur_t[0] if p!=v and p not in seen]
                next_trigs.remove(cur_t[0])
                if len(third)>0:
                    third = third[0]
                    if third not in seen:
                        ring.insert(rev_i,third)
                        seen.add(third)
                    if not reverse_order:
                        rev_i = len(ring)
                    v = third
            
            if need_padding:
                ring.insert(rev_i,-1)
                """
                ring_copy = list(ring[1:])
                rev_i = rev_i - 1
                for z in range(len(ring_copy)-2):
                    if padding == 'zero':
                        ring.insert(rev_i,-1) # -1 is our sink node
                    elif padding == 'mirror':
                        ring.insert(rev_i,ring_copy[rev_i-z-1])
                """

            spiral += ring  
        
        spirals.append(spiral)
    return spirals

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


def compute_tri_ver_normal(vertices, faces):
    vertices = vertices.permute(1, 2, 0).contiguous()
    tris = vertices[faces]
    normal = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm_tris = nn.functional.normalize(normal)
    norm_tris = norm_tris.permute(2, 0, 1)
    #print(norm_tris.permute(2, 0, 1))

    norm_vertex = vertices.clone().fill_(0)
    norm_vertex[faces[:, 0]] += normal
    norm_vertex[faces[:, 1]] += normal
    norm_vertex[faces[:, 2]] += normal
    norm_vertex = nn.functional.normalize(norm_vertex)
    norm_vertex = norm_vertex.permute(2, 0, 1)
    return norm_tris.contiguous(), norm_vertex.contiguous()
    
def clockwiseangle_and_distance(vector):
    # Vector between point and the origin: v = p - o
    #vector = [point[0]-origin[0], point[1]-origin[1]]
    vector = vector[1]
    refvec = [-0.5, 1]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def generate_knn(M, norm_vertices, kernal_size=[9, 9, 9, 9, 9]):
    import matplotlib.pyplot as plt
    knn_indexes = []
    knn = generic_argkmin(
        'SqDist(x, y)',  # Formula
        'a = Vi(9)',  # Output: 3 scalars per line
        'x = Vi(3)',  # 1st input: dim-100 vector per line
        'y = Vj(3)')  # 2nd input: dim-100 vector per line

    for j in range(len(M)):
        mesh_vertices  = M[j].v    
        mesh_vertices = torch.from_numpy(mesh_vertices).type(torch.FloatTensor)
        #reference, index = torch.max(mesh_vertices[:, 1], 0)
        #reference = mesh_vertices[index].unsqueeze(0)
        knn_index = knn(mesh_vertices, mesh_vertices, backend='CPU')
        knn_index = torch.cat([knn_index, torch.zeros(1, kernal_size[j], dtype=torch.int64)-1])
        knn_indexes.append(knn_index)
        print('knn index generation for hierarchy %d (%d vertices) finished' %(j,len(mesh_vertices)))

    return knn_indexes, kernal_size

def generate_pai(M, A, norm_vertices, kernal_size=[9, 9, 9, 9, 9], is_shuffle=False):
    import matplotlib.pyplot as plt
    knn_indexes = []
    for j in range(len(M)):
        mesh_vertices, adj = M[j].v, A[j]
        mesh_vertices = torch.from_numpy(mesh_vertices).type(torch.FloatTensor)
        index_list = []
        for i in range(len(adj)): #
            index1 = adj[i] #torch.from_numpy(adj[i])
            if is_shuffle is False:
                normal_vertex = norm_vertices[i, :]
                knn_neightbor = mesh_vertices[index1]
                # project neightbors to the plane
                knn_dot = torch.mm(knn_neightbor - knn_neightbor[0, :].unsqueeze(0), normal_vertex.unsqueeze(1))
                knn_neightbor_project = knn_neightbor - knn_dot*normal_vertex.unsqueeze(0)
                # order the points clockwise 
                knn_index_sub = [k[0] for k in 
                            sorted(enumerate(knn_neightbor_project[:, :2] - knn_neightbor_project[0, :2].unsqueeze(0)), 
                            key=clockwiseangle_and_distance)]
                # if i == 3522:
                #     plt.plot(knn_neightbor_project[knn_index_sub, 0].numpy(), knn_neightbor_project[knn_index_sub, 1].numpy(), '-ok')
                #     plt.show()
                #     plt.plot(knn_neightbor[knn_index_sub, 0].numpy(), knn_neightbor[knn_index_sub, 1].numpy(), '-or')
                #     plt.show()
                knn_index_sub = index1[knn_index_sub]
            else: 
                knn_index_sub = index1[torch.randperm(len(index1))]
            index_list.append(knn_index_sub[:9])
        index_list.append(torch.zeros(kernal_size[j], dtype=torch.int64)-1)
        #index_list = torch.stack([torch.cat([i, i.new_zeros(
        #    kernal_size[j] - i.size(0))-1], 0) for inx, i in enumerate(index_list)], 0)
        knn_indexes.append(index_list)
        print('index generation for hierarchy %d (%d vertices) finished' %(j,len(mesh_vertices)))
    return knn_indexes, kernal_size
