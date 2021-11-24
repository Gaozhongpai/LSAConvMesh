from re import template
import torch
import copy
from tqdm import tqdm
import numpy as np
import time
import pyrender
import cv2
from scipy.spatial.transform import Rotation as R
import trimesh

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant=1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    template = shapedata.reference_mesh
    with torch.no_grad():
        start_time = time.time()
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
                
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:,:-1]
                x = tx[:,:-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            
            x_recon = (x_recon * shapedata_std + shapedata_mean)
            for j in range(x_recon.shape[0]):
                template.vertices = x_recon[j].cpu().numpy() 
                rotate=trimesh.transformations.rotation_matrix(
                    angle=np.radians(-90.0),
                    direction=[0,1,0],
                    point=[0,0,0])
                template.vertices =template.vertices @ rotate[:3, :3]
                template.export('meshes/'+str(i).zfill(3)+str(j).zfill(3)+'.ply','ply')   
                mesh = pyrender.Mesh.from_trimesh(template)
                scene = pyrender.Scene()
                scene.add(mesh)
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
                s = 1 # np.sqrt(2)/2
                camera_pose = np.array([
                    [1.0, -s,   s,   0.15],
                    [0.0,  1.0, 0.0, -0.1],
                    [0.0,  s,   1.,   0.25],
                    [0.0,  0.0, 0.0, 1.0],
                    ])
                scene.add(camera, pose=camera_pose)
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(light, pose=camera_pose)
                r = pyrender.OffscreenRenderer(400, 400)
                color, depth = r.render(scene)
                cv2.imwrite('meshes/'+str(i).zfill(3)+str(j).zfill(3)+'.png', color)

                # pyrender.Viewer(scene, use_raymond_lighting=True)
            # x = (x * shapedata_std + shapedata_mean) * mm_constant
            # l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim=2)))*x.shape[0]/float(len(dataloader_test.dataset))
        print("--- %s seconds ---" % (time.time() - start_time))
        predictions = predictions.cpu()
        # l1_loss = l1_loss.item()
        # l2_loss = l2_loss.item()
    
    return predictions, l1_loss.cpu(), l2_loss.cpu()