import os
import torch
from tqdm import tqdm
import numpy as np

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn, 
                                 bsize, start_epoch, n_epochs, eval_freq, scheduler = None,
                                 writer=None, save_recons=True, shapedata = None,
                                 metadata_dir=None, samples_dir = None, checkpoint_path = None, 
                                 generative_model='autoencoder'):
    if not shapedata.normalization:
        shapedata_mean = torch.Tensor(shapedata.mean).to(device)
        shapedata_std = torch.Tensor(shapedata.std).to(device)
    
    total_steps = start_epoch*len(dataloader_train)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        
        #%% network
        #for k, param in model.named_parameters():
        #    param.requires_grad = False if 'index_weight' in k or 'D.' in k or 'U.' in k else True  
        #model.module.updateIndex()      
        tloss = []
        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
                
            tx = sample_dict['points'].to(device)
            cur_bsize = tx.shape[0]
            #model.module.updateIndex()
            tx_hat = model(tx)
            loss = loss_fn(tx, tx_hat)
            # index_weight = model.module.index_weight
            # loss_weight = sum([torch.abs(torch.mean(torch.norm(index_weight[i], p=1, dim=1)) - 1) \
            #                   for i in range(len(index_weight))])
            # loss_sum = loss + loss_weight 
            # loss_sum.backward()   
            loss.backward()
            optim.step()
            
            #for k, param in model.named_parameters():
            #    if 'index_weight' in k:
            #        param.data.clamp_(0., 1.)
            if shapedata.normalization:
                tloss.append(cur_bsize * loss.item())
            else:
                with torch.no_grad():
                    if shapedata.mean.shape[0]!=tx.shape[1]:
                        tx_norm = tx[:,:-1,:]
                        tx_hat_norm = tx_hat[:,:-1,:]
                    else:
                        tx_norm = tx
                        tx_hat_norm = tx_hat
                    tx_norm = (tx_norm - shapedata_mean)/shapedata_std
                    tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    tx_hat_norm = (tx_hat_norm -shapedata_mean)/shapedata_std
                    tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    loss_norm = loss_fn(tx_norm, tx_hat_norm)
                    tloss.append(cur_bsize * loss_norm.item())
            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1
        '''
        #%% index
        for k, param in model.named_parameters():
            param.requires_grad = False if 'index_weight' not in k or 'D.' in k or 'U.' in k else True
        for b, sample_dict in enumerate(tqdm(dataloader_train2)):
            optim.zero_grad()
            tx = sample_dict['points'].to(device)

            cur_bsize = tx.shape[0]
            model.module.updateIndex()
            tx_hat = model(tx)
            loss = loss_fn(tx, tx_hat)

            index_weight = model.module.index_weight
            loss_weight = sum([torch.abs(torch.mean(torch.norm(index_weight[i], p=1, dim=1)) - 1) \
                              for i in range(len(index_weight))])
            loss_sum = loss + loss_weight 
            loss_sum.backward()   
            optim.step()
        '''
        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b, sample_dict in enumerate(tqdm(dataloader_val)):

                tx = sample_dict['points'].to(device)
                # if '_autoencoder' in generative_model:
                #     tx = tx[:, :-1, :]
                cur_bsize = tx.shape[0]

                tx_hat = model(tx)               
                loss = loss_fn(tx, tx_hat)
                
                if shapedata.normalization:
                    vloss.append(cur_bsize * loss.item())
                else:
                    with torch.no_grad():
                        if shapedata.mean.shape[0]!=tx.shape[1]:
                            tx_norm = tx[:,:-1,:]
                            tx_hat_norm = tx_hat[:,:-1,:]
                        else:
                            tx_norm = tx
                            tx_hat_norm = tx_hat
                        tx_norm = (tx_norm - shapedata_mean)/shapedata_std
                        tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                        tx_hat_norm = (tx_hat_norm - shapedata_mean)/shapedata_std
                        tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                        loss_norm = loss_fn(tx_norm, tx_hat_norm)
                        vloss.append(cur_bsize * loss_norm.item())   

        if scheduler:
            scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        if len(dataloader_val.dataset) > 0:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
            # print(torch.topk(model.module.index_weight[0], k=8, dim=1))
        else:
            print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
        #model = model.cpu()
        # if '_autoencoder' in generative_model:
        #     shape_dict = model.module.state_dict()
        #     shape_dict = {k: v for k, v in shape_dict.items() if 'D.' not in k and \
        #             'U.' not in k and 'p.' not in k and 'L.' not in k and 'A.' not in k}
        # else:
        shape_dict = model.module.state_dict()
        shape_dict = {k: v for k, v in shape_dict.items() if 'D.' not in k and 'U.' not in k}
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': shape_dict,  #model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': shape_dict,  #model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        #model = model.to(device)

        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
                                                     msh, mesh_ind)
                mesh_ind = [0]
                msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                shapedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
                                                 msh, mesh_ind)

    print('~FIN~')


