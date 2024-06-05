import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from PDMP.datasets import is_image_dataset

# TODO : add importance sampling for lambda?
# do a single epoch of training


class TrainLoop:
    def __init__(self):
        assert False, 'deprecated'
        self.epochs = 0
        self.total_steps = 0
        
    def epoch(self, 
                dataloader, 
                model,
                model_vae,
                noising_process, 
                optimizer,
                optimizer_vae,
                learning_schedule,
                learning_schedule_vae,
                ema_models,
                nepochs = 1,
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None,
                progress_batch = False,
                epoch_pbar = None,
                max_batch_per_epoch = None,
                is_image=True,
                train_type=None,
                train_alternate=False,
                **kwargs):
        model.train()

        if model_vae is not None:
            model_vae.train()
        
        #if progress_batch:
        #    progress_batch = lambda x : tqdm(x)
        #else:
        #    progress_batch = lambda x : x
        
        #train_procedure = [['VAE', 'NORMAL']]*10 + [['VAE', 'NORMAL_WITH_VAE']]
        #['VAE']*5 + ['NORMAL']*2 + ['NORMAL_WITH_VAE']*1 + ['NORMAL']*2

        freeze_vae = False
        for epoch in range(nepochs):
            epoch_loss = steps = 0
            for i, (Xbatch, y) in enumerate(tqdm(dataloader)):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                
                if is_image:
                    # for image datasets.
                    Xbatch += 2*torch.rand_like(Xbatch) / (256)

                if (model_vae is not None):
                    if not train_alternate:
                        train_type = ['VAE']
                    else:
                        freeze_vae = True
                        model_vae.eval()
                        if (self.total_steps % 2) == 0:
                            train_type = ['NORMAL'] 
                        else:
                            train_type = ['NORMAL_WITH_VAE'] 
                    if not is_image:
                        train_type = ['VAE', 'NORMAL']
                else:
                    train_type = None
                
                print('train_type:', train_type)
                
                if train_type is not None:
                    loss = noising_process.training_losses(model, 
                                                       Xbatch, 
                                                       train_type=train_type,#train_procedure[self.total_steps % len(train_procedure)], 
                                                       model_vae=model_vae, 
                                                       **kwargs)
                else:
                    loss = noising_process.training_losses(model, Xbatch, **kwargs)
                

                #loss = pdmp.training_losses(model, Xbatch, Vbatch, time_horizons)
                #loss = loss.mean()
                #print('loss computed')
                # and finally gradient descent
                optimizer.zero_grad()
                if (optimizer_vae is not None) and (not freeze_vae):
                    optimizer_vae.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if (optimizer_vae is not None) and (not freeze_vae):
                    optimizer_vae.step()
                if (learning_schedule is not None):
                    learning_schedule.step()
                if (learning_schedule_vae is not None) and (not freeze_vae):
                    learning_schedule_vae.step()
                # update ema models
                if ema_models is not None:
                    for ema in ema_models:
                        ema.update(model)
                epoch_loss += loss.item()
                steps += 1
                self.total_steps += 1
                if batch_callback is not None:
                    batch_callback(loss.item())
                if (self.total_steps % 1 == 0):
                    print('batch_loss', loss.item())
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            epoch_loss = epoch_loss / steps
            print('epoch_loss', epoch_loss)
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)









