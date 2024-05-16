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
                train_type = 'NORMAL',
                **kwargs):
        model.train()
        if progress_batch:
            progress_batch = lambda x : tqdm(x)
        else:
            progress_batch = lambda x : x
        
        train_procedure = [['VAE', 'NORMAL']]*10 + [['VAE', 'NORMAL_WITH_VAE']]
        #['VAE']*5 + ['NORMAL']*2 + ['NORMAL_WITH_VAE']*1 + ['NORMAL']*2

        for epoch in range(nepochs):
            epoch_loss = steps = 0
            for i, (Xbatch, y) in progress_batch(enumerate(dataloader)):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                
                # for image datasets.
                Xbatch += 2*torch.rand_like(Xbatch) / (256)

                loss = noising_process.training_losses(model, 
                                                       Xbatch, 
                                                       train_type=train_procedure[self.total_steps % len(train_procedure)], 
                                                       model_vae=model_vae, 
                                                       **kwargs)

                #loss = pdmp.training_losses(model, Xbatch, Vbatch, time_horizons)
                #loss = loss.mean()
                #print('loss computed')
                # and finally gradient descent
                optimizer.zero_grad()
                optimizer_vae.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer_vae.step()
                if learning_schedule is not None:
                    learning_schedule.step()
                if learning_schedule_vae is not None:
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
                print('batch_loss', loss.item())
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            epoch_loss = epoch_loss / steps
            print('epoch_loss', epoch_loss)
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)









