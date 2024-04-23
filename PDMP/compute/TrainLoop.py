import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# TODO : add importance sampling for lambda?
# do a single epoch of training


class TrainLoop:
    def __init__(self):
        self.epochs = 0
        self.total_steps = 0
        
    def epoch(self, 
                dataloader, 
                model, 
                pdmp, 
                optimizer,
                learning_schedule,
                ema_models,
                nepochs = 1,
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None,
                progress_batch = False,
                epoch_pbar = None,
                max_batch_per_epoch = None):
        model.train()
        if progress_batch:
            progress_batch = lambda x : tqdm(x)
        else:
            progress_batch = lambda x : x
        
        for epoch in range(nepochs):
            epoch_loss = steps = 0
            for i, (Xbatch, y) in progress_batch(enumerate(dataloader)):
                if max_batch_per_epoch is not None:
                    if i >= max_batch_per_epoch:
                        break
                # generate random speed
                if pdmp.sampler == 'ZigZag':
                    Vbatch = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*Xbatch.shape[0],))]
                    Vbatch = Vbatch.reshape(Xbatch.shape[0], 1, 2)
                elif pdmp.sampler == 'HMC':
                    Vbatch = torch.randn_like(Xbatch)
                elif pdmp.sampler == 'BPS':
                    Vbatch = torch.randn_like(Xbatch)
                
                # generate random time horizons
                time_horizons = pdmp.T * (torch.rand(Xbatch.shape[0])**2)

                # must be of the same shape as Xbatch for the pdmp forward process, since it will be applied component wise
                t = time_horizons.clone().detach().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
                
                # clone our initial data, since it will be modified by forward process
                x = Xbatch.clone()
                v = Vbatch.clone()
                
                # apply the forward process. Everything runs on the cpu.
                pdmp.forward(Xbatch, t, Vbatch)
                
                # check that the data has been modified
                #idx = (x == Xbatch)
                #print(idx, x[idx], Xbatch[idx])
                #idx = (v == Vbatch)
                #print(idx, v[idx], Vbatch[idx])
                assert ((x == Xbatch).logical_not()).any() and ((v == Vbatch).logical_not()).any()
                # check that the time horizon has been reached for all data
                assert not (t != 0.).any()
                
                loss = pdmp.training_losses(model, Xbatch, Vbatch, time_horizons)
                loss = loss.mean()
                #print('loss computed')
                # and finally gradient descent
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if learning_schedule is not None:
                    learning_schedule.step()
                # update ema models
                if ema_models is not None:
                    for ema in ema_models:
                        ema.update(model)
                epoch_loss += loss.item()
                steps += 1
                self.total_steps += 1
                if batch_callback is not None:
                    batch_callback(loss.item())
                #print(loss.item())
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            epoch_loss = epoch_loss / steps
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)









