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
                noising_process, 
                optimizer,
                learning_schedule,
                ema_models,
                nepochs = 1,
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None,
                progress_batch = False,
                epoch_pbar = None,
                max_batch_per_epoch = None,
                **kwargs):
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
                
                loss = noising_process.training_losses(model, Xbatch, **kwargs)

                #loss = pdmp.training_losses(model, Xbatch, Vbatch, time_horizons)
                #loss = loss.mean()
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
                print('batch_loss', loss.item())
                #print(loss.item())
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            epoch_loss = epoch_loss / steps
            print('epoch_loss', epoch_loss)
            self.epochs += 1
            if epoch_callback is not None:
                epoch_callback(epoch_loss)









