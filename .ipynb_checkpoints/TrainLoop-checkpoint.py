import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# TODO : add importance sampling for lambda?
# do a single epoch of training


class TrainLoop:
    def __init__(self):
        self.epochs = 0
        self.total_steps = 0
        
    def epoch(self, 
                dataloader, 
                model, 
                diffusion, 
                optimizer,
                learning_schedule,
                eval,
                ema_models,
                lploss=1., 
                monte_carlo_steps=1,
                loss_monte_carlo = 'mean',
                grad_clip = None,
                batch_callback = None,
                epoch_callback = None):
        model.train()
        epoch_loss = steps = 0
        for i, (Xbatch, y) in enumerate(dataloader):
            Xbatch = Xbatch.to(diffusion.device)
            loss = diffusion.training_losses(model, 
                                            Xbatch, 
                                            lploss = lploss,
                                            monte_carlo_steps = monte_carlo_steps,
                                            loss_monte_carlo = loss_monte_carlo,
                                            )
            # and finally gradient descent
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            learning_schedule.step()
            # update ema models
            for ema in ema_models:
                ema.update(model)
            epoch_loss += loss.item()
            steps += 1
            batch_callback(loss.item())
            #eval.register_batch_loss(loss.item())
        epoch_loss = epoch_loss / steps
        epoch_callback(epoch_loss)
        #eval.register_epoch_loss(epoch_loss)
        self.epochs += 1
        self.total_steps += steps









