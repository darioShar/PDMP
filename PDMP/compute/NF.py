import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



class NF:
    
    def __init__(self, 
                 reverse_steps = 200,
                 device = None, 
                 time_spacing = None):
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
        self.time_spacing = time_spacing
    
    def rescale_noising(self, noising_steps, time_spacing = None):
        self.reverse_steps = noising_steps
        self.time_spacing = time_spacing
    
    def reverse_sampling(self,
                        model,
                        model_vae,
                        reverse_steps=None,
                        shape = None,
                        time_spacing = None,
                        backward_scheme = None,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        exponent = 2.,
                        print_progression = False,
                        get_sample_history = False,
                        placeholder=None,
                        ):
        assert initial_data is None, 'Using specified initial data is not yet implemented.'
        
        if model_vae is not None:
            samples = model_vae.sample(shape[0])
        else:
            samples = model().sample((shape[0], 1))

        return samples if not get_sample_history else [samples]
  

    def training_losses(self, model, 
                        X_batch, 
                        time_horizons = None, 
                        V_batch = None, 
                        train_type=['NORMAL'], 
                        model_vae=None,
                        placeholder=None,):
    
        if model_vae is not None:
            loss = - model_vae(X_batch.to(self.device))
        else:
            loss = -model().log_prob(X_batch.to(self.device))
        
        return loss.mean() #/ torch.prod(torch.tensor(X_batch.shape[1:]))
    
  