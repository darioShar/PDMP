import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil

import PDMP.pdmp_utils.Algorithms as Algo
from PDMP.pdmp_utils.Distributions import *
import PDMP.models.DiffusionBlocks as Block
import PDMP.models.Embeddings as Embedding
import zuko

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


#Can predict gaussian_noise, stable_noise, anterior_mean
class NormalizingFlowModel(nn.Module):
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'concatenate'
    ]

    def __init__(self, nfeatures, device, p_model_normalizing_flow, max_reverse_steps=1000):
        super(NormalizingFlowModel, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures        
        self.time_emb_type =    p_model_normalizing_flow['time_emb_type'] 
        self.time_emb_size =    p_model_normalizing_flow['time_emb_size']
        self.depth =            p_model_normalizing_flow['hidden_depth'] 
        self.width =            p_model_normalizing_flow['hidden_width']
        self.device =           device
        
        assert self.time_emb_type in self.possible_time_embeddings
        
        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        if self.time_emb_type == 'sinusoidal':
            print('using sinusoidal embedding with maximum of {} timesteps.'.format(max_reverse_steps))
            self.time_emb = \
            Embedding.SinusoidalPositionalEmbedding(max_reverse_steps, self.time_emb_size, self.device) # can do up to 1000 reverse_steps
        
        elif self.time_emb_type == 'learnable':
            self.time_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        elif self.time_emb_type == 'concatenate':
            pass

        if self.time_emb_type != 'concatenate':
            # possibly, remove the mlp and just use the embedding
            self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)

        # Neural spline flow (NSF) with dim sample features (V_t) and context features (X_t, t)
        self.normalizing_flow_model = zuko.flows.NSF(self.nfeatures, # generates V_t
                               self.nfeatures + self.time_emb_size,
                               transforms=p_model_normalizing_flow['transforms'], #3
                                hidden_features= [p_model_normalizing_flow['hidden_width']] * p_model_normalizing_flow['hidden_depth'] ) #[128] * 3)
        
        # zero the last module of the neural network.
        
    # return log_prob
    def forward(self, x_t, v_t, t):
        #data_shape = x_init.shape
        x_t = x_t.reshape(x_t.shape[0], -1)
        v_t = v_t.reshape(x_t.shape[0], -1)
        t = t.reshape(-1, 1)
        if self.time_emb_type == 'concatenate':
            t = t.repeat(1, self.time_emb_size)
        else:
            t = self.time_mlp(t)#timestep.to(torch.float32))

        log_p_t_model = self.normalizing_flow_model(torch.cat([x_t, t], dim = -1)).log_prob(v_t)
        return log_p_t_model
    
    # return v_t
    def sample(self, x_t, t):
        data_shape = x_t.shape
        x_t = x_t.reshape(x_t.shape[0], -1)
        t = t.reshape(-1, 1)
        if self.time_emb_type == 'concatenate':
            t = t.repeat(1, self.time_emb_size)
        else:
            t = self.time_mlp(t)#timestep.to(torch.float32))
        samples = self.normalizing_flow_model(torch.cat([x_t, t], dim = -1)).sample()
        samples = samples.reshape(*data_shape)
        return samples
