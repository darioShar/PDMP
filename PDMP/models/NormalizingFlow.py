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

    possible_x_embeddings = [
        'mlp',
        'unet',
        'concatenate'
    ]
    def __init__(self, nfeatures, device, p_model_normalizing_flow, max_reverse_steps=1000, unet=None):
        super(NormalizingFlowModel, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures # else just learn time
        self.x_emb_type =       p_model_normalizing_flow['x_emb_type']
        self.x_emb_size =       p_model_normalizing_flow['x_emb_size']
        self.time_emb_type =    p_model_normalizing_flow['time_emb_type']
        self.time_emb_size =    p_model_normalizing_flow['time_emb_size']
        self.depth =            p_model_normalizing_flow['hidden_depth']
        self.width =            p_model_normalizing_flow['hidden_width']
        self.type =             p_model_normalizing_flow['model_type']
        #self.flow_blocks =      p_model_normalizing_flow['flow_blocks']
        self.device =           device
        
        assert self.time_emb_type in self.possible_time_embeddings
        assert self.x_emb_type in self.possible_x_embeddings

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
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)

        if self.x_emb_type == 'mlp':
            self.x_emb = nn.Linear(self.nfeatures, self.x_emb_size).to(self.device)
            self.x_mlp = nn.Sequential(self.x_emb,
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act)
        elif self.x_emb_type == 'unet':
            assert unet is not None, 'must provide unet model'
            self.unet = unet
            self.x_emb_size = self.nfeatures
        elif self.x_emb_type == 'concatenate':
            self.x_emb_size = self.nfeatures
        # Neural spline flow (NSF) with dim sample features (V_t) and context features (X_t, t)
        if self.type == 'NSF':
            self.normalizing_flow_model = zuko.flows.NSF(self.nfeatures, # generates V_t
                               self.x_emb_size + self.time_emb_size,
                               transforms=p_model_normalizing_flow['transforms'], #3
                                hidden_features= [p_model_normalizing_flow['hidden_width']] * p_model_normalizing_flow['hidden_depth'] ) #[128] * 3)
        elif self.type == 'MAF':
            self.normalizing_flow_model = zuko.flows.MAF(self.nfeatures, # generates V_t  
                               self.x_emb_size + self.time_emb_size,
                               transforms=p_model_normalizing_flow['transforms'], #3
                                hidden_features= [p_model_normalizing_flow['hidden_width']] * p_model_normalizing_flow['hidden_depth'] ) #[128] * 3)
        else:
            raise Exception('normalizing flow model type {} not yet implemented'.format(self.type))
        
        # zero the last module of the neural network.
        # todo
    
    
    def _forward(self, x_t, t):
        if self.x_emb_type == 'unet':
            x_t = self.unet(x_t, t)    
        x_t = x_t.reshape(x_t.shape[0], -1)
        if self.x_emb_type == 'mlp':
            x_t = self.x_mlp(x_t)
        t = t.reshape(-1, 1)
        if self.time_emb_type == 'concatenate':
            t = t.repeat(1, self.time_emb_size)
        else:
            t = self.time_mlp(t)#timestep.to(torch.float32))
        return x_t, t

    # return log_prob
    def forward(self, x_t, v_t, t):
        x_t, t = self._forward(x_t, t)
        v_t = v_t.reshape(x_t.shape[0], -1)
        log_p_t_model = self.normalizing_flow_model(torch.cat([x_t, t], dim = -1)).log_prob(v_t)
        return log_p_t_model
    
    # return v_t
    def sample(self, x_t, t):
        data_shape = x_t.shape
        x_t, t = self._forward(x_t, t)
        samples = self.normalizing_flow_model(torch.cat([x_t, t], dim = -1)).sample()
        samples = samples.reshape(*data_shape)
        return samples
    


#Can predict gaussian_noise, stable_noise, anterior_mean
class NormalizingFlowModelJumpTime(nn.Module):
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'concatenate'
    ]

    possible_x_embeddings = [
        'mlp',
        'unet',
        'concatenate'
    ]
    def __init__(self, nfeatures, device, p_model_normalizing_flow, max_reverse_steps=1000, unet=None):
        super(NormalizingFlowModelJumpTime, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures # else just learn time
        self.x_emb_type =       p_model_normalizing_flow['x_emb_type']
        self.x_emb_size =       p_model_normalizing_flow['x_emb_size']
        self.time_emb_type =    p_model_normalizing_flow['time_emb_type']
        self.time_emb_size =    p_model_normalizing_flow['time_emb_size']
        self.depth =            p_model_normalizing_flow['hidden_depth']
        self.width =            p_model_normalizing_flow['hidden_width']
        self.type =             p_model_normalizing_flow['model_type']
        #self.flow_blocks =      p_model_normalizing_flow['flow_blocks']
        self.device =           device
        
        assert self.time_emb_type in self.possible_time_embeddings
        assert self.x_emb_type in self.possible_x_embeddings

        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        if self.time_emb_type == 'sinusoidal':
            print('using sinusoidal embedding with maximum of {} timesteps.'.format(max_reverse_steps))
            self.time_emb = \
            Embedding.SinusoidalPositionalEmbedding(max_reverse_steps, self.time_emb_size, self.device) # can do up to 1000 reverse_steps
            self.E_emb = \
            Embedding.SinusoidalPositionalEmbedding(max_reverse_steps, self.time_emb_size, self.device) # can do up to 1000 reverse_steps
        
        elif self.time_emb_type == 'learnable':
            self.time_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
            self.E_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        elif self.time_emb_type == 'concatenate':
            self.time_emb_size = 1

        if self.time_emb_type != 'concatenate':
            # possibly, remove the mlp and just use the embedding
            self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
            self.E_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)

        if self.x_emb_type == 'mlp':
            self.x_emb = nn.Linear(self.nfeatures, self.x_emb_size).to(self.device)
            self.v_emb = nn.Linear(self.nfeatures, self.x_emb_size).to(self.device)
            self.x_mlp = nn.Sequential(self.x_emb,
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act)
            self.v_mlp = nn.Sequential(self.x_emb,
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act,
                                      nn.Linear(self.x_emb_size, self.x_emb_size), 
                                      self.act)
        elif self.x_emb_type == 'unet':
            assert unet is not None, 'must provide unet model'
            assert False, 'unet for jumptime normalizing flow NYI: what do we do with velocity?'
            self.unet = unet
            self.x_emb_size = self.nfeatures
        
        elif self.x_emb_type == 'concatenate':
            self.x_emb_size = self.nfeatures
        # Neural spline flow (NSF) with dim sample features (V_t) and context features (X_t, t)
        if self.type == 'NSF':
            self.normalizing_flow_model = zuko.flows.NSF(1, # T_k from E_k
                               2*self.x_emb_size + 2*self.time_emb_size,
                               transforms=p_model_normalizing_flow['transforms'], #3
                                hidden_features= [p_model_normalizing_flow['hidden_width']] * p_model_normalizing_flow['hidden_depth'] ) #[128] * 3)
        elif self.type == 'MAF':
            self.normalizing_flow_model = zuko.flows.MAF(1, # T_k from E_k
                               2*self.x_emb_size + 2*self.time_emb_size,
                               transforms=p_model_normalizing_flow['transforms'], #3
                                hidden_features= [p_model_normalizing_flow['hidden_width']] * p_model_normalizing_flow['hidden_depth'] ) #[128] * 3)
        else:
            raise Exception('normalizing flow model type {} not yet implemented'.format(self.type))
        
        # zero the last module of the neural network.
        # todo
    
    
    def _forward(self, x_t, v_t, t, E):
        if self.x_emb_type == 'unet':
            assert False, 'nyi: what do we do with velocity?'
            x_t = self.unet(x_t, t)
        x_t = x_t.reshape(x_t.shape[0], -1)
        v_t = v_t.reshape(v_t.shape[0], -1)
        if self.x_emb_type == 'mlp':
            x_t = self.x_mlp(x_t)
            v_t = self.v_mlp(v_t)
        t = t.reshape(-1, 1)
        E = E.reshape(-1, 1)
        if self.time_emb_type == 'concatenate':
            t = t.repeat(1, self.time_emb_size)
            E = E.repeat(1, self.time_emb_size)
        else:
            t = self.time_mlp(t)#timestep.to(torch.float32))
            E = self.E_mlp(E)#timestep.to(torch.float32))
        return x_t, v_t, t, E

    # return log_prob
    def forward(self, x_t, v_t, t, prev_t, E):
        x_t, v_t, t, E = self._forward(x_t, v_t, t, E)
        prev_t = prev_t.reshape(-1, 1)
        log_p_t_model = self.normalizing_flow_model(torch.cat([x_t, v_t, t, E], dim = -1)).log_prob(prev_t)
        return log_p_t_model
    
    # return v_t
    def sample(self, x_t, v_t, t, E):
        #data_shape = x_t.shape
        x_t, v_t, t, E = self._forward(x_t, v_t, t, E)
        samples = self.normalizing_flow_model(torch.cat([x_t, v_t, t, E], dim = -1)).sample()
        samples = samples[:, 0]
        #samples = samples.reshape(*data_shape)
        #print('output sample shape:', samples.shape)
        return samples