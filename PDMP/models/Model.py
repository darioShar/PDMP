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


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module    

#Can predict gaussian_noise, stable_noise, anterior_mean
class MLPModel(nn.Module):
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'one_dimensional_input'
    ]

    def manage_time_embedding(self):
        if self.time_emb_type == 'sinusoidal':
            self.time_emb = \
            Embedding.SinusoidalPositionalEmbedding(10000, self.time_emb_size, self.device)
        
        elif self.time_emb_type == 'learnable':
            self.time_emb = nn.Linear(1, self.time_emb_size).to(self.device) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        elif self.time_emb_type == 'one_dimensional_input':
            self.additional_dim += 1
        else:
            raise ValueError('Wrong time embedding type : {}'.format(self.time_emb_type))
        
        if self.time_emb_type != 'one_dimensional_input':
            # possibly, remove the mlp and just use the embedding
            self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
        
    def __init__(self, nfeatures, device, p_model_mlp, noising_process):
        super(MLPModel, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures # x_t and v_t
        self.beta =             p_model_mlp['beta'] # 0.2
        self.threshold =        p_model_mlp['threshold'] # 20
        self.time_emb_type =    p_model_mlp['time_emb_type'] 
        self.time_emb_size =    p_model_mlp['time_emb_size']
        self.nblocks =          p_model_mlp['nblocks'] 
        self.nunits =           p_model_mlp['nunits']
        self.skip_connection =  p_model_mlp['skip_connection']
        self.group_norm =       p_model_mlp['group_norm']
        self.dropout_rate =     p_model_mlp['dropout_rate']
        self.device =           device
        self.noising_process =  noising_process
        # to be computed later depending on chosen architecture
        self.additional_dim =   0 

        assert noising_process in ['diffusion', 'ZigZag'], 'only supports MLP Model for ZigZag and diffusion'
        assert self.time_emb_type in self.possible_time_embeddings
        
        # dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.group_norm_in = nn.LayerNorm([self.nunits]) if self.group_norm else nn.Identity()
        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        self.manage_time_embedding()

        # Here we construct the model 'blocks'
        self.linear_in =  nn.Linear(self.nfeatures + self.additional_dim, self.nunits) # for ZigZag, take d dimension and output 2d dimension
        self.inblock = nn.Sequential(self.linear_in,
                                     self.group_norm_in, 
                                     self.act)
        self.midblocks = nn.ModuleList([Block.DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.group_norm,
                                            time_emb_size = self.time_emb_size if self.time_emb_type != 'one_dimensional_input' else False,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        # 2d outputs, v=1 for the first d ones, v=-1 for the other d ones
        out_dim = self.nfeatures if noising_process == 'diffusion' else 2*self.nfeatures
        self.outblocks = zero_module(nn.Linear(self.nunits, out_dim))
    
    def forward(self, x, timestep):
        timestep = timestep.unsqueeze(1).unsqueeze(2) # add batch and channel dimensions
        
        inp = [x]
        # managing time variable
        if self.time_emb_type == 'one_dimensional_input':
            inp += [timestep]
            # set to zero because we must feed a dummy to midblocks
            t = torch.zeros(size=timestep.size())
        else:
            t = self.time_mlp(timestep.to(torch.float32))
        
        # input
        val = torch.hstack(inp)
        # compute
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val, t)
        
        val = self.outblocks(val)
        if self.noising_process == 'ZigZag':
            # split and stack last dimensions to create new channels
            val = torch.split(val, val.shape[-1]//2, dim=-1)
            val = torch.concatenate(val, dim = 1) # add channels
            val = torch.nn.functional.softplus(val, beta=self.beta, threshold=self.threshold)
        return val
    
    