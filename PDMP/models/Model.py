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



class ExpBlock(nn.Module):
    def __init__(self, 
                 nunits, 
                 dropout_rate, 
                 activation = nn.SiLU):
        super(ExpBlock, self).__init__()
        
        self.act = activation(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.mlp_exp = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits))
        self.mlp_base = nn.Sequential(self.dropout, 
                                   nn.Linear(nunits, nunits))
        
    def forward(self, x: torch.Tensor):
        x_exp = nn.functional.relu(self.mlp_exp(x))
        x_base = self.mlp_base(x)
        return x_base * torch.exp(x_exp)
    

#Can predict gaussian_noise, stable_noise, anterior_mean
class LevyDiffusionModel(nn.Module):
    possible_time_embeddings = [
        'sinusoidal',
        'learnable',
        'one_dimensional_input'
    ]

    def __init__(self, nfeatures, device, p_model_mlp, noising_process):
        super(LevyDiffusionModel, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures # x_t and v_t
        self.use_a_t =          p_model_mlp['use_a_t']
        self.no_a =             p_model_mlp['no_a']
        self.a_pos_emb =        p_model_mlp['a_pos_emb']
        self.a_emb_size =       p_model_mlp['a_emb_size']
        self.time_emb_type =    p_model_mlp['time_emb_type'] 
        self.time_emb_size =    p_model_mlp['time_emb_size']
        self.nblocks =          p_model_mlp['nblocks'] 
        self.nunits =           p_model_mlp['nunits']
        self.skip_connection =  p_model_mlp['skip_connection']
        self.group_norm =       p_model_mlp['group_norm']
        self.dropout_rate =     p_model_mlp['dropout_rate']
        self.compute_gamma =    p_model_mlp['compute_gamma']
        self.device =           device
        self.noising_process =  noising_process
        # to be computed later depending on chosen architecture
        self.additional_dim =   0 

        assert noising_process in ['diffusion', 'ZigZag'], 'only supports MLP for ZigZag and diffusion'
        assert self.time_emb_type in self.possible_time_embeddings
        
        # for dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.group_norm_in = nn.LayerNorm([self.nunits]) if self.group_norm else nn.Identity()
        self.act = nn.SiLU(inplace=False)
        
        # manage time embedding type
        if self.time_emb_type == 'sinusoidal':
            self.time_emb = \
            Embedding.SinusoidalPositionalEmbedding(self.diffusion_steps, 
                                            self.time_emb_size, self.device)
        
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
        
        # I changed input a_t size from 2 to 1, since we will take it to be equal on both dims
        if self.no_a or self.a_pos_emb:
            self.additional_dim += 0
        elif self.use_a_t:
            self.additional_dim += 1
        else:
            self.additional_dim += 2
            
        # manage a_t embedding
        if self.a_pos_emb:
            dim = 1 if self.use_a_t else 2
            self.a_emb = nn.Linear(dim, self.a_emb_size)
            self.a_mlp = nn.Sequential(self.a_emb,
                                          self.act,
                                          nn.Linear(self.a_emb_size, self.a_emb_size), 
                                          self.act)
        
        # possible exp block, at the end
        features = self.nfeatures if self.noising_process == 'diffusion' else 2*self.nfeatures
        self.linear_in =  nn.Linear(features + self.additional_dim, self.nunits) # for ZigZag, take d dimension and output 2d dimension
        
        self.inblock = nn.Sequential(self.linear_in,
                                     self.group_norm_in, 
                                     self.act)
        
        self.midblocks = nn.ModuleList([Block.DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.group_norm,
                                            time_emb_size = self.time_emb_size \
                                                if self.time_emb_type != 'one_dimensional_input'\
                                                else False,
                                            a_emb_size = self.a_emb_size\
                                                if self.a_pos_emb \
                                                else False,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        self.midblocks_exp = nn.ModuleList([ExpBlock(self.nunits, self.dropout_rate) 
                                            for _ in range(self.nblocks)])
        
        #self.endblock_exp = ExpBlock(self.nfeatures, self.dropout_rate)
        # add one conditioned block and one MLP for both mean and variance computation
        self.outblocks_mean = nn.ModuleList([
                Block.DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.group_norm,
                                                time_emb_size = self.time_emb_size \
                                                    if self.time_emb_type != 'one_dimensional_input'\
                                                    else False,
                                                a_emb_size = self.a_emb_size\
                                                    if self.a_pos_emb \
                                                    else False,
                                                activation = nn.SiLU),
                zero_module(nn.Linear(self.nunits, self.nfeatures)) # 2d outputs, v=1 for the first d ones, v=-1 for the other d ones
            ])
        if self.compute_gamma:
            self.outblocks_var = nn.ModuleList([
                Block.DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.group_norm,
                                                time_emb_size = self.time_emb_size \
                                                    if self.time_emb_type != 'one_dimensional_input'\
                                                    else False,
                                                a_emb_size = self.a_emb_size\
                                                    if self.a_pos_emb \
                                                    else False,
                                                activation = nn.SiLU),
                zero_module(nn.Linear(self.nunits, self.nfeatures)), #1 if self.isotropic else self.features),
                #nn.Sigmoid()
            ])
        
        

    def forward(self, x, timestep, a_t_prime = None, a_t_1 = None):
        ''' g = match_last_dims(self.gammas, x.size())
        bg = match_last_dims(self.bargammas, x.size())

        if a_t_prime is None:
            a_t_prime = torch.ones(size = x.shape).to(self.device)
        if a_t_1 is None:
            a_t_1 = torch.ones(size = x.shape).to(self.device)
        
        a_t = Algo.compute_a_t(g, bg, self.alpha, timestep, a_t_prime, a_t_1)

        # take log of a's to reduce variance
        a_t = torch.log(a_t)
        a_t_prime = torch.log(a_t_prime)
        a_t_1 = torch.log(a_t_1)'''
        
        # inp = [x]
        
        a_t_prime = torch.ones(size = x.shape).to(self.device)
        a_t_1 = torch.ones(size = x.shape).to(self.device)
        a_t = torch.ones(size = x.shape).to(self.device)

        timestep = timestep.unsqueeze(1).unsqueeze(2) # add batch and channel dimensions
        
        # depending on positional embedding and how many a's
        if not self.a_pos_emb:
            if self.use_a_t:
                inp += [a_t]
            elif not self.no_a:
                inp += [a_t_prime, a_t_1]
            # set to zero because we must feed a dummy to midblocks
            a = torch.zeros(size=timestep.size())
        else:
            if self.use_a_t:
                a = self.a_mlp(a_t)
            elif not self.no_a:
                a = self.a_mlp(torch.hstack([a_t_prime, a_t_1]))
        
        # same but for time variable
        if self.time_emb_type == 'one_dimensional_input':
            inp += [timestep]
            # set to zero because we must feed a dummy to midblocks
            t = torch.zeros(size=timestep.size())
        else:
            t = self.time_mlp(timestep.to(torch.float32))
        
        # input
        val = x #torch.hstack(inp)
        # compute
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val, t, a)
        
        val_1 = self.outblocks_mean[0](val, t, a)
        val_1 = self.outblocks_mean[1](val_1)
        if self.compute_gamma:
            val_2 = self.outblocks_var[0](val, t, a)
            val_2 = self.outblocks_var[1](val_2)
            #val_2 = self.outblocks_var[2](val_2)
            #if self.isotropic:
            #    val_2 = val_2.expand(*val_2.shape[:-1], val_1.shape[-1])
            return torch.concat([val_1, val_2], dim = 1) # concat on channels dim
        else:
            if self.noising_process == 'ZigZag':
                # aply this function for positive output and better behaviour around 1.
                val_1 = torch.log(1 + torch.exp(val_1)) / np.log(2) # so that's it 1 at val_1=0
            return val_1
            
        # duplicate last
        #val = val.expand(val.shape[0], 2*val.shape[1], *val.shape[2:])

    
    