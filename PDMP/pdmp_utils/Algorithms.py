from tqdm import tqdm

import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim

import PDMP.pdmp_utils.Data as Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from PDMP.pdmp_utils.Distributions import *


# for the computations
def compute_gamma_factor(g, bg, alpha, t, a_t, a_t_1):
    return (1 - (g[t]**(2 / alpha) * (1 - bg[t-1])**(2 / alpha) * a_t_1) 
           / ((1 - bg[t])**(2 / alpha) * a_t))

def compute_sigma_tilde(g, bg, alpha, t, a_t, a_t_1):
    return compute_gamma_factor(g, bg, alpha,t, a_t, a_t_1) \
                * a_t_1 * (1 - bg[t-1])**(2/alpha)

# lambda, and not lambda squared
def compute_lambda(g, bg, alpha, t, a_t, a_t_1):
    return compute_gamma_factor(g, bg, alpha, t, a_t, a_t_1) \
            * (1 - bg[t-1])**(1/alpha)    

def compute_sigma(g, bg, alpha, a_skewed):
    # torch.cumsum(a_skewed*((1 - gammas)/bargammas)**(2 / alpha)) * bargammas**(2 / alpha)
    return torch.cumsum(a_skewed*((1 - g)/bg)**(2 / alpha)) * bg**(2 / alpha)

def compute_a_t(g, bg, alpha, t, a_t_prime, a_t_1):
    return g[t]**(2/alpha)*a_t_1 \
                        + (1 - g[t])**(2 / alpha) * a_t_prime


def gen_noise_schedule(steps):
    # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
    s = 0.008
    timesteps = torch.tensor(range(0, steps), dtype=torch.float32)
    schedule = torch.cos((timesteps / steps + s) / (1 + s) * torch.pi / 2)**2

    baralphas = schedule / schedule[0]
    betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
    alphas = 1 - betas

    # linear schedule for gamma
    # for the moment let's use the same schedule alphas
    gammas = alphas
    bargammas = np.cumprod(gammas)
    
    return gammas, bargammas

''' Generate noising processes'''
''' DEPCRECATED: SEE DIFFUSION CLASS'''

def gaussian_noising(Xbatch, t, baralphas, device = None):
    eps = torch.randn(size=Xbatch.size())
    if device is not None:
        eps = eps.to(device)
    noised = (baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
    return noised, eps

def stable_noising(Xbatch, t, alpha, bargammas, a = None, device = None):
    eps = gen_sas(alpha, Xbatch.size(), a = a, device=device)
    if device is not None:
        eps = eps.to(device)
    noised = (bargammas[t] ** (1/alpha)) * Xbatch + ((1 - bargammas[t]) ** (1/alpha))* eps
    #noised = (bargammas[t] ** (1/alpha)).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - bargammas[t]) ** (1/alpha)).repeat(1, Xbatch.shape[1]) * eps
    return noised, eps

''' Generate samples from model'''

''' DEPCRECATED: SEE DIFFUSION CLASS'''
# generate samples
def sample_stable(model, device, data_size, print_progression=True):
    if print_progression:
        tqdm._instances.clear()
        pbar = tqdm(total = model.diffusion_steps)
    alpha = model.alpha
    bargammas = model.bargammas.to(device)
    gammas = model.gammas.to(device)
    # nfeatures = model.nfeatures
    
    with torch.no_grad():
        model.eval()
        x = gen_sas(alpha, size = data_size, device=device)
        #x = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, size=(nsamples, nfeatures)), dtype=torch.float32).to(device)
        #torch.sqrt(a_skewed) * torch.randn(size=(nsamples, nfeatures)).to(device)
        xt = [x]
        if print_progression:
            pbar.update(1)
        
        # repeat last dimension
        skewed_levy_data = Data.Generator('skewed_levy', alpha = model.alpha, device=device)
        #skewed_levy_data.setTransform((lambda x: x.unsqueeze(1).repeat((1, nfeatures))))
        
        for t in range(model.diffusion_steps-1, 0, -1):
            a_t_1 = skewed_levy_data.generate(size = data_size)
            a_t_prime = skewed_levy_data.generate(size = data_size)

            g = match_last_dims(gammas, a_t_1.size())
            bg = match_last_dims(bargammas, a_t_1.size())

            a_t = compute_a_t(g, bg, alpha, t, a_t_prime, a_t_1)
            #a_t = (1 / (1 - bargammas[t])**(2/alpha))*(gammas[t]**(2 / alpha) \
            #        *a_t_1*(1 - bargammas[t-1])**(2/alpha) \
            #        + (1 - gammas[t])**(2/alpha)*a_t_prime)
                #print(torch.mean(a_t))
            if t == 1:
                a_t_1 = torch.zeros(data_size).to(device)
            
            predicted_noise = model(x, 
                                    torch.full([data_size[0]], t).to(device), 
                                    a_t_prime, 
                                    a_t_1)
            #if model.value_to_predict == 'gaussian_noise':
            #    predicted_noise *= torch.sqrt(a_t)
            
            factor = (compute_gamma_factor(g, bg, alpha, t, a_t, a_t_1) * (1 - bg[t])**(1/alpha)).to(device)
            sigma_tilde = compute_sigma_tilde(g, bg, alpha, t, a_t, a_t_1).to(device)
            #update_term = - ((1 / gammas[t]**(1 / alpha)) * factor).to(device) * predicted_noise + torch.sqrt(sigma_tilde).to(device) * torch.randn(size=(nsamples, nfeatures)).to(device)
            x = (1 / gammas[t]**(1 / alpha)) *(x - factor* predicted_noise)
            if t > 1:
                x += torch.sqrt(sigma_tilde) \
                        * torch.randn(size=data_size).to(device)
            xt += [x]
            if print_progression:
                pbar.update(1)
        if print_progression:
            pbar.close()
            tqdm._instances.clear()
        return x, xt
    

