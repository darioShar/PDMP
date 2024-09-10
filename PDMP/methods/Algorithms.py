from tqdm import tqdm

import numpy as np
import torch

import bem.datasets.Data as Data
from bem.datasets.Distributions import *


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

