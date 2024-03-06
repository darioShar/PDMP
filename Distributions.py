from sklearn.datasets import make_swiss_roll
from sklearn.mixture import GaussianMixture
import numpy as np
from inspect import signature
import torch
import scipy
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor



'''
This file contains the distributions that are used in the experiments.
'''



# repeat a tensor so that its last dimensions [1:] match size[1:]
# ideal for working with batches.
def match_last_dims(data, size):
    assert len(data.size()) == 1 # 1-dimensional, one for each batch
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

''' Generate fat tail distributions'''

# generates a levy stable distribution skewed to the right
def gen_skewed_levy(alpha, 
                    size, # assumes that the first dimension of size is the batch size
                    device = None, 
                    isotropic = True, # if True generates a single a for all the batch
                    clamp_a = 2000):
    if (alpha > 2.0 or alpha <= 0.):
        raise Exception('Wrong value of alpha ({}) for skewed levy r.v generation'.format(alpha))
    if alpha == 2.0:
        ret = 2 * torch.ones(size)
        return ret if device is None else ret.to(device)
    # generates the alplha/2, 1, 0, 2*np.cos(np.pi*alpha/4)**(2/alpha)
    if isotropic:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size[0]), dtype=torch.float32)
        ret = match_last_dims(ret, size)
    else:
        ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha/2, 1, loc=0, scale=2*np.cos(np.pi*alpha/4)**(2/alpha), size=size), dtype=torch.float32)
    ret = torch.clamp(ret, 0., clamp_a)
    return ret if device is None else ret.to(device)


#symmetric alpha stable noise of scale 1
# can generate from totally skewed noise if provided
def gen_sas(alpha, 
            size, 
            a = None, 
            device = None, 
            isotropic = True,
            clamp_eps = 20000):
    if a is None:
        a = gen_skewed_levy(alpha, size, device = device, isotropic = isotropic)
    ret = torch.randn(size=size, device=device)
    
    #if device is not None:
    #    ret = ret.to(device)
    #ret = repeat_shape(ret, size)
    return torch.clamp(torch.sqrt(a)* ret, -clamp_eps, clamp_eps)


''' All functions here must have the same signature'''

def sample_2_gmm(nsamples, alpha = None, n_mixture = None, std = None, theta = 1.0, weights = None, device = None, normalize=False, nfeatures=1):
    if weights is None:
        weights = np.array([0.5, 0.5])
    means = np.array([ [theta, 0], [-theta, 0] ])
    gmm = GaussianMixture(n_components=2)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(2)]
    x, _ = gmm.sample(nsamples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    return x[torch.randperm(x.size()[0])]

def sample_grid_gmm(nsamples, alpha = None, n_mixture = None, std = None, theta = None, weights = None, device = None, normalize=False, nfeatures=1):
    if weights is None:
        weights = np.array([1 / (n_mixture*n_mixture) for i in range(n_mixture*n_mixture)])
    means = []
    for i in range(n_mixture):
        for j in range(n_mixture):
            means.append([i, j])
    means = np.array(means)
    gmm = GaussianMixture(n_components=n_mixture*n_mixture)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(n_mixture*n_mixture)]
    x, _ = gmm.sample(nsamples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    return x[torch.randperm(x.size()[0])]


def gen_swiss_roll(nsamples, alpha = None, n_mixture = None, std = None, theta = None, weights = None, device = None, normalize=False, nfeatures=1):
    x, _ = make_swiss_roll(nsamples=nsamples, noise=std)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    return torch.tensor(x, dtype = torch.float32)


def sample_grid_sas(nsamples, alpha = 1.8, n_mixture = None, std = None, theta = 1.0, weights = None, device = None, normalize=False, nfeatures=1):
    if weights is None:
        weights = np.array([1 / (n_mixture*n_mixture) for i in range(n_mixture*n_mixture)])
    data = std * gen_sas(alpha, size = (nsamples, 2))
    weights = np.concatenate((np.array([0.0]), weights))
    idx = np.cumsum(weights)*nsamples
    for i in range(n_mixture):
        for j in range(n_mixture):
            # for the moment just selecting exact proportions
            s = int(idx[i*n_mixture + j])
            e = int(idx[i*n_mixture + j + 1])
            data[s:e] = data[s:e] + torch.tensor([i, j])

    if normalize:
        data = (data - data.mean()) / data.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    #data = torch.tensor(data, dtype = torch.float32)
    return data[torch.randperm(data.size()[0])]

