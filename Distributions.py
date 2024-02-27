# file to generate some default data like swiss roll, GMM, levy variables

from sklearn.datasets import make_swiss_roll
from sklearn.mixture import GaussianMixture
import numpy as np
from inspect import signature
import torch
import scipy
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# repeat a tensor so that its last dimensions [1:] match size[1:]
# ideal for working with batches.
def match_last_dims(data, size):
    assert len(data.size()) == 1 # 1-dimensional, one for each batch
    for i in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

''' Generate fat tail distributions'''
# assumes it is a batch size
# is isotropic, just generates a single 'a' tensored to the right shape
def gen_skewed_levy(alpha, 
                    size, 
                    device = None, 
                    isotropic = True,
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
# assumes it is a batch size
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

    '''
        if isotropic:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size[0]), dtype=torch.float32)
            ret = repeat_shape(ret, size)
        else:
            ret = torch.tensor(scipy.stats.levy_stable.rvs(alpha, 0, loc=0, scale=1, size=size), dtype=torch.float32)
        return ret if device is None else ret.to(device)
    else:
        if isotropic:
            ret = torch.randn(size=(size[0],))
            ret = repeat_shape(ret, size)
        else:
            ret = torch.randn(size=size)
        if device is not None:
            ret = ret.to(device)
        return torch.sqrt(a)* ret'''





''' They must have the same signature'''

def sample_2_gmm(n_samples, alpha = None, n = None, std = None, theta = 1.0, weights = None, device = None, normalize=False):
    if weights is None:
        weights = np.array([0.5, 0.5])
    means = np.array([ [theta, 0], [-theta, 0] ])
    gmm = GaussianMixture(n_components=2)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(2)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    return x[torch.randperm(x.size()[0])]

def sample_grid_gmm(n_samples, alpha = None, n = None, std = None, theta = None, weights = None, device = None, normalize=False):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    means = []
    for i in range(n):
        for j in range(n):
            means.append([i, j])
    means = np.array(means)
    gmm = GaussianMixture(n_components=n*n)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = [std*std*np.eye(2) for i in range(n*n)]
    x, _ = gmm.sample(n_samples)
    if normalize:
        x = (x - x.mean()) / x.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    x = torch.tensor(x, dtype = torch.float32)
    return x[torch.randperm(x.size()[0])]


def gen_swiss_roll(n_samples, alpha = None, n = None, std = None, theta = None, weights = None, device = None, normalize=False):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=std)
    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    return torch.tensor(x, dtype = torch.float32)


def sample_grid_sas(n_samples, alpha = 1.8, n = None, std = None, theta = 1.0, weights = None, device = None, normalize=False):
    if weights is None:
        weights = np.array([1 / (n*n) for i in range(n*n)])
    data = std * gen_sas(alpha, size = (n_samples, 2))
    weights = np.concatenate((np.array([0.0]), weights))
    idx = np.cumsum(weights)*n_samples
    for i in range(n):
        for j in range(n):
            # for the moment just selecting exact proportions
            s = int(idx[i*n + j])
            e = int(idx[i*n + j + 1])
            data[s:e] = data[s:e] + torch.tensor([i, j])

    if normalize:
        data = (data - data.mean()) / data.std()
    # don't forget to shuffle rows, otherwise sorted by mixture
    #data = torch.tensor(data, dtype = torch.float32)
    return data[torch.randperm(data.size()[0])]

