from sklearn.datasets import make_swiss_roll
from sklearn.mixture import GaussianMixture
import numpy as np
from inspect import signature
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from Distributions import *

''' 
Wrapper class to call all the generation functions
We can specify arbitrary *args and **kwargs. They must match the selected
function signature, but we have made it so all the distributions can be genrated
by functions with the same signature, see Distributions file.
'''
class Generator(Dataset):
    available_distributions = \
        {'gmm_2': sample_2_gmm,
        'gmm_grid': sample_grid_gmm,
        'swiss_roll':gen_swiss_roll,
        'skewed_levy': gen_skewed_levy,
        'sas': gen_sas,
        #'gaussian_noising': gaussian_noising,
        #'stable_noising': stable_noising,
        'sas_grid': sample_grid_sas
        }
    def __init__(self, 
                 dataset = None, # distribution to select 
                 transform = None,  # transform to apply to each generated sample
                 *args, **kwargs):
        assert dataset is not None
        self.transform = lambda x: x
        if transform is not None:
            self.transform = transform
        self.kwargs = kwargs
        self.args = args
        self.samples = None
        try:
            self.generator = self.available_distributions[dataset]
        except:
            raise Exception('Unknown distribution to sample from. \
            Available distributions: {}'.format(list(self.available_distributions.keys())))
    
    
    def setTransform(self, transform):
        self.transform = transform
    
    # replaces missing elements of args and kwargs by those of self.args and self.kwargs
    # replaces all elements of kwargs if non void
    def setParams(self, *args, **kwargs):
        if args == () and kwargs == {}:
            raise Exception('Given void parameters')
        
        self.args = tuple(map(lambda x, y: y if y is not None else x, self.args, args))
        self.kwargs.update(kwargs)

    def getSignature(self):
        return signature(self.generator)
    
    def getName(self):
        return 
    
    # generate samples using self.args instead of args if it is not ()
    # updates self.kwargs by kwargs.
    def generate(self, *args, **kwargs):
        tmp_kwargs = self.kwargs | kwargs        
        if args == () and kwargs == {} and self.kwargs == {}:
            raise Exception('No parameters for data generation')
        if args == ():
            self.samples = self.transform(self.generator(*self.args, **tmp_kwargs))
            return self.samples
        self.samples = self.transform(self.generator(*args, **tmp_kwargs))
        return self.samples

    def __len__(self):
        return self.samples.size()
    
    def __getitem__(self, idx):
        return self.samples[idx]
    