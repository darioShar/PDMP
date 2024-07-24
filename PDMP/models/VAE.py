import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import zuko

from torch import Tensor
from torch.distributions import Distribution, Normal, Bernoulli, Independent
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from PDMP.datasets import is_image_dataset, affine_transform, inverse_affine_transform


###### ATTENTION ########
''' 
Since model_vae output is a Bernoulli, it is between 0 and 1...
so I cannot fit the model on -1, 1 holy shit.
'''


hidden_dim_codec=1280

class ELBO(nn.Module):
    def __init__(
        self,
        encoder: zuko.flows.LazyDistribution,
        decoder: zuko.flows.LazyDistribution,
        prior: zuko.flows.LazyDistribution,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def forward(self, x, c = None):
        q = self.encoder(x)
        z = q.rsample()
        if c is not None:
            kl_loss = self.prior(c).log_prob(z) - q.log_prob(z)
            return self.decoder(z).log_prob(x) + 0.1*kl_loss
        else:
            kl_loss = self.prior().log_prob(z) - q.log_prob(z)
            return self.decoder(z).log_prob(x) + 0.0001*kl_loss
    


class GaussianModel(zuko.flows.LazyDistribution):
    def __init__(self, features: int, context: int):
        super().__init__()
        
        self.features = features
        self.hyper = nn.Sequential(
            nn.Linear(context, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, 2 * features),
        )

    def forward(self, c) -> Distribution:
        phi = self.hyper(c)
        mu, log_sigma = phi.chunk(2, dim=-1)
        #mu = torch.zeros((hidden_dim_codec, self.features))
        #log_sigma = torch.zeros((hidden_dim_codec, self.features))

        return Independent(Normal(mu, log_sigma.exp()), 1)


class BernoulliModel(zuko.flows.LazyDistribution):
    def __init__(self, features: int, context: int):
        super().__init__()

        self.features = features
        self.hyper = nn.Sequential(
            nn.Linear(context, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, features),
        )

    def forward(self, c) -> Distribution:
        phi = self.hyper(c)
        rho = torch.sigmoid(phi)
        #rho = 0.5*torch.ones((hidden_dim_codec, self.features))

        return Independent(Bernoulli(rho), 1)

class VAESimple(nn.Module):

    def __init__(self, nfeatures, p_model_nf):
        super(VAESimple, self).__init__()
        
        self.nfeatures=nfeatures

        assert self.nfeatures == 1024
        
        self.act = nn.SiLU(inplace=False)
        
        self.encoder = GaussianModel(16, 1024)
        self.decoder = GaussianModel(1024, 16)

        self.prior = zuko.flows.MAF(
            features=16,
            transforms=3,
            hidden_features=(256, 256),
        )

        self.elbo = ELBO(self.encoder, self.decoder, self.prior)
        

    # elbo loss
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = inverse_affine_transform(x)
        return self.elbo(x)
    
    # return v_t
    def sample(self, nsamples):
        z = self.prior().sample((nsamples,))
        x = self.decoder(z).mean.reshape(-1, 1, 32, 32)
        return affine_transform(x)



class VAE(nn.Module):

    def __init__(self, nfeatures, p_model_nf):
        super(VAE, self).__init__()
        
        self.nfeatures=nfeatures
        self.hidden_features = p_model_nf['model_vae_hidden_features']

        time_mlp_hidden_features = p_model_nf['model_vae_t_hidden_width'] #128 # 128
        time_mlp_output_dim = p_model_nf['model_vae_t_emb_size'] #32 # 32
        
        x_mlp_hidden_features = hidden_dim_codec 
        x_mlp_output_dim = p_model_nf['model_vae_x_emb_size'] #64 # 256

        # assert self.nfeatures == 1024
        
        self.act = nn.SiLU(inplace=False)
        
        self.encoder = GaussianModel(self.hidden_features, self.nfeatures)
        self.decoder = GaussianModel(self.nfeatures, self.hidden_features)

        self.prior = zuko.flows.MAF(
            features=self.hidden_features,
            context=x_mlp_output_dim + time_mlp_output_dim,
            transforms=4,
            hidden_features=(256, 256),
        )

        self.elbo = ELBO(self.encoder, self.decoder, self.prior)
        

        self.time_mlp = nn.Sequential(nn.Linear(1, time_mlp_hidden_features),
                                    self.act,
                                    nn.Linear(time_mlp_hidden_features, time_mlp_hidden_features), 
                                    self.act,
                                    nn.Linear(time_mlp_hidden_features, time_mlp_output_dim), 
                                    self.act)

        self.x_mlp = nn.Sequential(nn.Linear(self.nfeatures, x_mlp_hidden_features),
                                    self.act,
                                    nn.Linear(x_mlp_hidden_features, x_mlp_hidden_features), 
                                    self.act,
                                    nn.Linear(x_mlp_hidden_features, x_mlp_output_dim), 
                                    self.act)
    
    # elbo loss
    def forward(self, x_t, v_t, t):
        x_t, t = self._forward(x_t, t)
        v_t = v_t.reshape(x_t.shape[0], -1)
        return self.elbo(v_t, torch.cat([x_t, t], dim = -1))

    def _forward(self, x_t, t):
        x_t = x_t.reshape(x_t.shape[0], -1)
        x_t = self.x_mlp(x_t)
        t = t.reshape(-1, 1)
        t = self.time_mlp(t)#timestep.to(torch.float32))
        return x_t, t
    
    # return v_t
    def sample(self, x_t, t):
        data_shape = x_t.shape
        x_t, t = self._forward(x_t, t)
        z = self.prior(torch.cat([x_t, t], dim = -1)).sample((1,))
        x = self.decoder(z).mean.reshape(*data_shape)
        return x
    
class MultiVAE(nn.Module):

    def __init__(self, nfeatures, n_vae, time_horizon, p_model_nf):
        super(MultiVAE, self).__init__()
        
        self.nfeatures=nfeatures
        self.n_vae = n_vae
        self.time_horizon = time_horizon
        
        # assert self.nfeatures == 1024, 'only implemented on 32x32 mnist at the moment'
        # assert self.time_horizon == 10, 'only implements time horizon ==10 for the moment'
        assert self.n_vae == 16, 'only implements n_vae==16 for the moment'

        self.vae_t_bins = self.vae_time_bins()
        self.vae_list = nn.ModuleList([VAE(self.nfeatures, p_model_nf=p_model_nf) for _ in range(self.n_vae)])
        
    # which time horizon the VAEs should manage
    def vae_time_bins(self):
        # return list of size n_vae + 1. each vae manages one bin
        time_values = np.array([0.0, 1.5, 5, 10]) * self.time_horizon / 10.
        n = [7, 6, 3]
        assert sum(n) == 16, 'must use 16 VAE'
        time_bins = np.concatenate([np.linspace(time_values[i], time_values[i+1], n[i], endpoint=False) for i in range(len(n))])
        # add last value going to 10 to define the last bin.
        time_bins = np.concatenate([time_bins, [self.time_horizon]]) # goes to time horizon 10
        return time_bins

    # dispatch batch elements in their corresponding bin according to the current timestep
    def dispatch_batch(self, x_t, t):
        return [(t <= self.vae_t_bins[i+1]) & (t > self.vae_t_bins[i]) for i in range(len(self.vae_t_bins)-1)]
    
    # return bin i for time t
    def get_bin_t(self, t):
        assert not (t != t[0]).any(), 'for sampling all t\'s in t batch must be equal'
        return np.where((t[0] <= self.vae_t_bins[1:]) & (t[0] > self.vae_t_bins[:-1]))[0][0]

    # elbo loss
    def forward(self, x_t, v_t, t):
        time_bins_mask = self.dispatch_batch(x_t, t)
        elbo = torch.zeros_like(t)
        for i, t_mask in enumerate(time_bins_mask):
            if t_mask.sum() == 0:
                # no x_t, v_t in this time bin
                continue
            #print('i', i)
            #print('t_mask', t_mask.shape, len(np.where(t_mask.cpu())[0]))
            #print('t', t.shape)
            #print('x_t', x_t.shape)
            #print('v_t', v_t.shape)
            #print('x_t_mask', x_t[t_mask].shape)
            #print('v_t_mask', v_t[t_mask].shape)
            #print('t_mask', t[t_mask].shape)
            x_t_mask = x_t[t_mask]
            v_t_mask = v_t[t_mask]
            t_masked = t[t_mask]
            elbo[t_mask] += self.vae_list[i](x_t_mask, v_t_mask, t_masked)
        return elbo
    
    # return v_t
    def sample(self, x_t, t):
        #i = self.get_bin_t(t)
        #return self.vae_list[i].sample(x_t, t)
        v_t = torch.zeros_like(x_t)
        time_bins_mask = self.dispatch_batch(x_t, t)
        for i, t_mask in enumerate(time_bins_mask):
            if t_mask.sum() == 0:
                # no x_t, v_t in this time bin
                continue
            #print('i', i)
            #print('t_mask', t_mask.shape, len(np.where(t_mask.cpu())[0]))
            #print('t', t.shape)
            #print('x_t', x_t.shape)
            #print('v_t', v_t.shape)
            #print('x_t_mask', x_t[t_mask].shape)
            #print('v_t_mask', v_t[t_mask].shape)
            #print('t_mask', t[t_mask].shape)
            x_t_mask = x_t[t_mask]
            v_t_mask = v_t[t_mask]
            t_masked = t[t_mask]
            v_t[t_mask] = self.vae_list[i].sample(x_t_mask, t_masked)
        return v_t
    

class VAEJumpTime(nn.Module):

    def __init__(self, nfeatures, p_model_nf):
        super(VAEJumpTime, self).__init__()
        
        self.nfeatures=nfeatures
        
        time_mlp_hidden_features = p_model_nf['model_vae_t_hidden_width'] #128 # 128
        time_mlp_output_dim = p_model_nf['model_vae_t_emb_size'] #32 # 32
        
        x_mlp_hidden_features = hidden_dim_codec 
        x_mlp_output_dim = p_model_nf['model_vae_x_emb_size'] #64 # 256

        assert self.nfeatures == 1024
        
        self.act = nn.SiLU(inplace=False)
        
        self.encoder = GaussianModel(16, self.nfeatures+1)
        self.decoder = GaussianModel(self.nfeatures+1, 16)

        self.prior = zuko.flows.MAF(
            features=16,
            context=x_mlp_output_dim + time_mlp_output_dim,
            transforms=3,
            hidden_features=(256, 256),
        )

        self.elbo = ELBO(self.encoder, self.decoder, self.prior)
        

        self.time_mlp = nn.Sequential(nn.Linear(1, time_mlp_hidden_features),
                                    self.act,
                                    nn.Linear(time_mlp_hidden_features, time_mlp_hidden_features), 
                                    self.act,
                                    nn.Linear(time_mlp_hidden_features, time_mlp_output_dim), 
                                    self.act)

        self.x_mlp = nn.Sequential(nn.Linear(1024, x_mlp_hidden_features),
                                    self.act,
                                    nn.Linear(x_mlp_hidden_features, x_mlp_hidden_features), 
                                    self.act,
                                    nn.Linear(x_mlp_hidden_features, x_mlp_output_dim), 
                                    self.act)
    
    # elbo loss
    def forward(self, x_t, v_t, t, t_prev, E):
        x_t, t = self._forward(x_t, t)
        v_t = v_t.reshape(x_t.shape[0], -1)
        t_prev = t_prev.reshape(-1, 1)
        return self.elbo(torch.cat([v_t, t_prev], dim = -1), torch.cat([x_t, t], dim = -1))

    def _forward(self, x_t, t):
        x_t = x_t.reshape(x_t.shape[0], -1)
        x_t = self.x_mlp(x_t)
        t = t.reshape(-1, 1)
        t = self.time_mlp(t)#timestep.to(torch.float32))
        return x_t, t
    
    # return v_t
    def sample(self, x_t, v_t, t, E):
        x_t, t = self._forward(x_t, t)
        z = self.prior(torch.cat([x_t, t], dim = -1)).sample((1,))[0]
        x = self.decoder(z).mean
        v_t, t_prev = x[:, :-1], x[:, -1]
        v_t = v_t.reshape(-1, 1, 32, 32)
        return t_prev, v_t
    

    
class MultiVAEJumpTime(nn.Module):

    def __init__(self, nfeatures, n_vae, time_horizon, p_model_nf):
        super(MultiVAEJumpTime, self).__init__()
        
        self.nfeatures=nfeatures
        self.n_vae = n_vae
        self.time_horizon = time_horizon
        
        assert self.nfeatures == 1024, 'only implemented on 32x32 mnist at the moment'
        #assert self.time_horizon == 10, 'only implements time horizon ==10 for the moment'
        assert self.n_vae == 16, 'only implements n_vae==16 for the moment'

        self.vae_t_bins = self.vae_time_bins()
        self.vae_list = nn.ModuleList([VAEJumpTime(self.nfeatures, p_model_nf=p_model_nf) for _ in range(self.n_vae)])
        
    # which time horizon the VAEs should manage
    def vae_time_bins(self):
        # return list of size n_vae + 1. each vae manages one bin
        time_values = np.array([0.0, 1.5, 5, 10]) * self.time_horizon / 10.
        n = [7, 6, 3]
        assert sum(n) == 16, 'must use 16 VAE'
        time_bins = np.concatenate([np.linspace(time_values[i], time_values[i+1], n[i], endpoint=False) for i in range(len(n))])
        # add last value going to 10 to define the last bin.
        time_bins = np.concatenate([time_bins, [self.time_horizon]]) # goes to time horizon 10
        return time_bins

    # dispatch batch elements in their corresponding bin according to the current timestep
    def dispatch_batch(self, x_t, t):
        return [(t <= self.vae_t_bins[i+1]) & (t > self.vae_t_bins[i]) for i in range(len(self.vae_t_bins)-1)]
    
    # return bin i for time t
    def get_bin_t(self, t):
        assert not (t != t[0]).any(), 'for sampling all t\'s in t batch must be equal'
        return np.where((t[0] <= self.vae_t_bins[1:]) & (t[0] > self.vae_t_bins[:-1]))[0][0]

    # elbo loss
    def forward(self, x_t, v_t, t, t_prev, E):
        time_bins_mask = self.dispatch_batch(x_t, t)
        elbo = torch.zeros_like(t)
        for i, t_mask in enumerate(time_bins_mask):
            if t_mask.sum() == 0:
                # no x_t, v_t in this time bin
                continue
            #print('i', i)
            #print('t_mask', t_mask.shape, len(np.where(t_mask.cpu())[0]))
            #print('t', t.shape)
            #print('x_t', x_t.shape)
            #print('v_t', v_t.shape)
            #print('x_t_mask', x_t[t_mask].shape)
            #print('v_t_mask', v_t[t_mask].shape)
            #print('t_mask', t[t_mask].shape)
            x_t_mask = x_t[t_mask]
            v_t_mask = v_t[t_mask]
            t_masked = t[t_mask]
            t_prev_masked = t_prev[t_mask]
            if E is not None:
                E_masked = E[t_mask]
            else:
                E_masked = None
            elbo[t_mask] += self.vae_list[i](x_t_mask, v_t_mask, t_masked, t_prev_masked, E_masked)
        return elbo
    
    # return v_t
    def sample(self, x_t, v_t, t, E):
        #i = self.get_bin_t(t)
        #return self.vae_list[i].sample(x_t, t)
        v_t = torch.zeros_like(x_t)
        t_prev = torch.zeros_like(t)
        time_bins_mask = self.dispatch_batch(x_t, t)
        for i, t_mask in enumerate(time_bins_mask):
            if t_mask.sum() == 0:
                # no x_t, v_t in this time bin
                continue
            #print('i', i)
            #print('t_mask', t_mask.shape, len(np.where(t_mask.cpu())[0]))
            #print('t', t.shape)
            #print('x_t', x_t.shape)
            #print('v_t', v_t.shape)
            #print('x_t_mask', x_t[t_mask].shape)
            #print('v_t_mask', v_t[t_mask].shape)
            #print('t_mask', t[t_mask].shape)
            x_t_mask = x_t[t_mask]
            v_t_mask = v_t[t_mask]
            t_masked = t[t_mask]
            if E is not None:
                E_masked = E[t_mask]
            else:
                E_masked = None
            t_prev[t_mask], v_t[t_mask] = self.vae_list[i].sample(x_t_mask, v_t_mask, t_masked,E_masked)
        return t_prev, v_t
    