import torch
import torch.nn as nn
import torch.utils.data as data
import zuko

from torch import Tensor
from torch.distributions import Distribution, Normal, Bernoulli, Independent
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm


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

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        q = self.encoder(x)
        z = q.rsample()

        return self.decoder(z).log_prob(x) + self.prior(c).log_prob(z) - q.log_prob(z)
    


class GaussianModel(zuko.flows.LazyDistribution):
    def __init__(self, features: int, context: int):
        super().__init__()

        self.hyper = nn.Sequential(
            nn.Linear(context, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, 2 * features),
        )

    def forward(self, c: Tensor) -> Distribution:
        phi = self.hyper(c)
        mu, log_sigma = phi.chunk(2, dim=-1)

        return Independent(Normal(mu, log_sigma.exp()), 1)


class BernoulliModel(zuko.flows.LazyDistribution):
    def __init__(self, features: int, context: int):
        super().__init__()

        self.hyper = nn.Sequential(
            nn.Linear(context, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, hidden_dim_codec),
            nn.ReLU(),
            nn.Linear(hidden_dim_codec, features),
        )

    def forward(self, c: Tensor) -> Distribution:
        phi = self.hyper(c)
        rho = torch.sigmoid(phi)

        return Independent(Bernoulli(rho), 1)
    
class VAE(nn.Module):

    def __init__(self, nfeatures):
        super(VAE, self).__init__()
        
        self.nfeatures=nfeatures
        assert self.nfeatures == 1024
        self.act = nn.SiLU(inplace=False)
        
        self.encoder = GaussianModel(16, 1024)
        self.decoder = BernoulliModel(1024, 16)

        self.prior = zuko.flows.MAF(
            features=16,
            context=24,
            transforms=3,
            hidden_features=(256, 256),
        )

        self.elbo = ELBO(self.encoder, self.decoder, self.prior)
        
        self.time_mlp = nn.Sequential(nn.Linear(1, 8),
                                    self.act,
                                    nn.Linear(8, 8), 
                                    self.act,
                                    nn.Linear(8, 8), 
                                    self.act)

        self.x_mlp = nn.Sequential(nn.Linear(1024, hidden_dim_codec),
                                    self.act,
                                    nn.Linear(hidden_dim_codec, hidden_dim_codec), 
                                    self.act,
                                    nn.Linear(hidden_dim_codec, 16), 
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
        x_t, t = self._forward(x_t, t)
        z = self.prior(torch.cat([x_t, t], dim = -1)).sample((1,))
        x = self.decoder(z).mean.reshape(-1, 1, 32, 32)
        return x