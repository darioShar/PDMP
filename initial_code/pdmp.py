# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



class PDMP:
    
    def __init__(self, time_horizon = 10, reverse_steps = 200, device = None, sampler_name = 'ZigZag', refresh_rate = 1.):
        self.dim = 2
        self.Sigma = torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        self.Q = self.Sigma_inv
        self.T = time_horizon
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
        self.sampler = sampler_name
        self.refreshment_rate = refresh_rate
        # print(self.Q)
    
    # generate data
    ## For the backward process
    def flip_given_rate(self, v, lambdas, s):
        lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        v[event_time.sample() <= s] *= -1.

    def splitting_zzs_DBD(self, model, T, N, nsamples = None, x_init=None, v_init=None, print_progession = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = torch.linspace(1, 0, N+1)**2 * T
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (nsamples is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(nsamples, 1, 2)
        if v_init is None:
            v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (x_init.shape[0],))].reshape(x_init.shape[0], 1, 1).repeat(1, *(x_init.shape[1:]))
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []
        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x = x - v * delta / 2 # x - v * δ / 2
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                density_ratio = model(torch.concat((x,v), dim = -1).to(self.device),
                                    (torch.ones(x.shape[0])*time_mid).to(self.device))[:, :, :2]
                #print(density_ratio.mean(dim=0))
                switch_rate = density_ratio.cpu()* torch.maximum(torch.zeros(x.shape), -v * x)
                self.flip_given_rate(v, switch_rate, delta)
                x = x - v * delta / 2 #x - v * δ / 2
                #print(x, v)
                #chain.append(Skeleton(x.copy(), v.copy(), n * δ))
        chain.append(torch.concat((x, v), dim = -1))
        return chain
    
    def refresh_given_rate(self, x, v, t, lambdas, s, model):
        # lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        temp = event_time.sample()
        #temp = temp.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        tmp = model(torch.cat(
            (x,
             t * torch.ones(x.shape[0], 1, 1).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
        # print(temp[temp <= s].shape)
        # print((tmp[temp <= s]))
        v[temp <= s] = tmp[temp <= s]

    def splitting_HMC_DRD(self, model, T, N, nsamples = None, x_init=None, v_init=None, print_progession = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = torch.linspace(1, 0, N+1)**2 * T
        timesteps = timesteps.to(self.device)
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (nsamples is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(nsamples, 1, 2)
        if v_init is None:
            v_init = torch.randn(nsamples, 1, 2)
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []

        x_init = x_init.to(self.device)
        v_init = v_init.to(self.device)

        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x_init = x.clone()
                v_init = v.clone()
                x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                # model outputs one value per data point.
                # need to make sure who's model
                log_p_t_model = model(torch.cat((x,
                                                 time_mid * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                switch_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                # print(switch_rate)
                self.refresh_given_rate(x, v, time_mid, switch_rate, delta, model)
                x_init = x.clone()
                v_init = v.clone()
                x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))
                #print(x, v)
                #chain.append(Skeleton(x.copy(), v.copy(), n * δ))
        chain.append(torch.concat((x, v), dim = -1))
        return chain

    
    def forward(self, data, t, speed = None):
        #new_data = data.clone()
        #time_horizons = t.clone().detach()
        if self.sampler == 'ZigZag':
            if speed is None:
                speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*data.shape[0],))]
                speed = speed.reshape(data.shape[0], 1, 2)
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)
        elif self.sampler == 'HMC':
            if speed is None:
                speed = torch.randn_like(data).to(self.device)
            while (t > 0.).any():
                self.HMC_gauss_1event(data, speed, t)
                #speed = torch.randn_like(data)
                speed[t > 0] = torch.randn_like(speed[t > 0])
        elif self.sampler == 'BPS':
            if speed is None:
                speed = torch.randn(data.shape)
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)

    
    def reverse_sampling(self,
                        nsamples,
                        model,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        print_progession = False
                        ):
        if self.sampler == 'ZigZag':
            chain = torch.stack(self.splitting_zzs_DBD(model, 
                                                   self.T, 
                                                   self.reverse_steps, 
                                                   nsamples = nsamples,
                                                   print_progession=print_progession))
        elif self.sampler == 'HMC':
            chain = torch.stack(self.splitting_HMC_DRD(model, 
                                                   self.T, 
                                                   self.reverse_steps, 
                                                   nsamples = nsamples,
                                                   print_progession=print_progession))
        return chain
    
    
    def training_losses_zigzag(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        output = model(X_V_t, t)

        # invert time on component 1 and 2
        X_V_inv_t_0 = X_V_t.detach().clone() # clone to avoid modifying the original tensor, detach to avoid computing gradients on original tensor
        X_V_inv_t_1 = X_V_t.detach().clone()
        X_V_inv_t_0[:, :, 2] *= -1 # reverse speed on i = 1
        X_V_inv_t_1[:, :, 3] *= -1 # reverse speed on i = 2

        # run the model on each inverted speed component
        output_inv_0 = model(X_V_inv_t_0, t)
        output_inv_1 = model(X_V_inv_t_1, t)
        
        # compute the loss
        def g(x):
            return (1 / (1+x))
        loss = g(output[:, :, 0])**2 + g(output_inv_0[:, :, 0])**2
        loss += g(output[:, :, 1])**2 + g(output_inv_1[:, :, 1])**2
        loss -= 2*(g(output[:, :, 0]) + g(output[:, :, 1]))
        return loss
    
    def training_loss_hmc(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        #X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        t = t.unsqueeze(-1).unsqueeze(-1)
        output = -model(torch.cat([X_t, t], dim = -1)).log_prob(V_t) #(X_V_t, t)
        #output = output.mean()
        return output

    def training_losses(self, model, X_t, V_t, t):
        if self.sampler == 'ZigZag':
            return self.training_losses_zigzag(model, X_t, V_t, t)
        elif self.sampler == 'HMC':
            return self.training_loss_hmc(model, X_t, V_t, t)
        else:
            raise ValueError('Unknown sampler: {}'.format(self.sampler))

    def switchingtime_gauss(self, a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
        return -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape),a))**2/b**2 - 2 * torch.log(1-u)/b)
    

    def ZigZag_gauss_1event(self, x, v, time_horizons, excess_rate=0.0):

        a = v * x
        b = v * v

        Δt_switches = self.switchingtime_gauss(a, b, torch.rand_like(a))
        if excess_rate == 0.0:
            # Δt_excess = torch.inf
            x += v * torch.minimum(time_horizons,Δt_switches)
            v[time_horizons>Δt_switches] *= -1
            time_horizons -= torch.minimum(time_horizons,Δt_switches)
        else:
            Δt_excess = -torch.log(torch.rand_like(a)) / (excess_rate)
            x += v * torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))
            v[time_horizons > torch.minimum(Δt_switches,Δt_excess)] *= -1
            time_horizons -= torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))

    def generate_refreshment_time_per_data_point(tmp):
        return -torch.log(torch.rand((tmp.shape[0]))).reshape(-1, *([1]*len(tmp.shape[1:]))).repeat(1, *tmp.shape[1:])
    
    def HMC_gauss_1event(self, x, v, time_horizons):

        # Δt_refresh = -torch.log(torch.rand(x.shape[0])) / (refreshment_rate)
        Δt_refresh = -torch.log(torch.rand((x.shape[0]))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (self.refreshment_rate)
        #Δt_refresh = Δt_refresh.to(self.device)
        x_old = x.clone()
        x *= torch.cos(torch.minimum(time_horizons,Δt_refresh))
        x += v * torch.sin(torch.minimum(time_horizons,Δt_refresh))
        v *= torch.cos(torch.minimum(time_horizons,Δt_refresh)) 
        v -= x_old * torch.sin(torch.minimum(time_horizons,Δt_refresh))
        time_horizons -= torch.minimum(time_horizons,Δt_refresh)







'''
# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



class PDMP:
    
    def __init__(self, time_horizon = 10, reverse_steps = 200, device = None, sampler_name = 'ZigZag'):
        self.dim = 2
        self.Sigma = torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        self.Q = self.Sigma_inv
        self.T = time_horizon
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
        self.sampler = sampler_name
        # print(self.Q)
    
    # generate data
    ## For the backward process
    def flip_given_rate(self, v, lambdas, s):
        lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        v[event_time.sample() <= s] *= -1.

    def splitting_zzs_DBD(self, model, T, N, nsamples = None, x_init=None, v_init=None, print_progession = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = torch.linspace(1, 0, N+1)**2 * T
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (nsamples is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(nsamples, 1, 2)
        if v_init is None:
            v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (x_init.shape[0],))].reshape(x_init.shape[0], 1, 1).repeat(1, *(x_init.shape[1:]))
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []
        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x = x - v * delta / 2 # x - v * δ / 2
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                density_ratio = model(torch.concat((x,v), dim = -1).to(self.device),
                                    (torch.ones(x.shape[0])*time_mid).to(self.device))[:, :, :2]
                #print(density_ratio.mean(dim=0))
                switch_rate = density_ratio.cpu()* torch.maximum(torch.zeros(x.shape), -v * x)
                self.flip_given_rate(v, switch_rate, delta)
                x = x - v * delta / 2 #x - v * δ / 2
                #print(x, v)
                #chain.append(Skeleton(x.copy(), v.copy(), n * δ))
        chain.append(torch.concat((x, v), dim = -1))
        return chain

    
    def forward(self, data, t, speed = None):
        #new_data = data.clone()
        #time_horizons = t.clone().detach()
        if self.sampler == 'ZigZag':
            if speed is None:
                speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*data.shape[0],))]
                speed = speed.reshape(data.shape[0], 1, 2)
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)
        elif self.sampler == 'HMC':
            if speed is None:
                speed = torch.randn_like(data)
            while (t > 0.).any():
                self.HMC_gauss_1event(data, speed, t)
                speed = torch.randn_like(data)
        elif self.sampler == 'BPS':
            if speed is None:
                speed = torch.randn(data.shape)
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)

    
    def reverse_sampling(self,
                        nsamples,
                        model,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        print_progession = False
                        ):
        chain = torch.stack(self.splitting_zzs_DBD(model, 
                                                   self.T, 
                                                   self.reverse_steps, 
                                                   nsamples = nsamples,
                                                   print_progession=print_progession))
        return chain
    
    def training_loss_zigzag(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        output = model(X_V_t, t)

        # invert time on component 1 and 2
        X_V_inv_t_0 = X_V_t.detach().clone() # clone to avoid modifying the original tensor, detach to avoid computing gradients on original tensor
        X_V_inv_t_1 = X_V_t.detach().clone()
        X_V_inv_t_0[:, :, 2] *= -1 # reverse speed on i = 1
        X_V_inv_t_1[:, :, 3] *= -1 # reverse speed on i = 2

        # run the model on each inverted speed component
        output_inv_0 = model(X_V_inv_t_0, t)
        output_inv_1 = model(X_V_inv_t_1, t)
        
        # compute the loss
        def g(x):
            return (1 / (1+x))
        loss = g(output[:, :, 0])**2 + g(output_inv_0[:, :, 0])**2
        loss += g(output[:, :, 1])**2 + g(output_inv_1[:, :, 1])**2
        loss -= 2*(g(output[:, :, 0]) + g(output[:, :, 1]))
        return loss
    
    def training_loss_hmc(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        output = model(X_V_t, t)

        # invert time on component 1 and 2
        X_V_inv_t_0 = X_V_t.detach().clone() # clone to avoid modifying the original tensor, detach to avoid computing gradients on original tensor
        X_V_inv_t_1 = X_V_t.detach().clone()
        X_V_inv_t_0[:, :, 2] *= -1 # reverse speed on i = 1
        X_V_inv_t_1[:, :, 3] *= -1 # reverse speed on i = 2

        # run the model on each inverted speed component
        output_inv_0 = model(X_V_inv_t_0, t)
        output_inv_1 = model(X_V_inv_t_1, t)
        
        # compute the loss
        def g(x):
            return (1 / (1+x))
        loss = g(output[:, :, 0])**2 + g(output_inv_0[:, :, 0])**2
        loss += g(output[:, :, 1])**2 + g(output_inv_1[:, :, 1])**2
        loss -= 2*(g(output[:, :, 0]) + g(output[:, :, 1]))
        return loss

    def training_losses(self, model, X_t, V_t, t):
        if self.sampler == 'ZigZag':
            return self.training_loss_zigzag(model, X_t, V_t, t)
        elif self.sampler == 'HMC':
            return self.training_loss_hmc(model, X_t, V_t, t)
        else:
            raise ValueError('Unknown sampler: {}'.format(self.sampler))
    
    def switchingtime_gauss(self, a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
        return -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape),a))**2/b**2 - 2 * torch.log(1-u)/b)
    

    def ZigZag_gauss_1event(self, x, v, time_horizons, excess_rate=0.0):

        a = v * x
        b = v * v

        Δt_switches = self.switchingtime_gauss(a, b, torch.rand_like(a))
        if excess_rate == 0.0:
            # Δt_excess = torch.inf
            x += v * torch.minimum(time_horizons,Δt_switches)
            v[time_horizons>Δt_switches] *= -1
            time_horizons -= torch.minimum(time_horizons,Δt_switches)
        else:
            Δt_excess = -torch.log(torch.rand_like(a)) / (excess_rate)
            x += v * torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))
            v[time_horizons > torch.minimum(Δt_switches,Δt_excess)] *= -1
            time_horizons -= torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))

    def generate_refreshment_time_per_data_point(tmp):
        return -torch.log(torch.rand((tmp.shape[0]))).reshape(-1, *([1]*len(tmp.shape[1:]))).repeat(1, *tmp.shape[1:])
    
    def HMC_gauss_1event(self, x, v, time_horizons, refreshment_rate=1.0):

        # Δt_refresh = -torch.log(torch.rand(x.shape[0])) / (refreshment_rate)
        Δt_refresh = -torch.log(torch.rand((x.shape[0]))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (refreshment_rate)
        x *= torch.cos(torch.minimum(time_horizons,Δt_refresh)) 
        x += v * torch.sin(torch.minimum(time_horizons,Δt_refresh))
        time_horizons -= torch.minimum(time_horizons,Δt_refresh)
'''