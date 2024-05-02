# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



class PDMP:
    
    def __init__(self, time_horizon = 10, reverse_steps = 200, device = None, sampler = 'ZigZag', refresh_rate = 1., backward_scheme = 'splitting'):
        self.dim = 2
        self.Sigma = torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        self.Q = self.Sigma_inv
        self.T = time_horizon
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
        self.sampler = sampler
        self.refreshment_rate = refresh_rate
        self.backward_scheme = backward_scheme
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
                x -= v * delta / 2 # x - v * δ / 2
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                density_ratio = model(torch.concat((x,v), dim = -1).to(self.device),
                                    (torch.ones(x.shape[0])*time_mid).to(self.device))[:, :, :2]
                #print(density_ratio.mean(dim=0))
                switch_rate = density_ratio.cpu()* torch.maximum(torch.zeros(x.shape), -v * x)
                self.flip_given_rate(v, switch_rate, delta)
                x -= v * delta / 2 #x - v * δ / 2
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
            # v_init = torch.randn(nsamples, 1, 2)
            x_init = x_init.to(self.device)
            v_init =  model(torch.cat(
            (x_init,
             T * torch.ones(x_init.shape[0], 1, 1).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
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

    def reflect_given_rate(lambdas, v, v_reflect, s):
        lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        v[event_time < s] = v_reflect[event_time < s]
    
    def splitting_BPS_RDBDR(self, model, T, N, nsamples = None, x_init=None, v_init=None, print_progession = False):
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
            # v_init = torch.randn(nsamples, 1, 2)
            x_init = x_init.to(self.device)
            v_init =  model(torch.cat(
            (x_init,
             T * torch.ones(x_init.shape[0], 1, 1).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
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

                # half a step R
                log_p_t_model = model(torch.cat((x,
                                                 time * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                self.refresh_given_rate(x, v, time, refresh_rate, delta/2, model)

                # half a step D
                x -= v * delta/2

                # a full step B
                # time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                # scal_prod = torch.sum(v * x, dim = 2).squeeze(-1)
                # mask = scal_prod < 0
                # scal_prod = scal_prod.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                # norm_x = torch.sum(x[mask]**2,dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                # v_reflect = v[mask] - 2*scal_prod[mask]*x[mask]/norm_x
                # log_p_t_refl = model(torch.cat((x[mask],
                #                                  time_mid * torch.ones(x[mask].shape[0], 1, 1).to(self.device)), 
                #                                  dim = -1
                #                                  ).to(self.device)).log_prob(v_reflect).squeeze(-1) #[:, :, :2]
                # # print("switchrate ",switch_rate.shape)
                # log_p_t = model(torch.cat((x[mask],
                #                                  time_mid * torch.ones(x[mask].shape[0], 1, 1).to(self.device)), 
                #                                  dim = -1
                #                                  ).to(self.device)).log_prob(v[mask]).squeeze(-1)
                # scal_prod = scal_prod[mask]
                # scal_prod = scal_prod[...,0].squeeze(-1)
                # switch_rate = torch.maximum(-scal_prod,torch.zeros(scal_prod.shape).to(self.device)).to(self.device)
                # switch_rate *= torch.exp(log_p_t_refl-log_p_t)
                # event_time = torch.distributions.exponential.Exponential(switch_rate)
                # # temp = 2*delta*torch.ones(v.shape[0]).to(self.device)
                # temp = event_time.sample()
                # # print(temp.shape)
                # v[mask][temp <= delta] = v_reflect[temp <= delta]

                ## full implementation
                time_mid = time - delta/ 2
                scal_prod = torch.sum(v * x, dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                # print("scal_prod", scal_prod.shape)
                norm_x = torch.sum(x**2,dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                v_reflect = v - 2*scal_prod*x/norm_x
                # print(v_reflect.shape)
                log_p_t_refl = model(torch.cat((x,
                                                 time_mid * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v_reflect).squeeze(-1) #[:, :, :2]
                # print("switchrate ",switch_rate.shape)
                log_p_t = model(torch.cat((x,
                                                 time_mid * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v).squeeze(-1)
                scal_prod = scal_prod[...,0].squeeze(-1)
                # print(scal_prod.shape)
                switch_rate = torch.maximum(-scal_prod,torch.zeros(scal_prod.shape).to(self.device)).to(self.device)
                # print(switch_rate.shape)
                switch_rate *= torch.exp(log_p_t_refl-log_p_t)
                switch_rate[switch_rate == 0.] += 1e-9
                # print("switchrate ",switch_rate.shape)
                event_time = torch.distributions.exponential.Exponential(switch_rate)
                temp = event_time.sample()
                # print(temp.shape)
                v[temp <= delta] = v_reflect[temp <= delta]


                # half a step D
                x -= v * delta/2

                # half a step R
                log_p_t_model = model(torch.cat((x,
                                                 time_mid * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                self.refresh_given_rate(x, v, time_mid, refresh_rate, delta/2, model)
        chain.append(torch.concat((x, v), dim = -1))
        return chain
    
    def BPS_gauss_1step_backward(self, x, v, time_horizon, time, ratio_refl, ratio_refresh, model):

        a = - v * x
        a = torch.sum(a, dim = 2).squeeze(-1)
        a *= ratio_refl
        b = v * v
        b = torch.sum(b, dim=2).squeeze(-1)
        b *= ratio_refl
        # Δt_reflections = self.switchingtime_gauss(a, b, torch.rand_like(a).to(self.device))
        Δt_reflections = -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape).to(self.device),a))**2/b**2 - 2 * torch.log(torch.rand_like(a).to(self.device))/b)
        # Δt_reflections = Δt_reflections.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        Δt_refresh = -torch.log(torch.rand((x.shape[0])).to(self.device)) / (self.refreshment_rate * ratio_refresh)
        # Δt_refresh = -torch.log(torch.rand((x.shape[0]))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (self.refreshment_rate * ratio_refresh)

        x -= v * torch.minimum(time_horizon*torch.ones_like(Δt_reflections),torch.minimum(Δt_reflections,Δt_refresh)).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        # x -= v * torch.minimum(time_horizon*torch.ones_like(Δt_reflections),torch.minimum(Δt_reflections,Δt_refresh))
        a = v * x
        a = torch.sum(a, dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        norm_x = torch.sum(x**2,dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        mask = torch.logical_and(time_horizon > torch.minimum(Δt_reflections, Δt_refresh),Δt_reflections < Δt_refresh)
        v[mask] -= (2 * x[mask]*a[mask]/norm_x[mask])
        mask = torch.logical_and(time_horizon > torch.minimum(Δt_reflections,Δt_refresh), Δt_refresh<Δt_reflections)
        ## fix dimensions of everything
        if torch.any(mask):
            tmp = model(torch.cat(
                (x[mask],
                (time * torch.ones(x[mask].shape[0], *([1]*len(x.shape[1:]))).to(self.device) + Δt_refresh[mask].reshape(-1, *([1]*len(x.shape[1:]))))),
                dim = -1).to(self.device)
                ).sample()
            v[mask] = tmp
        # final move
        time_horizons = time_horizon*torch.ones_like(Δt_reflections) - torch.minimum(time_horizon,torch.minimum(Δt_reflections,Δt_refresh))
        x -= v * time_horizons.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])


    def Euler_BPS(self, model, T, N, nsamples = None, x_init=None, v_init=None, print_progession = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
        print('using Euler')
        timesteps = torch.linspace(1, 0, N+1)**2 * T
        timesteps = timesteps.to(self.device)
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (nsamples is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(nsamples, 1, 2)
        if v_init is None:
            # v_init = torch.randn(nsamples, 1, 2)
            x_init = x_init.to(self.device)
            v_init =  model(torch.cat(
            (x_init,
             timesteps[0] * torch.ones(x_init.shape[0], 1, 1).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
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

                # compute rates
                log_p_t_model = model(torch.cat((x,
                                                 time * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v)
                log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_ratios = torch.exp(log_nu_v - log_p_t_model)
                scal_prod = torch.sum(v * x, dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                norm_x = torch.sum(x**2,dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                v_reflect = v - 2*scal_prod*x/norm_x
                log_p_t_refl = model(torch.cat((x,
                                                 time * torch.ones(x.shape[0], 1, 1).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v_reflect).squeeze(-1)
                reflection_ratios = torch.exp(log_p_t_refl - log_p_t_model)
                self.BPS_gauss_1step_backward(x, v, delta, time, reflection_ratios, refresh_ratios, model)

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
            while (t > 1e-9).any():
                self.HMC_gauss_1event(data, speed, t)
                #speed = torch.randn_like(data)
                speed[t > 0] = torch.randn_like(speed[t > 0])
        elif self.sampler == 'BPS':
            if speed is None:
                speed = torch.randn(data.shape)
            while (t > 1e-9).any():
                self.BPS_gauss_1event(data, speed, t)

    
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
        elif self.sampler == 'BPS':
            if self.backward_scheme == 'splitting':
                chain = torch.stack(self.splitting_BPS_RDBDR(model, 
                                                    self.T, 
                                                    self.reverse_steps, 
                                                    nsamples = nsamples,
                                                    print_progession=print_progession))
            elif self.backward_scheme == 'euler':
                chain = torch.stack(self.Euler_BPS(model, 
                                                    self.T, 
                                                    self.reverse_steps, 
                                                    nsamples = nsamples,
                                                    print_progession=print_progession))
            # chain = torch.stack(self.Euler_BPS(model, 
            #                                 self.T, 
            #                                 self.reverse_steps, 
            #                                 nsamples = nsamples,
            #                                 print_progession=print_progession))
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
        t = t.unsqueeze(-1).unsqueeze(-1)

        loss = -model(torch.cat([X_t, t], dim = -1)).log_prob(V_t) #(X_V_t, t)

        #### adding some loss for the refreshment ratio
        log_nu_V_t = torch.distributions.Normal(0, 1).log_prob(V_t).sum(dim = list(range(1, len(V_t.shape)))).unsqueeze(-1)
        V = torch.randn_like(V_t)
        log_nu_V   = torch.distributions.Normal(0, 1).log_prob(V).sum(dim = list(range(1, len(V.shape)))).unsqueeze(-1)
        output_V = model(torch.cat([X_t, t], dim = -1)).log_prob(V) 
        
        ### three alternative losses 

        ## square loss: tends not to work well in my experience
        # loss += torch.exp(2*(log_nu_V_t + loss)) 
        # loss -= 2 * torch.exp(log_nu_V-output_V)
        
        ## KL divergence based loss: pretty good
        loss += torch.exp(log_nu_V_t + loss) 
        loss -= torch.log(torch.exp(log_nu_V-output_V))
        
        ## logistic regression based loss: seems fine
        # loss -= torch.log(1/(1+torch.exp(log_nu_V_t + loss)) )
        # loss -= torch.log(torch.exp(log_nu_V - output_V)/(1+torch.exp(log_nu_V - output_V)) )

        return loss

    def training_loss_bps(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)

        t = t.unsqueeze(-1).unsqueeze(-1)

        # Maximum likelihood loss to learn the conditional distribution
        loss = - model(torch.cat([X_t, t], dim = -1)).log_prob(V_t) #(X_V_t, t)

        ## adding the Hyvarinen loss for the reflection ratio
        output = -loss
        temp = V_t * X_t
        scal_prod = torch.sum(temp,dim=2).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
        norms_x = torch.sum(X_t**2,dim=2).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
        RV_t = V_t - 2*scal_prod * X_t / norms_x
        RV_t = RV_t.detach().clone()
        output_reflected = model(torch.cat([X_t, t], dim = -1)).log_prob(RV_t) 
        def g(x):
            return (1 / (1+x))
        loss += g(torch.exp(output-output_reflected))**2

        #### adding some loss for the refreshment ratio
        log_nu_V_t = torch.distributions.Normal(0, 1).log_prob(V_t).sum(dim = list(range(1, len(V_t.shape)))).unsqueeze(-1)
        V = torch.randn_like(V_t)
        log_nu_V   = torch.distributions.Normal(0, 1).log_prob(V).sum(dim = list(range(1, len(V.shape)))).unsqueeze(-1)
        output_V = model(torch.cat([X_t, t], dim = -1)).log_prob(V) 
        
        ### three alternative losses 

        ## square loss: tends not to work well in my experience
        # loss += torch.exp(2*(log_nu_V_t - output)) 
        # loss -= 2 * torch.exp(log_nu_V-output_V)
        
        ## KL divergence based loss: pretty good
        loss += torch.exp(log_nu_V_t - output) 
        loss -= torch.log(torch.exp(log_nu_V-output_V))
        
        ## logistic regression based loss: seems fine
        # loss -= torch.log(1/(1+torch.exp(log_nu_V_t - output)) )
        # loss -= torch.log(torch.exp(log_nu_V - output_V)/(1+torch.exp(log_nu_V - output_V)) )

        return loss

    def training_losses(self, model, Xbatch, Vbatch, time_horizons):
                # must be of the same shape as Xbatch for the pdmp forward process, since it will be applied component wise
        t = time_horizons.clone().detach().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
        
        # clone our initial data, since it will be modified by forward process
        x = Xbatch.clone()
        v = Vbatch.clone()
        
        # apply the forward process. Everything runs on the cpu.
        self.forward(Xbatch, t, Vbatch)
        
        # check that the data has been modified
        #idx = (x == Xbatch)
        #print(idx, x[idx], Xbatch[idx])
        #idx = (v == Vbatch)
        #print(idx, v[idx], Vbatch[idx])
        #assert ((x == Xbatch).logical_not()).any() and ((v == Vbatch).logical_not()).any()
        #assert not ((x == Xbatch)).any() and not ((v == Vbatch)).any()
        # check that the time horizon has been reached for all data
        assert not (t != 0.).any()

        if self.sampler == 'ZigZag':
            return self.training_losses_zigzag(model, Xbatch, Vbatch, time_horizons)
        elif self.sampler == 'HMC':
            return self.training_loss_hmc(model, Xbatch, Vbatch, time_horizons)
        elif self.sampler =='BPS':
            return self.training_loss_bps(model, Xbatch, Vbatch, time_horizons)

    def switchingtime_gauss(self, a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
        return -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape),a))**2/b**2 - 2 * torch.log(1-u)/b)

    def ZigZag_gauss_1event(self, x, v, time_horizons):

        a = v * x
        b = v * v

        Δt_switches = self.switchingtime_gauss(a, b, torch.rand_like(a))
        if self.refreshment_rate == 0.0:
            # Δt_excess = torch.inf
            x += v * torch.minimum(time_horizons,Δt_switches)
            v[time_horizons>Δt_switches] *= -1
            time_horizons -= torch.minimum(time_horizons,Δt_switches)
        else:
            Δt_excess = -torch.log(torch.rand_like(a)) / (self.refreshment_rate)
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

    def BPS_gauss_1event(self, x, v, time_horizons):

        a = v * x
        a = torch.sum(a, dim = 2)
        # print(a.shape)
        # a = a.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        a = a.squeeze(-1)
        # print(a.shape)
        b = v * v
        b = torch.sum(b, dim=2).squeeze(-1)
        # b = torch.sum(b, dim=2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        Δt_reflections = self.switchingtime_gauss(a, b, torch.rand_like(a))
        Δt_reflections = Δt_reflections.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        
        Δt_refresh = -torch.log(torch.rand((x.shape[0]))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (self.refreshment_rate)
        x += v * torch.minimum(time_horizons,torch.minimum(Δt_reflections,Δt_refresh))
        a = v * x
        a = torch.sum(a, dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        norm_x = torch.sum(x**2,dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        mask = torch.logical_and(time_horizons > torch.minimum(Δt_reflections, Δt_refresh),Δt_reflections < Δt_refresh)
        v[mask] -= (2 * x[mask]*a[mask]/norm_x[mask])
        mask = torch.logical_and(time_horizons > torch.minimum(Δt_reflections,Δt_refresh), Δt_refresh<Δt_reflections)
        v[mask] = torch.randn_like(v[mask])
        time_horizons -= torch.minimum(time_horizons,torch.minimum(Δt_reflections,Δt_refresh))



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