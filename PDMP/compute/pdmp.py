# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def match_last_d(t, x):
    return t.reshape(-1, *[1]*len(x.shape[1:])).repeat(1, *x.shape[1:])

class PDMP:
    
    def __init__(self, 
                 time_horizon = 10, 
                 reverse_steps = 200,
                 device = None, 
                 sampler = 'ZigZag', 
                 refresh_rate = 1.,
                 time_spacing = None,
                 add_losses = [],
                 use_softmax = False, # for ZigZag output
                 learn_jump_time = False,
                 bin_input_zigzag=False
                 ):
        self.T = time_horizon
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
        self.sampler = sampler
        self.refreshment_rate = refresh_rate
        self.time_spacing = time_spacing
        self.add_losses = add_losses
        self.use_softmax = use_softmax
        self.learn_jump_time = learn_jump_time
        self.bin_input_zigzag = bin_input_zigzag

        for x in self.add_losses:
            possible_losses = ['ml', 'hyvarinen', 'square', 'kl', 'logistic', 'hyvarinen_simple', 'kl_simple']
            assert x in possible_losses, 'specified loss {} unavailable. Possible losses to choose from : {}'.format(x, possible_losses)

    
    def get_timesteps(self, N, exponent = 2, **kwargs):
        return torch.linspace(1, 0, N+1)**exponent * self.T
    
    def get_random_timesteps(self, N, exponent = 2, **kwargs):
        return torch.rand(N)**exponent * self.T

    def rescale_noising(self, noising_steps, time_spacing = None):
        self.reverse_steps = noising_steps
        self.time_spacing = time_spacing


    def reverse_sampling(self,
                        model,
                        model_vae = None,
                        reverse_steps=None,
                        shape = None,
                        time_spacing = None,
                        backward_scheme = None,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        exponent = 2.,
                        print_progression = False,
                        get_sample_history = False,
                        ):
        assert initial_data is None, 'Using specified initial data is not yet implemented.'
        
        print('sampling with exponent {}'.format(exponent))

        reverse_sample_func = {
            'ZigZag': {
                'splitting': self.splitting_zzs_DBD,
                'euler': None
            },
            'HMC': {
                'splitting': self.splitting_HMC_DRD,
                'euler': None,
                'jump_times': self.jump_times_HMC,
            },
            'BPS': {
                'splitting': self.splitting_BPS_RDBDR,
                'euler': self.Euler_BPS
            },
        }
        # for the moment, don;t do anything with time spacing
        assert time_spacing is None, 'Specific time spacing is not yet implemented.'

        if self.learn_jump_time:
            backward_scheme = 'jump_times'
        
        func = reverse_sample_func[self.sampler][backward_scheme]
        assert func is not None, '{} not yet implemented'.format((self.sampler, backward_scheme))
        samples_or_chain = func(model,
                                model_vae,
                                self.T, 
                                reverse_steps,
                                shape = shape,
                                exponent=exponent,
                                print_progession=print_progression,
                                get_sample_history = get_sample_history)
        return samples_or_chain
    

    # generate data
    ## For the backward process
    def flip_given_rate(self, v, lambdas, s):
        lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        v[event_time.sample() <= s] *= -1.

    def refresh_given_rate(self, x, v, t, lambdas, s, model, model_vae):
        # lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        temp = event_time.sample()
        if model_vae is not None:
            tmp = model_vae.sample(x, t) 
        else:
            tmp = model.sample(x, t)
        v[temp <= s] = tmp[temp <= s]

    
    def splitting_zzs_DBD(self, 
                          model,
                          model_vae,
                          T, 
                          N, 
                          shape = None, 
                          x_init=None, 
                          v_init=None,
                          exponent = 2.,
                          print_progession = False, 
                          get_sample_history = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = self.get_timesteps(N, exponent=exponent)

        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (x_init.numel(), ))].reshape(*x_init.shape)
        chain = []
        x = x_init.clone().to(self.device)
        v = v_init.clone().to(self.device)
        model.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x -= v * delta / 2 # x - v * δ / 2
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                #density_ratio = model(torch.concat((x,v), dim = -1).to(self.device),
                #                    (torch.ones(x.shape[0])*time_mid).to(self.device))[..., :x.shape[-1]]
                if self.bin_input_zigzag:
                    output = model(x.to(self.device), (torch.ones(x.shape[0])*time_mid).to(self.device), v = v.to(self.device), bin_input = 0)
                    density_ratio = output
                else:
                    output = model(x.to(self.device), (torch.ones(x.shape[0])*time_mid).to(self.device))
                    # we extract the densities with a separate function
                    density_ratio, selected_output_inv = self.get_densities_from_zigzag_output(output, v.to(self.device))
                switch_rate = density_ratio* (torch.maximum(torch.zeros(x.shape).to(self.device), -v * x) + self.refreshment_rate * torch.ones_like(x).to(self.device))
                self.flip_given_rate(v, switch_rate, delta)
                x -= v * delta / 2 #x - v * δ / 2
        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)


    def jump_times_HMC(self, model, model_vae, T, N, shape = None, x_init = None, v_init = None, exponent = 2., print_progession = False, get_sample_history = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x

        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            v_init = torch.randn(*shape)
        chain = []

        x_init = x_init.to(self.device)
        v_init = v_init.to(self.device)
        
        data_shape = x_init.shape
        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        if model_vae is not None:
            model_vae.eval()
        t = torch.ones(x.shape[0]).to(self.device) * self.T
        counter = 0
        with torch.inference_mode():
            while (t > 0).any():
                counter += 1
                U = torch.rand(x.shape[0]).to(self.device)
                prev_t, v = model.sample(x, v, t, U)
                #prev_t = samples[:, -1]
                #v = samples[:, :-1]
                prev_t = torch.minimum(prev_t, t)
                prev_t = torch.maximum(prev_t, torch.zeros_like(prev_t))
                delta = t - prev_t
                delta = match_last_d(delta, x)
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))
                
                # compute x_n-1 from x_n
                x_init = x.clone()
                v_init = v.clone()

                x =   (x_init * torch.cos(delta) - v_init * torch.sin(delta))
                v =   (x_init * torch.sin(delta) + v_init * torch.cos(delta))
                
                #v = model_vae.sample(x, prev_t)

                t -= prev_t
                
                #x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                #v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))
                #
                #time_mid = time - delta/ 2
                #t = time_mid * torch.ones(x.shape[0]).to(self.device)
                #log_p_t_model = model(x, v, t)
#
                #log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                #switch_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                #self.refresh_given_rate(x, v, t, switch_rate, delta, model, model_vae)
                #x_init = x.clone()
                #v_init = v.clone()
                #x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                #v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))
        print('total reverse jumps:', counter)
        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)


    def splitting_HMC_DRD(self, model, model_vae, T, N, shape = None, x_init=None, v_init=None, exponent=2., print_progession = False, get_sample_history = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = self.get_timesteps(N, exponent=exponent)
        timesteps = timesteps.to(self.device)
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            v_init = torch.randn(*shape)
        chain = []

        x_init = x_init.to(self.device)
        v_init = v_init.to(self.device)
        
        data_shape = x_init.shape
        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        if model_vae is not None:
            model_vae.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x_init = x.clone()
                v_init = v.clone()
                x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                # model outputs one value per data point.
                #t = time_mid * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)
                t = time_mid * torch.ones(x.shape[0]).to(self.device)
                #model = model.to(self.device)
                #print('run model log_prob')
                #log_p_t_model = model(torch.cat([x, t], dim = -1)).log_prob(v) #(X_V_t, t)
                log_p_t_model = model(x, v, t)

                #log_p_t_model = model(torch.cat((x,
                #                                 time_mid * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)), 
                #                                 dim = -1
                #                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                #log_p_t_model = log_p_t_model.squeeze(-1)

                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                switch_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                self.refresh_given_rate(x, v, t, switch_rate, delta, model, model_vae)
                x_init = x.clone()
                v_init = v.clone()
                x =   (x_init * torch.cos(delta / 2) - v_init * torch.sin(delta / 2))
                v =   (x_init * torch.sin(delta / 2) + v_init * torch.cos(delta / 2))

        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)

    def reflect_given_rate(lambdas, v, v_reflect, s):
        lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        v[event_time < s] = v_reflect[event_time < s]
    
    def splitting_BPS_RDBDR(self, model, model_vae, T, N, shape = None, x_init=None, v_init=None, exponent=2., print_progession = False, get_sample_history = False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = self.get_timesteps(N, exponent=exponent)
        timesteps = timesteps.to(self.device)
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            v_init = torch.randn(*shape)
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []

        x_init = x_init.to(self.device)
        v_init = v_init.to(self.device)

        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        if model_vae is not None:
            model_vae.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))

                # half a step R
                t = time * torch.ones(x.shape[0]).to(self.device)#.reshape(-1, *[1]*len(x.shape[1:])).to(self.device)
                log_p_t_model = model(x, v, t)
                #log_p_t_model = model(torch.cat((x,
                #                                 time * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)), 
                #                                 dim = -1
                #                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                #log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                self.refresh_given_rate(x, v, t, refresh_rate, delta/2, model, model_vae)

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
                scal_prod = torch.sum(v * x, dim = list(range(1, len(x.shape))))
                scal_prod_elements = scal_prod.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                # print("scal_prod", scal_prod.shape)
                norm_x = torch.sum(x**2,dim = list(range(1, len(x.shape))))
                norm_x_elements = norm_x.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                v_reflect = v - 2*x*scal_prod_elements/norm_x_elements
                # print(v_reflect.shape)
                t = time_mid * torch.ones(x.shape[0]).to(self.device)#.reshape(-1, *[1]*len(x.shape[1:]))
                log_p_t_refl = model(x, v_reflect, t)#.squeeze(-1) #[:, :, :2]
                # print("switchrate ",switch_rate.shape)
                t = time_mid * torch.ones(x.shape[0]).to(self.device)#reshape(-1, *[1]*len(x.shape[1:])).
                log_p_t = model(x, v, t)
                switch_rate = torch.maximum(-scal_prod,torch.zeros(scal_prod.shape).to(self.device)).to(self.device)
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
                t = time_mid * torch.ones(x.shape[0]).to(self.device)#reshape(-1, *[1]*len(x.shape[1:])).
                log_p_t = model(x, v, t)
                #log_p_t_model = model(torch.cat((x,
                #                                 time_mid * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)), 
                #                                 dim = -1
                #                                 ).to(self.device)).log_prob(v) #[:, :, :2]
                #log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_rate = torch.exp(log_nu_v - log_p_t_model) * self.refreshment_rate
                self.refresh_given_rate(x, v, t, refresh_rate, delta/2, model, model_vae)

        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)

    def BPS_gauss_1step_backward(self, x, v, time_horizon, time, ratio_refl, ratio_refresh, model, model_vae):

        a = - v * x
        a = torch.sum(a, dim = list(range(2, len(x.shape)))).squeeze(-1)
        a *= ratio_refl
        b = v * v
        b = torch.sum(b, dim=list(range(2, len(x.shape)))).squeeze(-1)
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
            nn_sample = model if model_vae is None else model_vae
            tmp = nn_sample(torch.cat(
                (x[mask],
                (time * torch.ones(x[mask].shape[0], *([1]*len(x.shape[1:]))).to(self.device) + Δt_refresh[mask].reshape(-1, *([1]*len(x.shape[1:]))))),
                dim = -1).to(self.device)
                ).sample()
            v[mask] = tmp
        # final move
        time_horizons = time_horizon*torch.ones_like(Δt_reflections) - torch.minimum(time_horizon,torch.minimum(Δt_reflections,Δt_refresh))
        x -= v * time_horizons.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])


    def Euler_BPS(self, model, model_vae, T, N, shape = None, x_init=None, v_init=None, exponent=2., print_progession = False, get_sample_history=False):
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
        print('using Euler')
        timesteps = self.get_timesteps(N, exponent=exponent)
        timesteps = timesteps.to(self.device)
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            # v_init = torch.randn(nsamples, 1, 2)
            x_init = x_init.to(self.device)
            v_init =  model(torch.cat(
            (x_init,
             timesteps[0] * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []

        x_init = x_init.to(self.device)
        v_init = v_init.to(self.device)

        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        model_vae.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))

                # compute rates
                log_p_t_model = model(torch.cat((x,
                                                 time * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v)
                log_p_t_model = log_p_t_model.squeeze(-1)
                log_nu_v = torch.distributions.Normal(0, 1).log_prob(v).sum(dim = list(range(1, len(v.shape))))
                refresh_ratios = torch.exp(log_nu_v - log_p_t_model)
                scal_prod = torch.sum(v * x, dim = list(range(2, len(x.shape)))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                norm_x = torch.sum(x**2,dim = list(range(2, len(x.shape)))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
                v_reflect = v - 2*scal_prod*x/norm_x
                log_p_t_refl = model(torch.cat((x,
                                                 time * torch.ones(x.shape[0]).reshape(-1, *[1]*len(x.shape[1:])).to(self.device)), 
                                                 dim = -1
                                                 ).to(self.device)).log_prob(v_reflect).squeeze(-1)
                reflection_ratios = torch.exp(log_p_t_refl - log_p_t_model)
                self.BPS_gauss_1step_backward(x, v, delta, time, reflection_ratios, refresh_ratios, model, model_vae)


        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)

    
    
    # ZigZag's output is of format (B, 2*C, ...) where B is the batch size and C the number of channels of the data ($C=1$ for 2D data). 
    # The first C channels correspond to velocity=-1, the second to velocity=1. 
    # so we use torch.gather using the velocity tensor to retrieve the densities we need.
    def get_densities_from_zigzag_output(self, output, v):
        index = ((v +1.) / 2.0).type(torch.int64)
        B, C = v.shape[:2]
        assert C == 1, 'only one channel is supported right now'
        assert output.shape == (B, C * 2, *output.shape[2:])
        # is use_softmax is true, we interpret the output as p_t and p_t(..., R_t) rather than the ratios
        # so we apply softmax and then compute the ratios
        if self.use_softmax == True:
            # Reshape tensor to [B, C, 2, H, W]
            data_reshaped = output.view(B, C, 2, *output.shape[2:])

            # Apply softmax over the third dimension (channels)
            softmax_output = torch.nn.functional.softmax(data_reshaped, dim=2)

            # Initialize tensor to hold the ratios
            ratios = torch.empty_like(softmax_output)

            # Compute the ratios for each channel pair
            ratios[:, :, 0, ...] = softmax_output[:, :, 0, ...] / softmax_output[:, :, 1, ...]  # First channel over the second
            ratios[:, :, 1, ...] = softmax_output[:, :, 1, ...] / softmax_output[:, :, 0, ...]  # 
            # Optionally reshape back to the original shape if needed
            output = ratios.view(B, 2*C, *output.shape[2:])

        selected_output = torch.gather(output, 1, index)
        selected_output_inv = torch.gather(output, 1, 1 - index)

        return selected_output, selected_output_inv

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
        #output = model(X_V_t, t)
        if self.bin_input_zigzag:
            selected_output = model(X_t, t, V_t, bin_input = 0)
            selected_output_inv = model(X_t, t, V_t, bin_input = 1)
        else:
            output = model(X_t, t)
            selected_output, selected_output_inv = self.get_densities_from_zigzag_output(output, V_t)
        
        # compute the loss
        def g(x):
            return (1 / (1+x))
            
        loss = 0
        

        '''
        #assert X_t.shape[-1] == 2, 'only dimension 2 is implemened'

        # subusampling ceil(0.01 * d) components
        D = torch.prod(torch.tensor(V_t.shape[1:]))
        #subsamples = 5 #int(np.ceil(0.01 * D))
        components_drawn = set()
        for i in range(subsamples):
            # Generate random coordinates for each data point in the batch
            random_components = tuple([slice(None)]+[torch.randint(0, s, (1,))[0].item() for s in V_t.shape[1:]])
            while (random_components[1:] in components_drawn) and (len(components_drawn) < D):
                random_components = tuple([slice(None)]+[torch.randint(0, s, (1,))[0].item() for s in V_t.shape[1:]])
            if len(components_drawn) == D:
                break
            
            components_drawn.add(random_components[1:])
            # Negate the values at the randomly chosen coordinates
            X_inv = X_t.detach().clone()
            V_inv = V_t.detach().clone()
            V_inv[random_components] *= -1
            X_V_inv = torch.concat((X_inv, V_inv), dim = -1)
            output_inv = model(X_V_inv, t)
            loss += g(output[random_components])**2 + g(output_inv[random_components])**2
            loss -= 2*(g(output[random_components]))
        loss = loss / subsamples
        '''
        #invert time on component 1 and 2
        #X_V_inv_t_0 = X_V_t.detach().clone() # clone to avoid modifying the original tensor, detach to avoid computing gradients on original tensor
        #X_V_inv_t_1 = X_V_t.detach().clone()
        #X_V_inv_t_0[:, :, 2] *= -1 # reverse speed on i = 1
        #X_V_inv_t_1[:, :, 3] *= -1 # reverse speed on i = 2
#
        ### run the model on each inverted speed component
        #output_inv_0 = model(X_V_inv_t_0, t)
        #output_inv_1 = model(X_V_inv_t_1, t)

        
        
        #assert ('hyvarinen' in self.add_losses ) or ('kl' in self.add_losses), 'must use either hyvarinen or kl loss in ZigZag'
        def aux(a):
            if len(a) == 0:
                return False
            return (a[0] in self.add_losses) or aux(a[1:])
        zigzag_losses = ['hyvarinen', 'hyvarinen_simple', 'kl', 'kl_simple']
        assert len(list(set(self.add_losses) & set(zigzag_losses))) != 0, 'Did not specify a loss used in ZigZag. Losses specified: {}. Possible losses for ZigZag: {}'.format(self.add_losses, zigzag_losses)
        
        # adding losses
        if 'hyvarinen' in self.add_losses:
            loss += g(selected_output)**2 + g(selected_output_inv)**2
            loss -= 2*g(selected_output)
        elif 'hyvarinen_simple' in self.add_losses:
            loss += g(selected_output_inv)**2 
        if 'kl' in self.add_losses:
            # KL (17)
            loss += selected_output - torch.log(selected_output_inv)
        elif 'kl_simple' in self.add_losses:
            loss += - torch.log(selected_output_inv)

        #loss = g(output[:, :, 0])**2 + g(output_inv_0[:, :, 0])**2
        #loss += g(output[:, :, 1])**2 + g(output_inv_1[:, :, 1])**2
        #loss -= 2*(g(output[:, :, 0]) + g(output[:, :, 1]))
        return loss
    

    # t is the current time, prev_t is the previous time we must preduct
    def training_loss_hmc_jump_times(self, model, X_t, V_t, t, prev_t, E_t, train_type, model_vae):
        ''' train_type: 'VAE', 'MLE', 'RATIO'
        'MLE': only use model, which gives the log prob.
        'VAE': only train the vae
        'RATIO': if model_vae is given, trains the ratio with true v_t replaced by vae sample
        'RATIO': if model_vae is not given, trains the ratio with true v_t
        '''

        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        prev_t = prev_t.to(self.device)
        E_t = E_t.to(self.device)

        loss = 0

        # vae draws from jump kernel
        # model draws the time
        #loss -= model_vae(X_t, V_t, t) # p_t(V_t | X_t, t)
        
        # now model draws both prev_t and the speed before refreshment
        #loss -= model(X_t, V_t, t, prev_t, E_t) # guess prev_t from X_t, V_t, E_t, t
        loss -= model(X_t, V_t, t, prev_t, E_t) # guess prev_t, V_t from V_t, E_t, t

        return loss

    def training_loss_hmc(self, model, X_t, V_t, t, train_type, model_vae=None):
        ''' train_type: 'VAE', 'MLE', 'RATIO'
        'MLE': only use model, which gives the log prob.
        'VAE': only train the vae
        'RATIO': if model_vae is given, trains the ratio with true v_t replaced by vae sample
        'RATIO': if model_vae is not given, trains the ratio with true v_t
        '''

        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)

        loss = 0
        if 'VAE' in train_type:
            assert model_vae is not None
            # train the vae to learn (X_t | V_t) and return the loss
            loss -= model_vae(X_t, V_t, t)

        if 'RATIO' in train_type:
            assert False, 'We rather model the probability and use ml loss rather than modeling the ratio.'
            # use the output of VAE instead of V_t
            assert True in [x in ['kl', 'logistic'] for x in self.add_losses]
            loss = 0
            if model_vae is not None:
                V_t = model_vae.sample(X_t, t)
            output = model(X_t, V_t, t)
            V = torch.randn_like(V_t)
            output_V = model(X_t, V, t)
            if 'kl' in self.add_losses:
                ## KL divergence based loss: pretty good
                loss += output - torch.log(output_V)
            if 'logistic' in self.add_losses:
                ## logistic regression based loss: seems fine
                loss -= torch.log(1/(1+output))
                loss -= torch.log(output_V/(1+output_V))
        if 'NORMAL_WITH_VAE' in train_type:
            assert model_vae is not None
            #print('V_t true mean', V_t.mean())
            #print('V_t true std', V_t.std())
            with torch.inference_mode():
                # replace half the batch
                idx = V_t.shape[0]//2
                V_t[idx:] = model_vae.sample(X_t[idx:], t[idx:]).clone().detach()
            #V_t = V_t.detach().clone() # so that we're sure it doesn't affect model_vae
            #print('V_t vae mean', V_t.mean())
            #print('V_t vae std', V_t.std())
        # MLE
        if ('NORMAL' in train_type) or ('NORMAL_WITH_VAE' in train_type):
            model = model.to(self.device)
            output = model(X_t, V_t, t)

            # only refreshments in hmc
            possible_losses_hmc = ['ml', 'square', 'kl', 'logistic']
            assert True in [x in possible_losses_hmc for x in self.add_losses], 'need to include at least one loss used in HMC: {}'.format(possible_losses_hmc)
            if 'ml' in self.add_losses:
                # this should be the default loss
                loss -= output #(X_V_t, t)
            ### alternative losses to choose from
            if True in [x in ['square', 'kl', 'logistic'] for x in self.add_losses]:
                #### adding some loss for the refreshment ratio
                log_nu_V_t = torch.distributions.Normal(0, 1).log_prob(V_t).sum(dim = list(range(1, len(V_t.shape))))
                V = torch.randn_like(V_t)
                log_nu_V   = torch.distributions.Normal(0, 1).log_prob(V).sum(dim = list(range(1, len(V.shape))))
                #V_reshape = V.reshape(V.shape[0], -1)
                output_V = model(X_t, V, t) #(X_V_t, t)
                #output_V = model(torch.cat([X_t_t, t], dim = -1)).log_prob(V_reshape) #(X_V_t, t)
            if 'square' in self.add_losses:
                ## square loss: tends not to work well in my experience
                loss += torch.exp(2*(log_nu_V_t -output)) 
                loss -= 2 * torch.exp(log_nu_V-output_V)
            if 'kl' in self.add_losses:
                ## KL divergence based loss: pretty good
                loss += torch.exp(log_nu_V_t -output) 
                loss -= torch.log(torch.exp(log_nu_V-output_V))
            if 'logistic' in self.add_losses:
                ## logistic regression based loss: seems fine
                loss -= torch.log(1/(1+torch.exp(log_nu_V_t -output)) )
                loss -= torch.log(torch.exp(log_nu_V - output_V)/(1+torch.exp(log_nu_V - output_V)) )


        return loss
    
        # run the model
        #t = t.reshape(-1, *[1]*len(X_t.shape[1:]))
        #t = t.unsqueeze(-1).unsqueeze(-1)
        #X_t_t = X_t.reshape(X_t.shape[0], -1)
        #V_t_t = V_t.reshape(V_t.shape[0], -1)
        #t = t.reshape(-1, 1)
        #t = t.repeat(1, 32)
        #model = model.to(self.device)
        #output = model(X_t, V_t, t)
        #output = model(torch.cat([X_t_t, t], dim = -1)).log_prob(V_t_t) #(X_V_t, t)
        #model.eval()
        #with torch.inference_mode():
        #    output_sample = model(torch.cat([X_t_t, t], dim = -1)).sample()
        #    print('output sample:', output_sample[0])
        #model.train()

    def training_loss_bps(self, model, X_t, V_t, t, train_type=None, model_vae=None):
        #assert (model_vae is None), 'VAE is not implemented for BPS'
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        # X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        # output = model(X_V_t, t)
        #t = t.reshape(-1, *[1]*len(X_t.shape[1:]))
        #t = t.unsqueeze(-1).unsqueeze(-1)
        #output = model(torch.cat([X_t, t], dim = -1)).log_prob(V_t)
        loss = 0

        if 'VAE' in train_type:
            assert model_vae is not None
            # train the vae to learn (X_t | V_t) and return the loss
            loss -= model_vae(X_t, V_t, t)

        if 'RATIO' in train_type:
            assert False, 'We rather model the probability and use ml loss rather than modeling the ratio.'
            # use the output of VAE instead of V_t
            assert True in [x in ['kl', 'logistic'] for x in self.add_losses]
            loss = 0
            if model_vae is not None:
                V_t = model_vae.sample(X_t, t)
            output = model(X_t, V_t, t)
            V = torch.randn_like(V_t)
            output_V = model(X_t, V, t)
            if 'kl' in self.add_losses:
                ## KL divergence based loss: pretty good
                loss += output - torch.log(output_V)
            if 'logistic' in self.add_losses:
                ## logistic regression based loss: seems fine
                loss -= torch.log(1/(1+output))
                loss -= torch.log(output_V/(1+output_V))
        if 'NORMAL_WITH_VAE' in train_type:
            assert model_vae is not None
            V_t = model_vae.sample(X_t, t)
        # MLE
        if ('NORMAL' in train_type) or ('NORMAL_WITH_VAE' in train_type):
            model = model.to(self.device)
            output = model(X_t, V_t, t)
            possible_losses_bps = ['ml', 'hyvarinen', 'square', 'kl', 'logistic']
            assert True in [x in possible_losses_bps for x in self.add_losses], 'need to include at least one loss used in HMC: {}'.format(possible_losses_bps)
            if 'ml' in self.add_losses:
                # this should be the default loss
                loss -= output #(X_V_t, t)
            ### alternative losses to choose from
            if 'hyvarinen' in self.add_losses:
                ## adding the Hyvarinen loss for the reflection ratio
                temp = V_t * X_t
                scal_prod = torch.sum(temp,dim=list(range(2, len(X_t.shape)))).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
                norms_x = torch.sum(X_t**2,dim=list(range(2, len(X_t.shape)))).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
                RV_t = V_t - 2*scal_prod * X_t / norms_x
                RV_t = RV_t.detach().clone()
                output_reflected = model(X_t, RV_t, t)
                #output_reflected = model(torch.cat([X_t, t], dim = -1)).log_prob(RV_t)
                def g(x):
                    return 1 / (1+x)
                loss += g(torch.exp(output-output_reflected))**2
            if True in [x in ['square', 'kl', 'logistic'] for x in self.add_losses]:
                #### adding some loss for the refreshment ratio
                log_nu_V_t = torch.distributions.Normal(0, 1).log_prob(V_t).sum(dim = list(range(1, len(V_t.shape))))#.unsqueeze(-1)
                V = torch.randn_like(V_t)
                log_nu_V   = torch.distributions.Normal(0, 1).log_prob(V).sum(dim = list(range(1, len(V.shape))))#.unsqueeze(-1)
                output_V = model(X_t, V, t)
                #output_V = model(torch.cat([X_t, t], dim = -1)).log_prob(V) 
            if 'square' in self.add_losses:
                ## square loss: tends not to work well in my experience
                loss += torch.exp(2*(log_nu_V_t - output)) 
                loss -= 2 * torch.exp(log_nu_V-output_V)
            if 'kl' in self.add_losses:
                ## KL divergence based loss: pretty good
                loss += torch.exp(log_nu_V_t - output) 
                loss -= torch.log(torch.exp(log_nu_V-output_V))
            if 'logistic' in self.add_losses:
                ## logistic regression based loss: seems fine
                loss -= torch.log(1/(1+torch.exp(log_nu_V_t - output)) )
                loss -= torch.log(torch.exp(log_nu_V - output_V)/(1+torch.exp(log_nu_V - output_V)) )


        '''add this for reflection loss'''
        # output = model(torch.cat([X_t, t], dim = -1)).log_prob(V_t) 
        # temp = V_t * X_t
        # scal_prod = torch.sum(temp,dim=2).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
        # norms_x = torch.sum(X_t**2,dim=2).reshape(-1, *([1]*len(X_t.shape[1:]))).repeat(1, *X_t.shape[1:])
        # RV_t = V_t - 2*scal_prod * X_t / norms_x
        # RV_t = RV_t.detach().clone()
        # output_reflected = model(torch.cat([X_t, t], dim = -1)).log_prob(RV_t) 
        
        # # compute the loss
        # def g(x):
        #     return (1 / (1+x))
        # add this for reflection loss
        # loss = g(torch.exp(output-output_reflected))**2
        ##### loss = g(torch.exp(output_reflected-output))**2 + g(output_reflected)**2
        ##### loss -= 2*g(output) 
        # t = t.unsqueeze(-1).unsqueeze(-1)
        
        '''add this for refreshment loss'''
        # can also add the loss for the refreshments, see equation (29).

        return loss
    
    def training_losses(self, model, X_batch, time_horizons = None, V_batch = None, train_type=['NORMAL'], model_vae=None):
        if self.learn_jump_time:
            return self.training_losses_jump_time(model, X_batch, time_horizons, V_batch, train_type, model_vae)
        else:
            return self.training_losses_chain(model, X_batch, time_horizons, V_batch, train_type, model_vae)


    def training_losses_jump_time(self, model, X_batch, time_horizons = None, V_batch = None, train_type=['NORMAL'], model_vae=None):

        assert self.sampler == 'HMC', 'training loss jump time only defined for HMC'
        assert model_vae is not None, 'Must use VAE for jump times with HMC'

        # generate random time horizons
        if time_horizons is None:
            time_horizons = self.get_random_timesteps(N=X_batch.shape[0], exponent=2)
        
        t = time_horizons.clone().detach().reshape(-1, *[1]*len(X_batch.shape[1:])).repeat(1, *X_batch.shape[1:])
        x = X_batch.clone()        
        # actually faster to switch to cpu for forward process in the case of pdmps
        device = self.device
        self.device = 'cpu'
        # put everyhting on the device
        X_batch = X_batch.to(self.device)
        t = t.to(self.device)
        # apply the forward process. Everything runs on the cpu.
        X_batch, V_batch, t, prev_t, U = self.forward_jump(X_batch, t, speed = V_batch)
        self.device = device

        # check that the time horizon has been reached for all data
        assert not (t > 0.).any()
        assert not (prev_t < 0.).any()

        while len(t.shape) > 1:
            t = t[:, ..., 0]
            prev_t = prev_t[:, ..., 0]
        time_reached = time_horizons - t
        time_prev = time_horizons - prev_t

        losses = self.training_loss_hmc_jump_times(model, X_batch, V_batch, time_reached, time_prev, U, train_type=train_type, model_vae=model_vae)

        return losses.mean()


    def training_losses_chain(self, model, X_batch, time_horizons = None, V_batch = None, train_type=['NORMAL'], model_vae=None):
        
        # generate random time horizons
        if time_horizons is None:
            time_horizons = self.get_random_timesteps(N=X_batch.shape[0], exponent=2)
            #time_horizons = self.T * (torch.rand(X_batch.shape[0])**2)

        # must be of the same shape as Xbatch for the pdmp forward process, since it will be applied component wise
        t = time_horizons.clone().detach().reshape(-1, *[1]*len(X_batch.shape[1:])).repeat(1, *X_batch.shape[1:])
        # clone our initial data, since it will be modified by forward process
        x = X_batch.clone()
        #v = Vbatch.clone()
        
        # actually faster to switch to cpu for forward process in the case of pdmps
        device = self.device
        self.device = 'cpu'
        # put everyhting on the device
        X_batch = X_batch.to(self.device)
        t = t.to(self.device)
        # apply the forward process. Everything runs on the cpu.
        X_batch, V_batch = self.forward(X_batch, t, speed = V_batch)
        # put back on the device
        self.device = device

        # check that the time horizon has been reached for all data
        assert not (t != 0.).any()

        if self.sampler == 'ZigZag':
            losses = self.training_losses_zigzag(model, X_batch, V_batch, time_horizons)
        elif self.sampler == 'HMC':
            for type in train_type:
                assert type in ['VAE', 'RATIO', 'NORMAL', 'NORMAL_WITH_VAE'], 'Unkown train_type {}'.format(type)
            losses = self.training_loss_hmc(model, X_batch, V_batch, time_horizons, train_type=train_type, model_vae=model_vae)
        elif self.sampler =='BPS':
            for type in train_type:
                assert type in ['VAE', 'RATIO', 'NORMAL', 'NORMAL_WITH_VAE'], 'Unkown train_type {}'.format(type)
            losses = self.training_loss_bps(model, X_batch, V_batch, time_horizons, train_type=train_type, model_vae=model_vae)
        
        return losses.mean() #/ torch.prod(torch.tensor(X_batch.shape[1:]))
    
    def draw_velocity(self, data):
        # for the moment, everything is happening on the cpu
        if self.sampler == 'ZigZag':
            speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (data.numel(), ))]
            speed = speed.reshape(*data.shape)
        elif self.sampler == 'HMC':
            speed = torch.randn_like(data)
        elif self.sampler == 'BPS':
            speed = torch.randn(data.shape)
        else:
            raise Exception('sampler {} nyi'.format(self.sampler))
        # can put everyhting on the device
        return speed.to(self.device)

    def forward_jump(self, data, t, speed = None):
        assert self.sampler == 'HMC', 'learning jump rates only implemented for HMC'
        if speed is None:
            speed = self.draw_velocity(data)
        prev_t = t.detach().clone()
        U = 0
        while (t > 0).any():
            # Δt_refresh = -torch.log(torch.rand(x.shape[0])) / (refreshment_rate)
            U = torch.rand(data.shape[0]).to(self.device)
            U_tmp = U.reshape(-1, *([1]*len(data.shape[1:]))).repeat(1, *data.shape[1:])
            t_refresh = -torch.log(U_tmp) / self.refreshment_rate
            t_refresh = torch.minimum(t, t_refresh)
            x_old = data.clone()
            data *= torch.cos(t_refresh)
            data += speed * torch.sin(t_refresh)
            speed *= torch.cos(t_refresh)
            speed -= x_old * torch.sin(t_refresh)
            t[t>0] -= t_refresh[t>0]

            # refresh only if we continue the dynamic. Else we want to retrieve last speed
            speed[t > 0] = torch.randn_like(speed[t > 0])

            prev_t[t>0] = t[t > 0]

        # prev_t contains last time (T - t_{k-1}), t contains (T - t_k) <= 0, and data_{t_k}, speed_{t_k}
        return data, speed, t, prev_t, U

    def forward(self, data, t, speed = None):
        #new_data = data.clone()
        #time_horizons = t.clone().detach()
        if speed is None:
            speed = self.draw_velocity(data)

        if self.sampler == 'ZigZag':
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)
        elif self.sampler == 'HMC':
            while (t > 1e-9).any():
                self.HMC_gauss_1event(data, speed, t)
                #speed = torch.randn_like(data)
                speed[t > 0] = torch.randn_like(speed[t > 0])
        elif self.sampler == 'BPS':
            while (t > 1e-9).any():
                self.BPS_gauss_1event(data, speed, t)

        return data, speed
    
    

    def switchingtime_gauss(self, a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
        return -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape),a))**2/b**2 - 2 * torch.log(1-u)/b)

    def ZigZag_gauss_1event(self, x, v, time_horizons):
        a = v * x
        b = v * v

        Δt_switches = self.switchingtime_gauss(a, b, torch.rand_like(a).to(self.device))
        if self.refreshment_rate == 0.0:
            # Δt_excess = torch.inf
            x += v * torch.minimum(time_horizons,Δt_switches)
            v[time_horizons>Δt_switches] *= -1
            time_horizons -= torch.minimum(time_horizons,Δt_switches)
        else:
            Δt_excess = -torch.log(torch.rand_like(a).to(self.device)) / (self.refreshment_rate)
            x += v * torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))
            v[time_horizons > torch.minimum(Δt_switches,Δt_excess)] *= -1
            time_horizons -= torch.minimum(time_horizons,torch.minimum(Δt_switches,Δt_excess))

    def generate_refreshment_time_per_data_point(tmp):
        return -torch.log(torch.rand((tmp.shape[0]))).reshape(-1, *([1]*len(tmp.shape[1:]))).repeat(1, *tmp.shape[1:])
    
    def HMC_gauss_1event(self, x, v, time_horizons):

        # Δt_refresh = -torch.log(torch.rand(x.shape[0])) / (refreshment_rate)
        Δt_refresh = -torch.log(torch.rand((x.shape[0])).to(self.device)).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (self.refreshment_rate)
        #Δt_refresh = Δt_refresh.to(self.device)
        x_old = x.clone()
        x *= torch.cos(torch.minimum(time_horizons,Δt_refresh))
        x += v * torch.sin(torch.minimum(time_horizons,Δt_refresh))
        v *= torch.cos(torch.minimum(time_horizons,Δt_refresh)) 
        v -= x_old * torch.sin(torch.minimum(time_horizons,Δt_refresh))
        time_horizons -= torch.minimum(time_horizons,Δt_refresh)

    def switchingtime_gauss(self, a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
        return -a/b + torch.sqrt((torch.maximum(torch.zeros(a.shape).to(self.device),a))**2/b**2 - 2 * torch.log(1-u)/b)
    
    def BPS_gauss_1event(self, x, v, time_horizons):
        #assert len(x.shape) == 3, 'nyi for more than nd data'
        a = v * x
        a = torch.sum(a, dim = list(range(2, len(x.shape))))
        a = a.squeeze(-1)
        b = v * v
        b = torch.sum(b, dim = list(range(2, len(x.shape)))).squeeze(-1)
        Δt_reflections = self.switchingtime_gauss(a, b, torch.rand_like(a).to(self.device))
        Δt_reflections = Δt_reflections.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
    
        '''a = v * x
        a = torch.sum(a, dim = 2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        b = v * v
        b = torch.sum(b, dim=2).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        Δt_reflections = self.switchingtime_gauss(a, b, torch.rand_like(a).to(self.device))'''
        
        Δt_refresh = -torch.log(torch.rand((x.shape[0])).to(self.device)).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:]) / (self.refreshment_rate)
        x += v * torch.minimum(time_horizons,torch.minimum(Δt_reflections,Δt_refresh))
        a = v * x
        a = torch.sum(a, dim = list(range(2, len(x.shape)))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        norm_x = torch.sum(x**2,dim = list(range(2, len(x.shape)))).reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        mask = torch.logical_and(time_horizons > torch.minimum(Δt_reflections, Δt_refresh),Δt_reflections < Δt_refresh)
        v[mask] -= (2 * x[mask]*a[mask]/norm_x[mask])
        mask = torch.logical_and(time_horizons > torch.minimum(Δt_reflections,Δt_refresh), Δt_refresh<Δt_reflections)
        v[mask] = torch.randn_like(v[mask]).to(self.device)
        time_horizons -= torch.minimum(time_horizons,torch.minimum(Δt_reflections,Δt_refresh))







'''
    def get_densities_from_zigzag_output(self, output, v):
        D = output.shape[-1]//2
        index = ((v +1.) / 2.0).int()
        indices_for_gathering = torch.arange(D).unsqueeze(0).unsqueeze(0).to(self.device) + D * index
        indices_for_gathering_inv = torch.arange(D).unsqueeze(0).unsqueeze(0).to(self.device) + D * (1 - index)
        #print(indices_for_gathering.shape)
        final_indices = indices_for_gathering.expand(-1, -1, D)
        final_indices_inv = indices_for_gathering_inv.expand(-1, -1, D)
        #print(final_indices.shape)
        selected_output = torch.gather(output, 2, final_indices)
        selected_output_inv = torch.gather(output, 2, final_indices_inv)


'''




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