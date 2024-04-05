# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm



class PDMP:
    
    def __init__(self, time_horizon = 10, reverse_steps = 200, device = None):
        self.dim = 2
        self.Sigma = torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        self.Q = self.Sigma_inv
        self.T = time_horizon
        self.reverse_steps = reverse_steps
        self.device = 'cpu' if device is None else device
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
        if speed is None:
            speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*data.shape[0],))]
            speed = speed.reshape(data.shape[0], 1, 2)
        
        while (t > 0.).any():
            self.ZigZag_gauss_1event(data, speed, t)
        #return data

    
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
    
    
    def training_losses(self, model, X_t, V_t, t):
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





    ''' OLD IMPLEMENTATION '''


    def switchingtime(self, a, b, u):
        # generate switching time for rate of the form max(0, a + b s)
        if b > 0:
            if a < 0:
                return -a/b + self.switchingtime(0.0, b, u)
            else: # a >= 0
                return -a/b + torch.sqrt(a**2/b**2 - 2 * torch.log(1-u)/b)
        elif b == 0: # degenerate case
            if a < 0:
                return float('inf')
            else: # a >= 0
                return -torch.log(1-u)/a
        else: # b <= 0
            if a <= 0:
                return float('inf')
            else: # a > 0
                y = -torch.log(1-u)
                t1 = -a/b
                if y >= a * t1 + b * t1**2/2:
                    return float('inf')
                else:
                    return -a/b - math.sqrt(a**2/b**2 + 2 * y /b)

# Example usage
# a = 2.0
# b = 3.0
# result = switchingtime(a, b)
# print("Result:", result)



# Skeleton = namedtuple("Skeleton", ["position", "velocity", "time"])


    def partial_pot(self, i,x):
        return np.dot(self.Sigma_inv[i,:],x)

    def ZigZag(self, partial_E, Q, T, nsamples = None, x_init=None, v_init=None, excess_rate=0.0):
        
        
        
        #dim = Q.shape[0]
        if x_init is None:
            x_init = torch.randn((nsamples, 1, self.dim))#np.random.normal(0,1,dim)
        if v_init is None:
            v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (x_init.shape[0],))]
            v_init = v_init.reshape(x_init.shape[0], 1, 1).repeat(1, *(x_init.shape[1:]))
            #v_init = np.random.choice([-1, 1], dim)
        
        b = torch.linalg.norm(self.Q, dim = 0)
        b *= torch.sqrt(self.dim)
        # b = np.linalg.norm(Q, axis=0)
        #b *= np.sqrt(dim)

        t = torch.zeros(x_init.shape)
        x = x_init
        v = v_init
        updateSkeleton = False
        finished = False
        skel_chain = torch.concat((x, v, t), dim = -1)# will update skel_chain until the minimum time is above pdmp.T # [Skeleton(np.copy(x), np.copy(v), t)]
        
        rejected_switches = 0
        accepted_switches = 0
        initial_gradient = torch.transpose(torch.dot(self.Sigma_inv, torch.transpose(x))) #np.array([partial_E(i, x) for i in range(self.dim)])
        a = v * initial_gradient

        Δt_proposed_switches = np.array([switchingtime(a[j], b[j],np.random.random()) for j in range(dim)])
        if excess_rate == 0.0:
            Δt_excess = np.inf
        else:
            Δt_excess = -np.log(np.random.rand()) / (dim * excess_rate)

        while not finished:
            i = np.argmin(Δt_proposed_switches)
            Δt_switch_proposed = Δt_proposed_switches[i]
            Δt = min(Δt_switch_proposed, Δt_excess)
            if t + Δt > T:
                Δt = T - t
                finished = True
                updateSkeleton = True

            x += v * Δt
            t += Δt
            a += b * Δt

            if not finished and Δt_switch_proposed < Δt_excess:
                switch_rate = v[i] * partial_E(i, x)
                proposedSwitchIntensity = a[i]
                if proposedSwitchIntensity < switch_rate:
                    print("ERROR: Switching rate exceeds bound.")
                    print(" simulated rate: ", proposedSwitchIntensity)
                    print(" actual switching rate: ", switch_rate)
                    raise ValueError("Switching rate exceeds bound.")
                if np.random.rand() * proposedSwitchIntensity <= switch_rate:
                    v[i] *= -1
                    a[i] = -switch_rate
                    updateSkeleton = True
                    accepted_switches += 1
                else:
                    a[i] = switch_rate
                    updateSkeleton = False
                    rejected_switches += 1
                Δt_excess -= Δt_switch_proposed
                Δt_proposed_switches -= Δt_switch_proposed
                Δt_proposed_switches[i] = switchingtime(a[i], b[i],np.random.random())
            elif not finished: #switch due to excess rate
                updateSkeleton = True
                i = np.random.randint(dim)
                v[i] *= -1
                a[i] = v[i] * partial_E(i, x)
                Δt_proposed_switches -= Δt_excess
                Δt_excess = -np.log(np.random.rand()) / (dim * excess_rate)

            if updateSkeleton:
                skel_chain.append(Skeleton(np.copy(x), np.copy(v), t))
                # print(x)
                # print(v)
                updateSkeleton = False

        return skel_chain


