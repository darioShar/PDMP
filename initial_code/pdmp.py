# 
import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import torch

class Skeleton:
    def __init__(self, position, velocity, time):
        self.position = position
        self.velocity = velocity
        self.time = time

# Create an instance of Skeleton
# position = np.array([1.0, 2.0, 3.0])
# velocity = np.array([0.1, 0.2, 0.3])
# time = 10.0
# skeleton_instance = Skeleton(position, velocity, time)

# # Accessing attributes
# print("Position:", skeleton_instance.position)
# print("Velocity:", skeleton_instance.velocity)
# print("Time:", skeleton_instance.time)

def switchingtime(a, b, u):
    # generate switching time for rate of the form max(0, a + b s)
    if b > 0:
        if a < 0:
            return -a/b + switchingtime(0.0, b, u)
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


class PDMP:
    
    def partial_pot(self, i,x):
        return np.dot(self.Sigma_inv[i,:],x)
    
    def __init__(self, total_time = 100):
        self.dim = 2
        self.Sigma = torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)
        self.Q = self.Sigma_inv
        self.T = total_time
        # print(self.Q)
    
    def forward(self, data, t, speed = None):
        #new_data = data.clone()
        #time_horizons = t.clone().detach()
        if speed is None:
            speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*data.shape[0],))]
            speed = speed.reshape(data.shape[0], 1, 2)
        
        while (t > 0.).any():
            self.ZigZag_gauss_1event(data, speed, t)
        #return data
        #return self.ZigZag(self.partial_pot, 
        #                   self.Q, 
        #                   t,
        #                  x_init = data,
        #                  v_init = speed)
                          #v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (data.shape[0],))].repeat(1, *(data.shape[1:])),
                          # excess_rate = 0.)
    
    def reverse_sampling(self,
                        nsamples,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        get_history = False # else store history of data points in a list
                        ):
        pass
    
    
    def training_losses(self, model, data, t):
        pass
    
    
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


