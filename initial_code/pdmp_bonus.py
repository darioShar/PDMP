import math
# import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

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
            return -a/b + math.sqrt(a**2/b**2 - 2 * math.log(1-u)/b)
    elif b == 0: # degenerate case
        if a < 0:
            return float('inf')
        else: # a >= 0
            return -math.log(1-u)/a
    else: # b <= 0
        if a <= 0:
            return float('inf')
        else: # a > 0
            y = -math.log(1-u)
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

def ZigZag(partial_E, Q, T, x_init=None, v_init=None, excess_rate=0.0):

    dim = Q.shape[0]
    if x_init is None:
        x_init = np.random.normal(0,1,dim)
    if v_init is None:
        v_init = np.random.choice([-1, 1], dim)
    
    b = np.linalg.norm(Q, axis=0)
    b *= np.sqrt(dim)
    
    t = 0.0
    x = x_init
    v = v_init
    updateSkeleton = False
    finished = False
    skel_chain = [Skeleton(np.copy(x), np.copy(v), t)]
    
    rejected_switches = 0
    accepted_switches = 0
    initial_gradient = np.array([partial_E(i, x) for i in range(dim)])
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
    
    return skel_chain,accepted_switches,rejected_switches

# Example usage

dim = 2
Sigma = np.identity(dim)
Sigma_inv = np.linalg.inv(Sigma)
Q = Sigma_inv
print(Q)

def partial_pot(i,x):
    return np.dot(Sigma_inv[i,:],x)

T = 100.0
skel, n_acc, n_rej = ZigZag(partial_pot, Q, T)
n_rej
n_acc

# for i in skel:
#     print(i.position,i.velocity,i.time)

all_positions = np.array([skeleton.position for skeleton in skel])
x_values = [position[0] for position in all_positions]
y_values = [position[1] for position in all_positions]

# Plot the positions with lines connecting successive points
# plt.plot(x_values, y_values, marker='o', linestyle='-')

# # Add labels and title
# plt.xlabel('First coordinate')
# plt.ylabel('Second coordinate')
# plt.title('Trajectories')
# plt.grid(True)
# # Show the plot
# plt.show()

## For the backward process
def flip_given_rate(v, lambdas, s):
    event_time = np.random.exponential(lambdas)
    for i in range(len(v)):
        if event_time[i] <= s:
            v[i] *= -1

def splitting_zzs_DBD(grad_U, δ, N, x_init=None, v_init=None):
    if x_init is None:
        x_init = np.random.normal(0,1,dim)
    if v_init is None:
        v_init = np.random.choice([-1, 1], dim)
    chain = [Skeleton(np.copy(x_init), np.copy(v_init), 0)]
    x = x_init.copy()
    v = v_init.copy()
    for n in range(1, N + 1):
        x = x + v * δ / 2
        grad_x = grad_U(x)
        switch_rate = np.maximum(0, v * grad_x)
        flip_given_rate(v, switch_rate, δ)
        x = x + v * δ / 2
        chain.append(Skeleton(x.copy(), v.copy(), n * δ))
    return chain
