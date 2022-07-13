import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')


import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time as time 
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng, dde_RK45


def mutual_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_multi(x, t, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_decouple_eff(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([B + x[0] * (1 - x[0]/K) * (x[0]/C - 1), B + x[1] * (1-x[1]/K) * (x[1]/C - 1)])
    sum_g = np.array([w * x[0]*x[1] / (D + E * x[0] +H * x[1]), beta * x[1]**2 / (D + (E+H) * x[1])])
    dxdt = sum_f + sum_g
    return dxdt

def genereg_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * xd
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt

def genereg_multi(x, t, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * x 
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt




network_type = 'SF'
N = 1000
beta = 0.1
betaeffect = 0
seed = 0
d = 2000
seed = [20, 20]
d = [2.5, 999, 3]

B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1
B_gene = 1
arguments = (B, C, D, E, H, K_mutual)
arguments = (B_gene, )

A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
N_actual = np.size(A, 0)
net_arguments = (index_i, index_j, A_interaction, cum_index)

#xs = odeint(mutual_multi, np.ones(N_actual) * 5.0 , np.arange(0, 1000, 0.01), args=(arguments, net_arguments))
#initial_condition = xs[-1] - 1e-3
t = np.arange(0, 200, 0.01)
delay = 1.2
t1 = time.time()
#dyn_RK = dde_RK45(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
t2 = time.time()
#dyn = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
t3 = time.time()
#dyn_eff = dde_RK45(mutual_decouple_eff, np.ones(2) * 5, t, *(delay, 0.03, 0.447, arguments))
#dyn_eff = dde_RK45(mutual_decouple_eff, np.ones(2) * 5, t, *(delay, 0.03, 0.447, arguments))


xs = odeint(genereg_multi, np.ones(N_actual) * 5.0 , np.arange(0, 1000, 0.01), args=(arguments, net_arguments))
initial_condition = xs[-1] - 1e-3
dyn_RK = dde_RK45(genereg_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
plt.plot(np.mean(dyn_RK, 1), alpha=0.8, linewidth=3)
plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('$t$', fontsize= 20)
plt.ylabel('$\\langle x \\rangle$', fontsize = 20)
plt.locator_params(axis='x', nbins=5)
plt.show()

