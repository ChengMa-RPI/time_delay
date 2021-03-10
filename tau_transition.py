import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng

import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
import networkx as nx
import multiprocessing as mp
import time
from ddeint import ddeint
from numpy import linalg as LA
import pandas as pd 
import scipy.io
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import itertools
from scipy import linalg as slin
from scipy.sparse.linalg import eigs as sparse_eig
from scipy.signal import find_peaks

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

cpu_number = 40
B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

fs = 22
ticksize = 16
legendsize = 14
alpha = 0.8
lw = 2

def mutual_single(x, t, beta, arguments):
    """original dynamics N species interaction.

    :x: 1 dynamic variable
    :t: time series
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + beta * x**2 / (D + (E+H) * x)
    return dxdt

def mutual_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = B + x * (1 - xd/K) * ( x/C - 1) + beta * x**2/(D+E*x+H*x)
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

def mutual_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_tau_1D(beta_list, initial_condition, arguments):
    """TODO: Docstring for mutual_tau_1D.
    :returns: TODO

    """
    tau_list = []
    t = np.arange(0, 1000, 0.01) 
    B, C, D, E, H, K = arguments
    for beta in beta_list:
        xs = odeint(mutual_single, initial_condition, t, args=(beta, arguments))[-1]
        P =  -(1 -xs / K) * (2 * xs / C-1)- (2 * beta * xs)/(D + E * xs + H * xs) + (beta * (E+H) * xs**2)/(D + (E+H) * xs)**2 
        Q = xs/K*(xs/C-1)
        if abs(P/Q)<=1:
            tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
            nu = np.arccos(-P/Q)/tau
            tau_list.append(tau[0])
        else:
            tau_list.append(-1)
    return tau_list

def mutual_bifurcation(beta_list, initial_condition, arguments, tau):
    """TODO: Docstring for mutual_tau_1D.
    :returns: TODO

    """
    t = np.arange(0, 500, 0.01) 
    B, C, D, E, H, K = arguments
    xs = np.ones(len(beta_list)) * (-1)
    for beta, i in zip(beta_list, range(len(beta_list))):
        dyn_all = ddeint_Cheng(mutual_single_delay, [initial_condition], t, *(tau, beta, arguments))[-100:]
        #xs = odeint(mutual_single, initial_condition, t, args=(beta, arguments))[-1]
        if np.ptp(dyn_all) < 1e-3:
            xs[i] = dyn_all[-1]
    data = np.vstack((beta_list, xs))
    des = f'../data/mutual/single/tau={tau}/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'x0={initial_condition}.csv'
    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index =None, header=None)
    return xs


def harvest_single(x, t, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    r, K, c = arguments
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    return dxdt

def harvest_multi(x, t, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    r, K, c = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt

def harvest_single_delay(f, x0, t, dt, d, arguments):
    """TODO: Docstone_delayring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    r, K, c = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1)
    return dxdt

def harvest_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    index_i, index_j, A_interaction, cum_index = net_arguments
    r, K, c = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def tau_1D(dynamics, beta_list, initial_condition, arguments):
    """TODO: Docstring for mutual_tau_1D.
    :returns: TODO

    """
    tau_list = []
    beta_plot = []
    t = np.arange(0, 1000, 0.01) 
    for beta in beta_list:
        if dynamics == 'harvest':
            r, K, c = arguments
            arguments = (r, K, beta)
            xs = odeint(harvest_single, initial_condition, t, args=(arguments,))[-1]
            P = -r * (1-xs/K) + 2 * beta * xs / (xs**2+1)**2 
            Q = r * xs / K
        elif dynamics == 'mutual':
            B, C, D, E, H, K = arguments
            xs = odeint(mutual_single, initial_condition, t, args=(beta, arguments))[-1]
            P =  -(1 -xs / K) * (2 * xs / C-1)- (2 * beta * xs)/(D + E * xs + H * xs) + (beta * (E+H) * xs**2)/(D + (E+H) * xs)**2 
            Q = xs/K*(xs/C-1)

            if abs(P/Q)<=1:
                tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
                nu = np.arccos(-P/Q)/tau
                tau_list.append(tau[0])
                beta_plot.append(beta)
    plt.plot(beta_plot, tau_list, linewidth=lw, alpha=alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$\\beta$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)

    return tau_list

def harvest_bifurcation(beta_list, initial_condition, arguments, tau):
    """TODO: Docstring for mutual_tau_1D.
    :returns: TODO

    """
    t = np.arange(0, 500, 0.01) 
    r, K, c = arguments
    xs = np.ones(len(beta_list)) * (-1)
    for beta, i in zip(beta_list, range(len(beta_list))):
        arguments = (r, K, beta)
        dyn_all = ddeint_Cheng(harvest_single_delay, [initial_condition], t, *(tau, arguments))[-100:]
        #xs = odeint(mutual_single, initial_condition, t, args=(beta, arguments))[-1]
        if np.ptp(dyn_all) < 1e-3:
            xs[i] = dyn_all[-1]
    data = np.vstack((beta_list, xs))
    des = f'../data/harvest/single/tau={tau}/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'x0={initial_condition}.csv'
    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index =None, header=None)
    return xs

def parallel_bifurcation(dynamics, beta_list, initial_condition_list, tau_list, arguments):
    """TODO: Docstring for parallel_bifurcation.

    :beta_list: TODO
    :initial_condition_list: TODO
    :tau_list: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    if dynamics == 'mutual':
        p.starmap_async(mutual_bifurcation, [(beta_list, initial_condition, arguments, tau) for tau in tau_list for initial_condition in initial_condition_list]).get()
    elif dynamics == 'harvest':
        p.starmap_async(harvest_bifurcation, [(beta_list, initial_condition, arguments, tau) for tau in tau_list for initial_condition in initial_condition_list]).get()
    p.close()
    p.join()
    return None

def transition_harvest(beta, tau, initial_condition, arguments):
    """TODO: Docstring for transition_harvest.

    :tau: TODO
    :initial_condition: TODO
    :arguments: TODO
    :returns: TODO

    """
    t = np.arange(0, 500, 0.01) 
    r, K, c = arguments
    arguments = (r, K, beta)
    dyn_all = ddeint_Cheng(harvest_single_delay, initial_condition, t, *(tau, arguments))
    plt.plot(t, dyn_all, linewidth=lw, alpha=alpha) 
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)

    #plt.show()

def evolution_multi(network_type, arguments, N, beta, betaeffect, d, seed, delay, initial_value):
    """TODO: Docstring for evolution_compare.

    :network_type: TODO
    :dynamics: TODO
    :arguments: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    initial_condition = np.ones((N_actual)) * initial_value
    t = np.arange(0, 500, 0.01)
    xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))[-1]
    dyn_beta = betaspace(A, dyn_multi)
    if np.max(np.abs(dyn_multi-xs))< 1e-2:
        xs = dyn_multi[-1]
    else:
        xs = -1 * np.ones(N_actual)
    data = np.hstack((seed, xs))
    des = f'../data/mutual/' + network_type + '/xs/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect == 0:
        des_file = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
    else:
        des_file = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, mode='a', index=None, header=None)

    #dyn_multi = ddeint_Cheng(mutual_multi_delay, xs-1e-3, t, *(delay, arguments, net_arguments))
    #dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, xs_decouple - 1e-3, t, *(delay, w, beta, arguments))
    return None

def parallel_evolution(network_type, arguments, N, beta, betaeffect, d, seed_list, delay, initial_value):
    """TODO: Docstring for parallel_evolution.

    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(evolution_multi, [(network_type, arguments, N, beta, betaeffect, d, seed, delay, initial_value) for seed in seed_list]).get()
    p.close()
    p.join()
    return None


r= 1
K= 10
c = 1

B_gene = 1 
B_SIS = 1
B_BDP = 1
B_PPI = 1
F_PPI = 0.1
f = 1
h =2
a = 5
b = 1




beta_list = np.arange(1, 10, 0.5)
tau = 0.2
tau_list = [0.1, 0.15, 0.2, 0.025, 0.3]
initial_condition_list = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

for tau in tau_list:
    for initial_condition in initial_condition_list:
        #xs = mutual_bifurcation(beta_list, [initial_condition], arguments, tau)
        pass

"harvest"
dynamics = 'harvest'
arguments = (r, K, c)
beta_list = np.arange(1, 4, 0.01)
initial_condition = [6.0]
initial_condition_list = np.array([6.0, 7.0])
tau_list = np.array([1.7, 1.8])

beta = 2.6
tau = 2.1
initial_condition = [5.1]
#transition_harvest(beta, tau, initial_condition, arguments)

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
beta_list = np.arange(1, 10, 0.1)
tau_list = [0.1, 0.15, 0.2, 0.025, 0.3]
initial_condition_list = [0.1, 5.0, 6.0, 10.0]
initial_condition = [5.0]

#tau_critical = tau_1D(dynamics, beta_list, initial_condition, arguments)
#parallel_bifurcation(dynamics, beta_list, initial_condition_list, tau_list, arguments)

network_type = 'SF'
N = 100
beta = 1
betaeffect = 1
d = [3, 99, 3]
seed1 = np.arange(100).tolist()
seed_list = np.vstack((seed1, seed1)).transpose().tolist()
delay = 0.2
initial_value = 5.0
beta_list = np.arange(1, 10, 0.5)
for beta in beta_list:
    parallel_evolution(network_type, arguments, N, beta, betaeffect, d, seed_list, delay, initial_value)
