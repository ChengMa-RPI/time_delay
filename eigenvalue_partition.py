import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng
from kcore_KNN_degree_partition import group_partition_degree, find_core_group, kcore_shell, kcore_degree, kcore_KNN_degree

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

cpu_number = 4
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


def linear_expansion_coeff(network_type, N, d, weight, seed, dynamics, method, number_groups, space, degree_interval, xs_multi):
    """TODO: Docstring for eigenvalue_A.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :weight: TODO
    :seed: TODO
    :dynamics: TODO
    :: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)
    w = np.sum(A, 0)
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    if method == 'degree':
        group_index, rearange_index = group_partition_degree(w, number_groups, N_actual, space)
    elif method == 'kcore_degree':
        _, _, _, group_index = kcore_degree(G, A, xs_multi, space, degree_interval)
    elif method == 'kcore':
        _, _, _, group_index = kcore_shell(G, A, xs_multi, 'None')
    elif method == 'kcore_KNN_degree':
        core_group = find_core_group(G)
        _, _, _, group_index = kcore_KNN_degree(G, A, core_group, xs_multi, 'None', space, degree_interval, 'None')

    gamma_kl = np.zeros((len(group_index), len(group_index)))
    beta_kl = np.zeros((len(group_index), len(group_index)))
    alpha_k = np.zeros((len(group_index)))
    x_eff = np.zeros((len(group_index)))
    for k, group_k in enumerate(group_index):
        s_k = w[group_k]
        a_k = s_k / sum(s_k)
        for l, group_l in enumerate(group_index):
            A_lk = A[group_l][:, group_k]
            a_l = w[group_l] / sum(w[group_l])
            s_kl = np.sum(A_lk, 0)
            t1 = A_lk.dot(a_k)
            t2 = a_l * (s_kl * a_k).sum()
            if abs(sum(t2)) <1e-10:
                gamma_kl[k, l] = 1
            else:
                gamma_kl[k, l] = sum(t1 * t2) / sum(t2**2)
            beta_kl[k, l] = t1.sum() 
        t3 = a_k * s_k
        t4 = (a_k * s_k).sum() * a_k
        alpha_k[k] = sum(t3 * t4) / sum(t4 ** 2)
        x_eff[k] = (xs_multi[group_k] * a_k).sum()
    return beta_kl, alpha_k, gamma_kl, x_eff

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
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_linear_expansion(x, t, arguments, net_arguments):
    """TODO: Docstring for mutual_linear_expansion.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    beta_kl, alpha_k, gamma_kl = net_arguments
    xk = (alpha_k * x).reshape(len(x), 1)
    xl = gamma_kl * x
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + np.sum(beta_kl * xk * xl/ (D + E*xk + H*xl), 1)
    return dxdt

def partition_linear_expansion(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups, space, degree_interval, method):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """
    if method == 'degree':
        des = '../data/' + dynamics + '/' + network_type + f'/linear_expansion_degree_group_decouple_' + space + '/' 
        des_file = des + f'N={N}_d=' + str(d) + f'_group_num={number_groups}_seed={seed}.csv'
    elif method == 'kcore':
        des = '../data/' + dynamics + '/' + network_type + f'/linear_expansion_kcore/' 
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    elif method == 'kcore_degree':
        des = '../data/' + dynamics + '/' + network_type + f'/linear_expansion_kcore_degree_' + space + f'={degree_interval}/' 
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    elif method == 'kcore_KNN_degree':
        des = '../data/' + dynamics + '/' + network_type + f'/linear_expansion_kcore_KNN_degree_' + space + f'={degree_interval}/' 
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'

    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    dynamics_linear_expansion = globals()[dynamics + '_linear_expansion']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by partition, calculated by linear expansion"
    beta_kl, alpha_k, gamma_kl, x_eff_deg_part = linear_expansion_coeff(network_type, N, d, weight, seed, dynamics, method, number_groups, space, degree_interval, xs_multi)
    net_arguments_reduction_deg_part = (beta_kl, alpha_k, gamma_kl)
    initial_condition_reduction_deg_part = np.ones(len(beta_kl)) * attractor_value
    xs_reduction_deg_part = odeint(dynamics_linear_expansion, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
    "save data"
    data=  np.hstack((weight, np.ravel(beta_kl), x_eff_deg_part, xs_reduction_deg_part))
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None





network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
weight = 0.1
seed = [0, 0]
dynamics = 'mutual'
number_groups = 5
space = 'log'

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)



t = np.arange(0, 1000, 0.01)
attractor_value = 0.1
number_groups_list = 5
weight_list = np.arange(0.01, 1, 0.01)
method = 'kcore_degree'
method = 'kcore_KNN_degree'
degree_interval = 1.5
for weight in weight_list:
    partition_linear_expansion(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups, space, degree_interval, method)
    pass
