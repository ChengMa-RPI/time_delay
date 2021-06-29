import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng, dde_RK45

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

cpu_number = 5
B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

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
a = 3
b = 1

fs = 22
ticksize = 16
legendsize = 14
alpha = 0.8
lw = 2




def mutual_group_decouple(x, t, arguments, w, xs_group_transpose):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + np.sum(w * x * xs_group_transpose / (D + E * x + H * xs_group_transpose), 0)
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
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def BDP_group_decouple(x, t, arguments, w, xs_group_transpose):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    B, = arguments
    dxdt = - B * x ** 2 + np.sum(w * xs_group_transpose , 0)
    return dxdt 

def BDP_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x ** 2 
    sum_g = A_interaction * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def PPI_group_decouple(x, t, arguments, w, xs_group_transpose):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    B, F = arguments
    dxdt = F - B * x - np.sum(w * x * xs_group_transpose, 0)
    return dxdt 

def PPI_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, F = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = F - B * x
    sum_g = - A_interaction * x[index_i] * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def CW_group_decouple(x, t, arguments, w, xs_group_transpose):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    a, b = arguments
    dxdt = -x + np.sum(w / (1 + np.exp(a - b * xs_group_transpose)), 0)
    return dxdt 

def CW_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    a, b = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - x
    sum_g = A_interaction / (1 + np.exp(a - b * x[index_j]))
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt




# from multi_dynamics.py

def group_partition_degree(w, group_num, N_actual, space):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    w_unique = np.unique(w)
    if group_num > np.size(w_unique):
        if group_num == N_actual:
            group_index = []
            for i in range(N_actual):
                group_index.append([i])
        else:
            print('method not available')
            return None
    else:
        length_groups = 0
        bins = group_num
        while group_num > length_groups:
            group_index = []
            if space == 'log':
                w_separate = np.logspace(np.log10(w.min()), np.log10(w.max()), bins)
            elif space == 'linear':
                w_separate = np.linspace(w.min(), w.max(), bins)
            elif space == 'bimode':
                w_separate = np.linspace(w.min(), w.max(), bins)
            
            w_separate[-1] = w_separate[-1] * 2
            w_separate[0] = w_separate[0] *0.5
            group_index = []
            for w_i, w_j in zip(w_separate[:-1], w_separate[1:]):
                index = np.where((w < w_j)  & (w >= w_i ))[0]
                if len(index):
                    group_index.append(index)
            length_groups = len(group_index)
            bins += 1
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(w_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index

def group_partition_degree_nn(A, group_num, N_actual, space, r):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    w = np.sum(A, 0)
    w_nn = np.array([np.mean(w[np.where(A[i] == 1)[0]]) for i in range(N_actual)])
    k_i_nn = w * w_nn ** r
    k_i_nn_unique = np.unique(k_i_nn)
    if group_num > np.size(k_i_nn_unique):
        if group_num == N_actual:
            group_index = []
            for i in range(N_actual):
                group_index.append([i])
        else:
            print('method not available')
            return None
    else:
        length_groups = 0
        bins = group_num
        while group_num > length_groups:
            group_index = []
            if space == 'log':
                k_separate = np.logspace(np.log10(k_i_nn.min()), np.log10(k_i_nn.max()), bins)
            elif space == 'linear':
                k_separate = np.linspace(k_i_nn.min(), k_i_nn.max(), bins)
            elif space == 'bimode':
                k_separate = np.linspace(k_i_nn.min(), k_i_nn.max(), bins)
            
            k_separate[-1] = k_separate[-1] * 2
            k_separate[0] = k_separate[0] *0.5
            group_index = []
            for k_i, k_j in zip(k_separate[:-1], k_separate[1:]):
                index = np.where((k_i_nn < k_j)  & (k_i_nn >= k_i ))[0]
                if len(index):
                    group_index.append(index)
            length_groups = len(group_index)
            bins += 1
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(w_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index

def group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    if network_type == 'SF':
        group_index, rearange_index = group_partition_degree(w, group_num, N_actual, 'log')
    elif network_type == 'ER' :
        group_index, rearange_index = group_partition_degree(w, group_num, N_actual, 'linear')
    elif network_type == 'star':
        group_index, rearange_index = group_partition_degree(w, group_num, N_actual, 'bimode')
    elif network_type == 'RGG':
        group_index, rearange_index = group_partition_degree_nn(A, group_num, N_actual, 'linear', r)

    length_groups = len(group_index)
    each_group_length = [len(i) for i in group_index]

    "degree in each subgroup for all nodes"
    A_rearange = A[rearange_index]
    reduce_index = np.hstack((0, np.cumsum(each_group_length)))
    w_group = np.add.reduceat(A_rearange, reduce_index[:-1])


    "construct reduction adjacency matrix"
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        for j in range(length_groups):
            k_i = w[group_index[i]] 
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    initial_condition = np.ones(length_groups) * attractor_value
    t = np.arange(0, 1000, 0.01)
    if dynamics == 'mutual':
        xs_group = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    elif dynamics == 'BDP':
        xs_group = odeint(BDP_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    elif dynamics == 'PPI':
        xs_group = odeint(PPI_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    elif dynamics == 'CW':
        xs_group = odeint(CW_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    return xs_group, group_index, w_group

def group_decouple_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r):
    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    xs_group, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    t = np.arange(0, 1000, 0.01)

    net_arguments = (index_i, index_j, A_interaction, cum_index)

    length_groups = len(xs_group)
    xs_group_transpose = xs_group.reshape(length_groups, 1)
    if dynamics == 'mutual':
        xs_multi = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_group_decouple = odeint(mutual_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    elif dynamics == 'BDP':
        xs_multi = odeint(BDP_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_group_decouple = odeint(BDP_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    elif dynamics == 'PPI':
        xs_multi = odeint(PPI_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_group_decouple = odeint(PPI_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    elif dynamics == 'CW':
        xs_multi = odeint(CW_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_group_decouple = odeint(CW_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]

    "save data"
    rearange_index = np.hstack((group_index))
    group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
    data = np.vstack((group_number, rearange_index, xs_multi, xs_group_decouple))
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_knn/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_r={r}_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_r={r}_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return xs_group_decouple

def parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_decouple_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

    

seed1 = np.arange(10).tolist()
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
seed_ER = seed1

N = 1000
beta = 1
betaeffect = 0
attractor_value = 10

dynamics = 'CW'
arguments = (a, b)

dynamics = 'BDP'
arguments = (B_BDP, )


dynamics = 'PPI'
arguments = (B_PPI, F_PPI)

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)






network_type = 'SF'
d_list = [[2.1, 999, 2], [2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
seed_list = seed_SF
group_num_list = np.arange(1, 21, 1)


network_type = 'ER'
d_list = [2000, 4000, 8000]
seed_list = seed_ER
group_num_list = np.arange(1, 10, 1)


network_type = 'RGG'
d_list = [0.04, 0.05, 0.07]
seed_list = seed_ER
group_num_list = np.arange(1, 10, 1)



r = 0.1
r_list = [0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

for r in r_list:
    for d in d_list:
        for group_num in group_num_list:
            xs_group_decouple = parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r)
            pass


