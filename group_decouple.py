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

cpu_number = 2
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
a = 5
b = 1

fs = 22
ticksize = 16
legendsize = 14
alpha = 0.8
lw = 3




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

def mutual_nearest_neighbor(x, t, arguments, net_arguments, xs_group_decouple):
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
    sum_g = A_interaction * xs_group_decouple[index_j] / (D + E * x[index_i] + H * xs_group_decouple[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_three_level_decouple(x, t, arguments, w, x_i_nn):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + w * x * x_i_nn / (D + E * x + H * x_i_nn)
    return dxdt

def mutual_three_level_beta(x, t, arguments, beta):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + beta * x * x / (D + E * x + H * x)
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

def group_partition_degree_equal_node(A, group_num, N_actual):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    w = np.sum(A, 0)
    N_each = round(N_actual / group_num)

    w_nn = np.array([np.mean(w[np.where(A[i] == 1)[0]]) for i in range(N_actual)])
    w_nn_norm = w_nn / max(w_nn + 1)
    w_all = w + w_nn_norm
    sort_index = np.argsort(w_all)
    group_index = []
    for i in range(group_num):
        if i < group_num - 1:
            group_index.append(sort_index[i*N_each : (i+1)*N_each])
        else:
            group_index.append(sort_index[i*N_each :])

    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(w_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index

def group_movement(A, group_index, rearange_index, ratio_threshold):
    """TODO: Docstring for group_movement.

    :A: TODO
    :group_index: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    neighbors_degree = [w[np.where(A[i] == 1)[0]] for i in range(N_actual)]
    group_num = len(group_index)
    w_min = [np.min(w[group_index[i]]) for i in range(group_num)]
    w_max = [np.max(w[group_index[i]]) for i in range(group_num)]
    w_mean = [np.mean(w[group_index[i]]) for i in range(group_num)]
    w_interval = np.hstack((w_min, w_max[-1]))

    for i, group_i in enumerate(group_index):
        for j in group_i:
            number_each_group = np.histogram(neighbors_degree[j], bins=w_interval)[0]
            number_max = max(number_each_group) / sum(number_each_group)
            number_max_group = np.argmax(number_each_group)
            #if number_max >= ratio_threshold and number_max_group != i and (abs(w[j] - w_min[number_max_group]) <= 1 or abs(w[j] - w_max[number_max_group]) <= 1) and ((np.mean(neighbors_degree[j]) > w_mean[number_max_group] and i<number_max_group) or (np.mean(neighbors_degree[j]) < w_mean[number_max_group] and i>number_max_group)):
            if number_max >= ratio_threshold and number_max_group != i and (abs(w[j] - w_min[number_max_group]) <= 2 or abs(w[j] - w_max[number_max_group]) <= 2):
                #print(w_min, neighbors_degree[j], number_max_group, i)
                "j should move to the number_max_group" 
                group_index[number_max_group] = np.hstack((group_index[number_max_group], j))
                group_index[i] = np.delete(group_index[i], np.where(group_index[i] == j)[0])

    rearange_index = np.hstack((group_index))
    return group_index, rearange_index

def group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    if ratio_threshold == 'None':
        if space == 'equal_node':
            group_index, rearange_index = group_partition_degree_equal_node(A, group_num, N_actual)
        elif space == 'linear' or space == 'log' or space == 'bimode':
            group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
    else:
        if space == 'equal_node':
            group_index, rearange_index = group_partition_degree_equal_node(A, group_num, N_actual)
        else:
            group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
        group_index, rearange_index = group_movement(A, group_index, rearange_index, ratio_threshold)

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


def group_decouple_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space):
    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    xs_group, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
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
    xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
    data = np.vstack((group_number, rearange_index, xs_multi[rearange_index], xs_group_decouple[rearange_index], xs_groups, w[rearange_index]))
    if ratio_threshold == 'None':
        des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    else:
        des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_move_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if ratio_threshold == 'None':
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    else:
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_r={ratio_threshold}_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_r={ratio_threshold}_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return xs_group_decouple

def parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_decouple_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def group_decouple_nn_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space, iteration_step):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    xs_group, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    t = np.arange(0, 1000, 0.01)

    net_arguments = (index_i, index_j, A_interaction, cum_index)

    length_groups = len(xs_group)
    xs_group_transpose = xs_group.reshape(length_groups, 1)
    dynamics_multi = globals()[dynamics + '_multi']
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    dynamics_nearest_neighbor = globals()[dynamics + '_nearest_neighbor']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_group_decouple = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    xs_nn = xs_group_decouple.copy()
    xs_nn_list = []
    for l in range(iteration_step):
        print(l)
        xs_nn = odeint(dynamics_nearest_neighbor, initial_condition, t, args=(arguments, net_arguments, xs_nn))[-1]
        xs_nn_list.append(xs_nn)
    xs_nn_list = np.vstack((xs_nn_list))

    "save data"
    rearange_index = np.hstack((group_index))
    group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
    xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
    data = np.vstack((group_number, rearange_index, xs_multi[rearange_index], xs_group_decouple, xs_nn_list[:, rearange_index], xs_groups, w[rearange_index]))
    print(data.shape)
    if ratio_threshold == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_' + space + '/'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_move_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if ratio_threshold == 'None':
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    else:
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_r={ratio_threshold}_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_r={ratio_threshold}_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_group_decouple_nn_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space, iteration_step):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_decouple_nn_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space, iteration_step) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def three_level_stable(network_type, N, beta, betaeffect, seed, d, dynamics, arguments, attractor_value):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    N_actual = np.size(A, 0)
    w = np.sum(A, 0)
    degree = np.sum(A>0, 0)
    w_nn = np.array([np.sum(w[np.where(A[i] > 0)[0]]) for i in range(N_actual)]) / degree
    beta_eff = np.mean(w * w) / np.mean(w)
 
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    t = np.arange(0, 1000, 0.01)

    dynamics_multi = globals()[dynamics + '_multi']
    dynamics_three_level_beta = globals()[dynamics + '_three_level_beta']
    dynamics_three_level_beta = globals()[dynamics + '_three_level_beta']
    dynamics_three_level_decouple = globals()[dynamics + '_three_level_decouple']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_beta = odeint(dynamics_three_level_beta, attractor_value, t, args=(arguments, beta_eff))[-1]
    xs_i_nn = odeint(dynamics_three_level_decouple, initial_condition, t, args=(arguments, w_nn, xs_beta))[-1]
    xs_i = odeint(dynamics_three_level_decouple, initial_condition, t, args=(arguments, w, xs_i_nn))[-1]

    "save data"
    data = np.vstack((np.arange(N_actual), xs_multi, xs_i, xs_i_nn, xs_beta * np.ones(N_actual), w))
    des = '../data/' + dynamics + '/' + network_type + '/xs_three_level/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_three_level_stable(network_type, N, beta, betaeffect, seed_list, d, dynamics, arguments, attractor_value):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(three_level_stable, [(network_type, N, beta, betaeffect, seed, d, dynamics, arguments, attractor_value) for seed in seed_list]).get()
    p.close()
    p.join()
    return None



"evolution"

def ode_stable(dynamics, initial_condition, t_start, t_interval, dt, arguments, error):
    """TODO: Docstring for ode_stable.

    :dynamics: TODO
    :initial_condition: TODO
    :t: TODO
    :arguments: TODO
    :error: TODO
    :returns: TODO

    """
    difference = 100
    i = 0
    while difference > error:
        i += 1
        t = np.arange(t_start, t_start + t_interval, dt)
        x = odeint(dynamics, initial_condition, t, args=arguments)
        x_last = x[-1].copy()
        difference = np.abs(x_last - initial_condition).sum()
        initial_condition = x[-1].copy()
        t_start = t_start + t_interval 
    print(i)
    return x


def group_evolution(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    if ratio_threshold == 'None':
        if space == 'equal_node':
            group_index, rearange_index = group_partition_degree_equal_node(A, group_num, N_actual)
        elif space == 'linear' or space == 'log' or space == 'bimode':
            group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
    else:
        if space == 'equal_node':
            group_index, rearange_index = group_partition_degree_equal_node(A, group_num, N_actual)
        else:
            group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
        group_index, rearange_index = group_movement(A, group_index, rearange_index, ratio_threshold)

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
    args_group = (arguments, net_arguments)
    return group_index, w_group, args_group

def evolution(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space, t_start, t_interval, dt, error, plot_type, plot_group):
    """TODO: Docstring for evolution.

    :network_type: TODO
    :N: TODO
    :dynamics: TODO
    :d: TODO
    :arguments: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)

    group_index, w_group, args_group = group_evolution(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space)
    length_groups = len(group_index)
    dynamics_group = globals()[dynamics + '_multi']
    
    initial_condition_group = attractor_value * np.ones(length_groups)
    xs_group = ode_stable(dynamics_group, initial_condition_group, t_start, t_interval, dt, args_group, error)

    initial_condition = attractor_value * np.ones(N_actual)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    args_multi = (arguments, net_arguments)
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = ode_stable(dynamics_multi, initial_condition, t_start, t_interval, dt, args_multi, error)

    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    xs_group_transpose = xs_group[-1].reshape(length_groups, 1)
    args_group_decouple = (arguments, w_group, xs_group_transpose)
    xs_group_decouple = ode_stable(dynamics_group_decouple, initial_condition, t_start, t_interval, dt, args_group_decouple, error)

    t = np.arange(t_start, t_interval, dt)
    if plot_type == 'alpha':
        "compare x_i and x_i^{alpha}"
        plt.plot(t, xs_multi[:, group_index[plot_group][:-1]], linewidth=2, color='#66c2a5' )
        plt.plot(t, xs_multi[:, group_index[plot_group][-1]], label='original', linewidth=2, color='#66c2a5' )
        plt.plot(t, xs_group[:, plot_group], label='group', linewidth=lw, color='#fc8d62')
    elif plot_type == 'i':
        plt.plot(t, xs_multi[:, group_index[plot_group][0]], linewidth=2, label='original', color='#66c2a5' )
        plt.plot(t, xs_group_decouple[:, group_index[plot_group][0]], label='decouple', linewidth=lw, color='#fc8d62')

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$t$', fontsize=fs)
    plt.ylabel('x', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.show()

    return xs_multi, xs_group, xs_group_decouple



    

seed1 = np.arange(10).tolist()
seed1 = np.array([4])
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
seed_ER = seed1

N = 1000
beta = 0.1
betaeffect = 0
attractor_value = 0.1



dynamics = 'BDP'
arguments = (B_BDP, )


#attractor_value = 0.01
dynamics = 'CW'
arguments = (a, b)

dynamics = 'PPI'
arguments = (B_PPI, F_PPI)


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)


network_type = 'RGG'
d_list = [0.04, 0.05, 0.07]
seed_list = seed_ER
group_num_list = np.arange(1, 10, 1)

network_type = 'ER'
d_list = [2000, 4000, 8000]
seed_list = seed_ER
group_num_list = np.arange(1, 10, 1)

network_type = 'SF'
d_list = [[2.1, 999, 2], [2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_list = [[2.5, 999, 3]]
seed_list = seed_SF
group_num_list = np.arange(1, 20, 1)




r_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
r_list = [0.7, 0.8, 0.9, 1]
r_list = ['None']

space = 'log'

iteration_step = 10
for r in r_list:
    for d in d_list:
        for group_num in group_num_list:
            #parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space)
            parallel_group_decouple_nn_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space, iteration_step)
            pass
        #parallel_three_level_stable(network_type, N, beta, betaeffect, seed_list, d, dynamics, arguments, attractor_value)
        pass



space = 'log'
plot_type = 'i'
plot_type = 'alpha'
plot_type ='None'
plot_group = 0
r = 'None'


network_type = 'SF'
d = [2.5, 999, 3]
group_num = 3
seed = [0, 0]
beta = 0.1
attractor_value = 0.1
t_start = 0 
t_interval = 100
dt = 0.01
error = 0.01
#xs_multi, xs_group, xs_group_decouple = evolution(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space, t_start, t_interval, dt, error, plot_type, plot_group)

"analysis"

"""
A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
w = np.sum(A, 0)
N_actual = np.size(A, 0)
w_nn = np.array([w[np.where(A[i] > 0)[0]] for i in range(N_actual)])
xs_multi_last = xs_multi[-1]
xs_high = xs_multi_last[xs_multi_last > 5]
w_high = w[xs_multi_last>5]
w_nn_high = w_nn[xs_multi_last>5]
sort_index = np.argsort(xs_high)
w_high_sort = w_high[sort_index]
w_nn_high_sort = w_nn_high[sort_index]

dynamics_group_decouple = globals()[dynamics + '_group_decouple']
A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
w = np.sum(A, 0)
N_actual = np.size(A, 0)
initial_condition = np.ones(N_actual) * attractor_value
t_start = 0
t_interval = 100
dt = 0.01
error = 0.01

xs_group_decouple_interval = np.linspace(xs_group_decouple[-1].min(), xs_group_decouple[-1].max(), group_num + 8)

xs_group_decouple_interval[0] = xs_group_decouple[-1].min() * 0.5
xs_group_decouple_interval[-1] = xs_group_decouple[-1].max() * 2
regroup_index = []
for i, j in zip(xs_group_decouple_interval[:-1], xs_group_decouple_interval[1:]):
    index = np.where((xs_group_decouple[-1] < j)  & (xs_group_decouple[-1] >= i ))[0]
    #index = np.where((xs_multi[-1] < j)  & (xs_multi[-1] >= i ))[0]
    if len(index):
        regroup_index.append(index)
rearange_index = np.hstack((regroup_index))

length_groups = len(regroup_index)
each_group_length = [len(i) for i in regroup_index]

"degree in each subgroup for all nodes"
A_rearange = A[rearange_index]
reduce_index = np.hstack((0, np.cumsum(each_group_length)))
w_regroup = np.add.reduceat(A_rearange, reduce_index[:-1])

"construct reduction adjacency matrix"
A_reduction = np.zeros((length_groups, length_groups))
for  i in range(length_groups):
    for j in range(length_groups):
        k_i = w[regroup_index[i]] 
        A_reduction[i, j] = np.sum(k_i * np.sum(A[regroup_index[i]][:, regroup_index[j]], 1)) / k_i.sum()
A_index = np.where(A_reduction>0)
A_interaction = A_reduction[A_index]
index_i = A_index[0] 
index_j = A_index[1] 
degree_reduction = np.sum(A_reduction>0, 1)
cum_index = np.hstack((0, np.cumsum(degree_reduction)))
net_arguments = (index_i, index_j, A_interaction, cum_index)
args_regroup = (arguments, net_arguments)
dynamics_regroup = globals()[dynamics + '_multi']
initial_condition_regroup = attractor_value * np.ones(length_groups)
xs_regroup = ode_stable(dynamics_regroup, initial_condition_regroup, t_start, t_interval, dt, args_regroup, error)


dynamics_regroup_decouple = globals()[dynamics + '_group_decouple']
xs_regroup_transpose = xs_regroup[-1].reshape(length_groups, 1)
args_regroup_decouple = (arguments, w_regroup, xs_regroup_transpose)
xs_regroup_decouple = ode_stable(dynamics_regroup_decouple, initial_condition, t_start, t_interval, dt, args_regroup_decouple, error)


"""
