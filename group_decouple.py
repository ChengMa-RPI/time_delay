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
from collections import Counter

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

def mutual_partknown(x, t, arguments, w, xs_known):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + np.sum(w * x * xs_known / (D + E * x + H * xs_known) )
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
    """BDP model

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

def group_partition_state_degree(xs, w, group_num, N_actual, space, diff_states):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    xs_sort_diff = np.diff(np.sort(xs))
    num_gaps = np.sum(xs_sort_diff > diff_states) 
    if num_gaps == 0 or group_num == 1:
        return group_partition_degree(w, group_num, N_actual, space)
    elif num_gaps == 1:
        index_transition = np.where(xs_sort_diff > diff_states)[0]
        xs_low_criteria = np.sort(xs)[index_transition] + 1e-8
        xs_high_criteria = np.sort(xs)[index_transition + 1] - 1e-8
        index_w_low = np.where(xs < xs_low_criteria)[0]
        index_w_high = np.where(xs > xs_high_criteria)[0]
        w_unique = np.unique(w)
        if group_num > np.size(w_unique):
            if group_num == N_actual:
                group_index = []
                for i in range(N_actual):
                    group_index.append([i])
            else:
                print('method not available')
                return None
        elif group_num == 2:
            group_index = [index_w_low, index_w_high]

        else:
            length_groups = 0
            bins = 1
            while group_num:
                group_low_index = []
                group_high_index = []
                if space == 'log':
                    w_separate = np.logspace(np.log10(w.min()), np.log10(w.max()), bins)
                elif space == 'linear':
                    w_separate = np.linspace(w.min(), w.max(), bins)
                elif space == 'bimode':
                    w_separate = np.linspace(w.min(), w.max(), bins)
                
                w_separate[-1] = w_separate[-1] * 2
                w_separate[0] = w_separate[0] *0.5
                for w_i, w_j in zip(w_separate[:-1], w_separate[1:]):
                    index = np.where((w < w_j)  & (w >= w_i ))[0]
                    index_low = np.intersect1d(index, index_w_low)
                    index_high = np.intersect1d(index, index_w_high)
                    if len(index_low):
                        group_low_index.append(index_low)
                    if len(index_high):
                        group_high_index.append(index_high)
                length_groups = len(group_low_index) + len(group_high_index)
                if length_groups <= group_num:
                    bins += 1
                elif length_groups > group_num:
                    break
            group_index = group_low_index
            group_index.extend(group_high_index)
        rearange_index = np.hstack((group_index))
        if len(rearange_index) != N_actual:
            print(w_separate, len(rearange_index))
            print('groups wrong')
        return group_index, rearange_index
    elif num_gaps > 1:
        print('num_gaps:', num_gaps, xs_sort_diff[-10:])
        return None

def group_partition_degree_core(A, group_num, N_actual, space):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    G = nx.convert_matrix.from_numpy_matrix(A)
    core_number = np.array(list(nx.core_number(G).values())) 
    core_neighbor = [core_number[np.where(A[i] > 0)[0]] for i in range(N_actual)]
    core_neighbor_sum = np.array([np.sum(i) for i in core_neighbor])

    w = np.sum(A, 0)
    indicator = w + core_neighbor_sum / (core_neighbor_sum.max()+1) + np.random.random(N_actual) / core_neighbor_sum.max()
    indicator_unique = np.unique(indicator)
    if group_num > np.size(indicator_unique):
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
                indicator_separate = np.logspace(np.log10(indicator.min()), np.log10(indicator.max()), bins)
            elif space == 'linear':
                indicator_separate = np.linspace(indicator.min(), indicator.max(), bins)
            elif space == 'bimode':
                indicator_separate = np.linspace(indicator.min(), indicator.max(), bins)
            
            indicator_separate[-1] = indicator_separate[-1] * 2
            indicator_separate[0] = indicator_separate[0] *0.5
            group_index = []
            for indicator_i, indicator_j in zip(indicator_separate[:-1], indicator_separate[1:]):
                index = np.where((indicator < indicator_j)  & (indicator >= indicator_i ))[0]
                if len(index):
                    group_index.append(index)
            length_groups = len(group_index)
            bins += 1
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(indicator_separate, len(rearange_index))
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
    neighbors_degree = [w[np.where(A[i] > 0 )[0]] for i in range(N_actual)]
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
            #group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
            group_index, rearange_index = group_partition_degree_core(A, group_num, N_actual, space)
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

def group_state(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    w = xs
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)

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

def group_state_two_cluster(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs, diff_states):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    w = np.sum(A, 0)
    group_index, rearange_index = group_partition_state_degree(xs, w, group_num, N_actual, space, diff_states)

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

def group_core(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    G = nx.convert_matrix.from_numpy_matrix(A)
    core_number = np.array(list(nx.core_number(G).values())) 
    core_neighbor = [core_number[np.where(A[i] > 0)[0]] for i in range(N_actual)]
    core_neighbor_sum = np.array([np.sum(i) for i in core_neighbor])

    w = np.sum(A, 0)
    group_index, rearange_index = group_partition_degree(core_neighbor_sum, group_num, N_actual, space)

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



def group_decouple_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space, partition_indicator):
    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    if partition_indicator == 'weights':
        xs_group, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space)
    elif partition_indicator == 'core_neighbor_sum':
        xs_group, group_index, w_group = group_core(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, ratio_threshold, space)

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
    if partition_indicator == 'weights':
        if ratio_threshold == 'None':
            des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
        else:
            des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_move_' + space + '/'

    elif partition_indicator == 'core_neighbor_sum':
        if ratio_threshold == 'None':
            des = '../data/' + dynamics + '/' + network_type + '/xs_group_core_neighbor_sum_' + space + '/'
        else:
            des = '../data/' + dynamics + '/' + network_type + '/xs_group_core_neighbor_sum_move_' + space + '/'


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

def parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space, partition_indicator):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_decouple_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, r, space, partition_indicator) for seed in seed_list]).get()
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

def group_iteration_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    xs_beta, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, 1, dynamics, arguments, attractor_value, r, space)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    t = np.arange(0, 1000, 0.01)

    net_arguments = (index_i, index_j, A_interaction, cum_index)

    length_groups = len(xs_beta)
    xs_group_transpose = xs_beta.reshape(length_groups, 1)
    dynamics_multi = globals()[dynamics + '_multi']
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    dynamics_nearest_neighbor = globals()[dynamics + '_nearest_neighbor']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_group_decouple = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    xs_nn = xs_group_decouple.copy()
    xs_all_list = []
    for l in range(iteration_step):
        xs_nn = odeint(dynamics_nearest_neighbor, initial_condition, t, args=(arguments, net_arguments, xs_nn))[-1]
        xs_group, group_index, w_group = group_state(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs_nn)
        rearange_index = np.hstack((group_index))
        group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
        xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
        length_groups = len(xs_group)
        xs_group_transpose = xs_group.reshape(length_groups, 1)
        xs_group_decouple_iteration = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]

        xs_all = np.vstack((group_number, rearange_index, xs_nn[rearange_index], xs_groups, xs_group_decouple_iteration[rearange_index]))
        xs_all_list.append(xs_all)


    "save data"
    data=  np.vstack((xs_multi, w, xs_beta * np.ones(N_actual), xs_group_decouple, np.vstack((xs_all_list))))
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_iteraction_{iteration_step}_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_group_iteration_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_iteration_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def group_iteration_two_cluster_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """

    xs_beta, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, 1, dynamics, arguments, attractor_value, r, space)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    t = np.arange(0, 1000, 0.01)

    net_arguments = (index_i, index_j, A_interaction, cum_index)

    length_groups = len(xs_beta)
    xs_group_transpose = xs_beta.reshape(length_groups, 1)
    dynamics_multi = globals()[dynamics + '_multi']
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    dynamics_nearest_neighbor = globals()[dynamics + '_nearest_neighbor']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_group_decouple = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]
    xs_nn = xs_group_decouple.copy()
    xs_all_list = []
    for l in range(iteration_step):
        xs_nn = odeint(dynamics_nearest_neighbor, initial_condition, t, args=(arguments, net_arguments, xs_nn))[-1]
        xs_group, group_index, w_group = group_state_two_cluster(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs_nn, diff_states)
        rearange_index = np.hstack((group_index))
        group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
        xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
        length_groups = len(xs_group)
        xs_group_transpose = xs_group.reshape(length_groups, 1)
        xs_group_decouple_iteration = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]

        xs_all = np.vstack((group_number, rearange_index, xs_nn[rearange_index], xs_groups, xs_group_decouple_iteration[rearange_index]))
        xs_all_list.append(xs_all)


    "save data"
    data=  np.vstack((xs_multi, w, xs_beta * np.ones(N_actual), xs_group_decouple, np.vstack((xs_all_list))))
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_group_iteration_two_cluster_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(group_iteration_two_cluster_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states) for seed in seed_list]).get()
    p.close()
    p.join()
    return None


def index_next_calculation(xs_adaptive, w, neighbors, xs_high_criteria, index_calculated):
    """TODO: Docstring for index_next_calculation.

    :xs_adaptive: TODO
    :w: TODO
    :returns: TODO

    """
    index_high = np.where(xs_adaptive > xs_high_criteria)[0]
    if len(index_high) == 0:
        index_cal = [np.argsort(w)[-1]]
    elif len(index_high):
        high_neighbor = [neighbors[index_high_i] for index_high_i in index_high]
        high_neighbor_all = np.hstack((high_neighbor))
        high_neighbor_counter = Counter(high_neighbor_all)
        counter_keys = np.array(list(high_neighbor_counter.keys()))
        counter_values = np.array(list(high_neighbor_counter.values()))
        counter_values_unique = np.unique(counter_values)
        index_cal = []
        for i in counter_values_unique[::-1]:
            high_neighbor_i = counter_keys[np.where(counter_values == i)[0]]
            high_neighbor_i_candidate = np.setdiff1d(high_neighbor_i, index_calculated)
            if len(high_neighbor_i_candidate):
                high_neighbor_i_max_degree = high_neighbor_i_candidate[np.argsort(w[high_neighbor_i_candidate])[::-1]]
                index_cal.extend(high_neighbor_i_max_degree)
    return index_cal

def xs_adaptive_calculation(xs_adaptive, w, neighbors, xs_high_criteria, A, arguments, xs_beta, attractor_value):
    """TODO: Docstring for xs_adaptive.

    :arg1: TODO
    :returns: TODO

    """
    index_calculated = []
    t = np.arange(0, 1000, 0.01)
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    change = 1
    last_step_all_once = 0
    while last_step_all_once == 0:
        if change == 0:
            last_step_all_once = 1
            index_all = np.arange(len(xs_adaptive))
            index_last_step = np.setdiff1d(index_all, index_calculated)
            initial_condition = np.ones(len(index_last_step)) * attractor_value
            dynamics_nearest_neighbor = globals()[dynamics + '_nearest_neighbor']
            A_last_step = A[index_last_step]
            A_index = np.where(A_last_step>0)
            A_interaction = A_last_step[A_index]
            index_i = A_index[0] 
            index_j = A_index[1] 
            degree = np.sum(A_last_step>0, 1)
            cum_index = np.hstack((0, np.cumsum(degree)))
            net_arguments = (index_i, index_j, A_interaction, cum_index)
            xs_adaptive[index_last_step] = odeint(dynamics_nearest_neighbor, initial_condition, t, args=(arguments, net_arguments, xs_adaptive))[-1]
            break

        elif change > 0:
            index_cal_list = index_next_calculation(xs_adaptive, w, neighbors, xs_high_criteria, index_calculated)
            change = 0
            for index_cal in index_cal_list:
                index_cal_neighbor = neighbors[index_cal]
                w_cal = A[index_cal]
                w_cal_eff_index = np.where(w_cal > 0)[0]
                xs_cal_neighbor = xs_adaptive[w_cal_eff_index]
                xs_index_cal = odeint(dynamics_group_decouple, attractor_value, t, args=(arguments, w_cal[w_cal_eff_index], xs_cal_neighbor))[-1]
                xs_adaptive[index_cal] = xs_index_cal
                if xs_index_cal > xs_high_criteria :
                    index_calculated.append(index_cal)
                    change += 1
                else:
                    if change > 0:
                        break
    return xs_adaptive 

def group_iteration_adaptive_two_cluster_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states, xs_high_criteria):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    #xs_beta, group_index, w_group = group_degree_weighted(network_type, N, beta, betaeffect, seed, d, 1, dynamics, arguments, attractor_value, r, space)
    xs_all_list = []
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    t = np.arange(0, 1000, 0.01)
    xs_beta = odeint(mutual_group_decouple, attractor_value, t, args=(arguments, 0, 10))[-1]
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    neighbors = [np.where(A[i] > 0)[0] for i in range(N_actual)]
    initial_condition = attractor_value * np.ones(N_actual)
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_adaptive = xs_beta * np.ones(N_actual)
    for l in range(iteration_step):
        xs_adaptive = xs_adaptive_calculation(xs_adaptive, w, neighbors, xs_high_criteria, A, arguments, xs_beta, attractor_value)
        xs_group, group_index, w_group = group_state_two_cluster(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs_adaptive, diff_states)
        rearange_index = np.hstack((group_index))
        group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
        xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
        length_groups = len(xs_group)
        xs_group_transpose = xs_group.reshape(length_groups, 1)
        xs_group_decouple_iteration = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]

        xs_all = np.vstack((group_number, rearange_index, xs_adaptive[rearange_index], xs_groups, xs_group_decouple_iteration[rearange_index]))
        xs_all_list.append(xs_all)

    "save data"
    data=  np.vstack((xs_multi, w, xs_beta * np.ones(N_actual), np.vstack((xs_all_list))))
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def iteration_adaptive(network_type, N, beta, betaeffect, seed, d, dynamics, arguments, attractor_value, iteration_step, xs_high_criteria):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    xs_all_list = []
    t = np.arange(0, 1000, 0.01)
    xs_beta = odeint(mutual_group_decouple, attractor_value, t, args=(arguments, 0, 10))[-1]
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    neighbors = [np.where(A[i] > 0)[0] for i in range(N_actual)]
    initial_condition = attractor_value * np.ones(N_actual)
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    xs_adaptive = xs_beta * np.ones(N_actual)
    for l in range(iteration_step):
        xs_adaptive = xs_adaptive_calculation(xs_adaptive, w, neighbors, xs_high_criteria, A, arguments, xs_beta, attractor_value)
        xs_all_list.append(xs_adaptive.copy())

    "save data"
    data=  np.vstack((xs_all_list))
    des = '../data/' + dynamics + '/' + network_type + f'/xs_adaptive_iteration_step={iteration_step}/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def group_iteration_adaptive_two_cluster_stable_improve(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states):

    """TODO: Docstring for group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    xs_all_list = []
    dynamics_group_decouple = globals()[dynamics + '_group_decouple']
    t = np.arange(0, 1000, 0.01)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    xs_beta = odeint(mutual_group_decouple, attractor_value, t, args=(arguments, 0, 10))[-1]
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    initial_condition = attractor_value * np.ones(N_actual)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_adaptive_des = '../data/' + dynamics + '/' + network_type + f'/xs_adaptive_iteration_step={iteration_step}/'
    if betaeffect:
        xs_adaptive_des_file = xs_adaptive_des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_seed={seed}.csv'
    else:
        xs_adaptive_des_file = xs_adaptive_des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_seed={seed}.csv'
    xs_adaptive_list = np.array(pd.read_csv(xs_adaptive_des_file, header=None).iloc[:, :])

    for l in range(iteration_step):
        xs_adaptive = xs_adaptive_list[:, l]
        xs_group, group_index, w_group = group_state_two_cluster(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, xs_adaptive, diff_states)
        rearange_index = np.hstack((group_index))
        group_number = np.hstack(([i * np.ones(len(j)) for i, j in enumerate(group_index)]))
        xs_groups = np.hstack(([xs_group[i] * np.ones(len(j)) for i, j in enumerate(group_index)]))
        length_groups = len(xs_group)
        xs_group_transpose = xs_group.reshape(length_groups, 1)
        xs_group_decouple_iteration = odeint(dynamics_group_decouple, initial_condition, t, args=(arguments, w_group, xs_group_transpose))[-1]

        xs_all = np.vstack((group_number, rearange_index, xs_adaptive[rearange_index], xs_groups, xs_group_decouple_iteration[rearange_index]))
        xs_all_list.append(xs_all)

    "save data"
    data=  np.vstack((xs_multi, w, xs_beta * np.ones(N_actual), np.vstack((xs_all_list))))
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'

    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    df = pd.DataFrame(data.transpose())
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None



def parallel_group_iteration_adaptive_two_cluster_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states, xs_high_criteria):
    """TODO: Docstring for parallel_group_decouple_stable.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    #p.starmap_async(group_iteration_adaptive_two_cluster_stable, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states, xs_high_criteria) for seed in seed_list]).get()
    p.starmap_async(group_iteration_adaptive_two_cluster_stable_improve, [(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states) for seed in seed_list]).get()
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



    

seed1 = np.array([6])
seed1 = np.arange(10).tolist()
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
seed_ER = seed1

N = 1000
beta = 0.5
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
group_num_list = np.array([1])
group_num_list = np.arange(1, 20, 1)




r_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
r_list = [0.7, 0.8, 0.9, 1]
r_list = ['None']

space = 'log'
partition_indicator = 'core_neighbor_sum'
partition_indicator = 'weights'

iteration_step = 3
diff_states = 4
xs_high_criteria = 5

for d in d_list:
    for seed in seed_list:
        #iteration_adaptive(network_type, N, beta, betaeffect, seed, d, dynamics, arguments, attractor_value, iteration_step, xs_high_criteria)
        pass

for r in r_list:
    for d in d_list:
        for group_num in group_num_list:
            parallel_group_decouple_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space, partition_indicator)
            #parallel_group_decouple_nn_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, r, space, iteration_step)
            #parallel_group_iteration_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step)
            #parallel_group_iteration_two_cluster_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states)
            #parallel_group_iteration_adaptive_two_cluster_stable(network_type, N, beta, betaeffect, seed_list, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states, xs_high_criteria)
            pass
        #parallel_three_level_stable(network_type, N, beta, betaeffect, seed_list, d, dynamics, arguments, attractor_value)
        pass

xs_high_criteria = 5
seed = [0, 0]
#index_calculated = group_iteration_adaptive_two_cluster_stable(network_type, N, beta, betaeffect, seed, d, group_num, dynamics, arguments, attractor_value, space, iteration_step, diff_states, xs_high_criteria)

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




dfdxj = (xi * (D + E * xi + H * xj) - xi * xj * H ) / (D + E * xi + H * xj) ** 2 
dfdxi = k * (xj * (D + E * xi + H * xj) - xi * xj * E ) / (D + E * xi + H * xj) ** 2  + (1/C + 1/K) * 2 * xi - 3 *xi**2/C/K - 1
"""
