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

def mutual_1D_spectral(x, t, alpha, beta, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + alpha * beta * x**2 / (D + E*x*beta +H*x)
    return dxdt

def SBM_ER(N_group, p, weight, seed):
    """TODO: Docstring for SBM_ER.

    :N_group: TODO
    :p: TODO
    :seed: TODO
    :returns: TODO

    """
    G = nx.stochastic_block_model(N_group, p, seed=seed)
    A = nx.to_numpy_array(G)
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    return A, A_interaction, index_i, index_j, cum_index

def reducednet_effstate(A, x, group_index):
    """TODO: for network with communities.

    :arg1: TODO
    :returns: TODO

    """
    length_groups = len(group_index)
    rearange_index = np.hstack((group_index))
    x_eff = np.zeros(length_groups)
    w = np.sum(A, 0)
    each_group_length = [len(i) for i in group_index]
    "degree in each subgroup for all nodes"
    A_rearange = A[rearange_index]
    reduce_index = np.hstack((0, np.cumsum(each_group_length)))
    w_group = np.add.reduceat(A_rearange, reduce_index[:-1])
    "construct reduction adjacency matrix"
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        w_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * w_i) / np.mean(w_i)
        for j in range(length_groups):
            "the interaction from group j to group i"
            A_reduction[i, j] = np.sum(w_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / w_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    return A_reduction, net_arguments, x_eff

def reducednet_from_community(N_group, A, x, number_groups):
    """TODO: for network with communities.

    :arg1: TODO
    :returns: TODO

    """
    group_cum = np.hstack((0, np.cumsum(N_group)))
    N = sum(N_group)
    node_index = np.arange(N)
    group_index = [node_index[i:j] for i, j in zip(group_cum[:-1], group_cum[1:])]
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index

def find_core_group(G):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    core_number = np.array(list(nx.core_number(G).values())) 
    core_group = np.where(core_number == core_number.max())[0]
    return core_group

def neighbors_shell(G, A, core_group, x, regroup):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    neighbors = G.neighbors
    N = len(G.nodes())
    allocated_nodes = core_group
    KNN = []
    KNN.append(core_group)
    neighbors_target = core_group
    while len(allocated_nodes) < N:
        k_neighbors_all = []
        for i in neighbors_target:
            k_neighbors_all.extend(list(neighbors(i)))

        k_neighbors = np.setdiff1d(np.unique(k_neighbors_all), allocated_nodes)
        allocated_nodes = np.append(allocated_nodes, k_neighbors)
        neighbors_target = k_neighbors
        KNN.append(k_neighbors)
    if regroup == 'None':
        group_index = KNN
    else:
        group_index = [] 
        for i in regroup:
            group_index.append(np.hstack(([KNN[j] for j in i])).tolist())

    length_groups = len(group_index)
    "construct reduction adjacency matrix"
    x_eff = np.zeros(length_groups)
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        k_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * k_i) / np.mean(k_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    return A_reduction, net_arguments, x_eff

def kcore_shell(G, A, x, regroup):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """

    w = np.sum(A, 0)
    core_number = np.array(list(nx.core_number(G).values())) 
    core_number_unique = np.sort(np.unique(core_number))[::-1]
    kcore_group = []
    for i in core_number_unique:
        group_i = np.where(core_number == i)[0]
        kcore_group.append(group_i)

    if regroup == 'None':
        group_index = kcore_group
    else:
        group_index = [] 
        for i in regroup:
            group_index.append(np.hstack(([kcore_group[j] for j in i])).tolist())

    length_groups = len(group_index)
    "construct reduction adjacency matrix"
    x_eff = np.zeros(length_groups)
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        k_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * k_i) / np.mean(k_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    return A_reduction, net_arguments, x_eff

def neighbors_degree_parallel_shell(G, A, core_group, x, regroup, space, degree_interval):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    neighbors = G.neighbors
    N = len(G.nodes())
    allocated_nodes = core_group
    KNN = []
    KNN.append(core_group)
    neighbors_target = core_group
    while len(allocated_nodes) < N:
        k_neighbors_all = []
        for i in neighbors_target:
            k_neighbors_all.extend(list(neighbors(i)))

        k_neighbors = np.setdiff1d(np.unique(k_neighbors_all), allocated_nodes)
        allocated_nodes = np.append(allocated_nodes, k_neighbors)
        neighbors_target = k_neighbors
        KNN.append(k_neighbors)

    KNN_degree = []
    for i in KNN:
        N_i = len(i)
        w_i = w[i]
        group_index_i, rearange_index_i = group_partition_degree_interval(w_i, degree_interval, N_i, space)
        for j in group_index_i:
            KNN_degree.append(i[j])
    if regroup == 'None':
        group_index = KNN_degree
    else:
        group_index = [] 
        for i in regroup:
            group_index.append(np.hstack(([KNN_degree[j] for j in i])).tolist())

    length_groups = len(group_index)
    "construct reduction adjacency matrix"
    x_eff = np.zeros(length_groups)
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        k_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * k_i) / np.mean(k_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    return A_reduction, net_arguments, x_eff

def neighbors_degree_parallel_shell_regroup(G, A, core_group, x, regroup_criteria, space, degree_interval, data_noregroup):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    k = np.sum(A>0, 0)
    neighbors = G.neighbors
    N = len(G.nodes())
    allocated_nodes = core_group
    KNN = []
    KNN.append(core_group)
    neighbors_target = core_group
    while len(allocated_nodes) < N:
        k_neighbors_all = []
        for i in neighbors_target:
            k_neighbors_all.extend(list(neighbors(i)))

        k_neighbors = np.setdiff1d(np.unique(k_neighbors_all), allocated_nodes)
        allocated_nodes = np.append(allocated_nodes, k_neighbors)
        neighbors_target = k_neighbors
        KNN.append(k_neighbors)

    KNN_degree = []
    for i in KNN:
        N_i = len(i)
        w_i = w[i]
        k_i = k[i]
        group_index_i, rearange_index_i = group_partition_degree_interval(k_i, degree_interval, N_i, space)
        for j in group_index_i:
            KNN_degree.append(i[j])

    if regroup_criteria == 'None':
        group_index = KNN_degree
    else:
        length_groups = len(KNN_degree)

        xs_reduction = data_noregroup[-1, length_groups**2+length_groups+1:length_groups**2+2*length_groups+1]
        regroup_index = []
        ungroup = np.arange(length_groups)
        while len(ungroup):
            xs_target_index = ungroup[np.argmax(xs_reduction[ungroup])]
            xs_target = xs_reduction[xs_target_index]
            xs_target_group_index = [xs_target_index]
            ungroup = np.delete(ungroup, np.argmax(xs_reduction[ungroup]))
            within_group_index = np.where( (xs_target - xs_reduction[ungroup]) / xs_target < regroup_criteria)[0]
            if len(within_group_index ):
                xs_target_append_index = ungroup[within_group_index]
                xs_target_group_index.extend(xs_target_append_index.tolist())
            regroup_index.append(xs_target_group_index)
            ungroup = np.delete(ungroup, within_group_index)
        group_index = []
        for i in regroup_index:
            group_index.append(np.hstack(([KNN_degree[j] for j in i])).tolist())
    length_groups = len(group_index)
    "construct reduction adjacency matrix"
    x_eff = np.zeros(length_groups)
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        w_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * w_i) / np.mean(w_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(w_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / w_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    return A_reduction, net_arguments, x_eff, group_index

def neighbors_degree_shell(G, A, core_group, x, regroup, degree_interval, space):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    neighbors = G.neighbors
    N = len(G.nodes())
    allocated_nodes = core_group
    core_index = []
    group_index = []
    group_index_degree_mean = []
    group_index.append(core_group)
    group_index_degree_mean.append(np.mean(w[core_group]))
    neighbors_target = core_group
    core_index.append(0)
    while len(allocated_nodes) < N:
        k_neighbors_all = []
        for i in neighbors_target:
            k_neighbors_all.extend(list(neighbors(i)))

        k_neighbors = np.setdiff1d(np.unique(k_neighbors_all), allocated_nodes)
        N_k_neighbors = len(k_neighbors)
        w_k_neighbors = w[k_neighbors]
        if N_k_neighbors:
            group_index_k_neighbors, rearange_index_k_neighbors = group_partition_degree_interval(w_k_neighbors, degree_interval, N_k_neighbors, space)
            for j in group_index_k_neighbors:
                group_index.append(k_neighbors[j])
                group_index_degree_mean.append(np.mean(w[k_neighbors[j]]))

            for index in np.argsort(group_index_degree_mean)[::-1]: 
                if  index not in core_index:
                    core_index.append(index)
                    break
            neighbors_target = group_index[core_index[-1]]

        else:
            group_index.append(np.setdiff1d(np.arange(N), np.hstack((group_index))))
        allocated_nodes = np.hstack((group_index))
    if regroup == 'None':
        group_index = group_index
    else:
        regroup_index = [] 
        for i in regroup:
            regroup_index.append(np.hstack(([group_index[j] for j in i])).tolist())
        group_index = regroup_index

    length_groups = len(group_index)
    "construct reduction adjacency matrix"
    x_eff = np.zeros(length_groups)
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        k_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * k_i) / np.mean(k_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    return A_reduction, net_arguments, x_eff


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
    group_index = group_index[::-1]
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(w_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index

def group_partition_degree_interval(w, degree_interval, N_actual, space):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    w_unique = np.unique(w)
    w_max = np.max(w)
    w_min = np.min(w)
    if space == 'log':
        m = np.int(np.ceil(np.log(w_max / w_min) / degree_interval))
        m = max(m, 1)
        w_separate = [w_min * np.exp(i * degree_interval) for i in range(m+1)]
    elif space == 'linear':
        m = np.int(np.ceil((w_max - w_min) / degree_interval))
        w_separate = [w_min + (i * degree_interval) for i in range(m+1)]
    
    w_separate[-1] = w_separate[-1] * 2
    w_separate[0] = w_separate[0] *0.5
    group_index = []
    for w_i, w_j in zip(w_separate[:-1], w_separate[1:]):
        index = np.where((w < w_j)  & (w >= w_i ))[0]
        if len(index):
            group_index.append(index)
    group_index = group_index[::-1]
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(w_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index

def reduced_network_group_partition_degree(A, x, number_groups, space):
    """TODO: Docstring for reduced_network.

    :arg1: TODO
    :returns: TODO

    """
    w = np.sum(A, 0)
    N_actual = len(A)
    #space = 'linear'
    group_index, rearange_index = group_partition_degree(w, number_groups, N_actual, space)
    length_groups = len(group_index)
    x_eff = np.zeros(length_groups)
    each_group_length = [len(i) for i in group_index]
    "degree in each subgroup for all nodes"
    A_rearange = A[rearange_index]
    reduce_index = np.hstack((0, np.cumsum(each_group_length)))
    w_group = np.add.reduceat(A_rearange, reduce_index[:-1])

    "construct reduction adjacency matrix"
    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        k_i = w[group_index[i]] 
        x_eff[i] = np.mean(x[group_index[i]] * k_i) / np.mean(k_i)
        for j in range(length_groups):
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    return A_reduction, net_arguments, x_eff

def xs_compare(N_group, p, weight, seed, dynamics, arguments, attractor_value, number_groups):
    """TODO: Docstring for evolution_compare.

    :arg1: TODO
    :returns: TODO

    """
    "the original network"
    A, A_interaction, index_i, index_j, cum_index = SBM_ER(N_group, p, weight, seed)
    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]


    "the reduced system by community"
    A_reduction, net_arguments_reduction, x_eff = reducednet_from_community(N_group, A, xs_multi, number_groups)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "the one-dimension system by degree weighted average"
    dynamics_spectral_1D = globals()[dynamics + '_1D_spectral']
    beta_1D, x_eff_1D = betaspace(A, xs_multi)
    initial_condition_1D = np.ones(1) * attractor_value
    xs_reduction_1D = odeint(dynamics_spectral_1D, initial_condition_1D, t, args=(beta_1D, 1, arguments))[-1]

    "the reduced system by spectral decomposition"
    dynamics_spectral_1D = globals()[dynamics + '_1D_spectral']
    alpha_list, beta_list, R_list = spectral(A, xs_multi, number_groups)
    initial_condition_spectral = np.ones(number_groups) * attractor_value
    xs_spectral = odeint(dynamics_spectral_1D, initial_condition_spectral, t, args=(alpha_list, beta_list, arguments))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction, alpha_list, beta_list, R_list, xs_spectral, beta_1D, x_eff_1D, xs_reduction_1D))
    network_type = 'SBM_ER'
    des = '../data/' + dynamics + '/' + network_type + f'/xs_compare_multi_community_spectral/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'N={N_group}_p=' + str(p.tolist()) + f'_group_num={number_groups}_seed={seed}.csv'

    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return A, xs_multi, A_reduction, x_eff, xs_reduction, alpha_list, beta_list, R_list, xs_spectral, beta_1D, x_eff_1D, xs_reduction_1D

def degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + f'/bifurcation_group_decouple_' + space + '/' 
    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by degree partition"
    for number_groups in number_groups_list:
        A_reduction_deg_part, net_arguments_reduction_deg_part, x_eff_deg_part = reduced_network_group_partition_degree(A, xs_multi, number_groups, space)
        initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
        xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]

        "save data"
        data=  np.hstack((weight, np.ravel(A_reduction_deg_part), x_eff_deg_part, xs_reduction_deg_part))
        des_file = des + f'N={N}_d=' + str(d) + f'_group_num={number_groups}_seed={seed}.csv'

        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def KNN_kcore_regroup_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """

    if regroup == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore/'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_regroup/'

    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by KNN partition"
    G = nx.from_numpy_array(A)
    core_group = find_core_group(G)
    #core_group = np.arange(N_group[0])
    w = np.sum(A, 0)
    A_reduction, net_arguments_reduction, x_eff = neighbors_shell(G, A, core_group, xs_multi, regroup)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction))
    if regroup == 'None':
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup={regroup}.csv'

    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def KNN_kcore_degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup, degree_interval, space, core_N):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """

    if regroup == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_' + space + f'={degree_interval}/'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_regroup/'

    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    w = np.sum(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by KNN partition"
    G = nx.from_numpy_array(A)
    core_group = find_core_group(G)
    core_group = np.argsort(w)[::-1][:core_N]
    A_reduction, net_arguments_reduction, x_eff = neighbors_degree_shell(G, A, core_group, xs_multi, regroup, degree_interval, space)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction))
    if regroup == 'None':
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_core={core_N}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup={regroup}.csv'

    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def KNN_kcore_degree_parallel_partition_regroup_criteria(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup_criteria, degree_interval, space):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """

    if regroup_criteria == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_degree_parallel_' + space + f'={degree_interval}/'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_degree_parallel_regroup_criteria_' + space + f'={degree_interval}/'

    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    w = np.sum(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by KNN partition"
    G = nx.from_numpy_array(A)
    core_group = find_core_group(G)

    if regroup_criteria == 'None':
        data_noregroup = 'None'
    else:
        des_noregroup = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_degree_parallel_' + space + f'={degree_interval}/'
        des_file_noregroup = des_noregroup + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
        data_noregroup = np.array(pd.read_csv(des_file_noregroup, header=None).iloc[:, :])
    A_reduction, net_arguments_reduction, x_eff, group_index = neighbors_degree_parallel_shell_regroup(G, A, core_group, xs_multi, regroup_criteria, space, degree_interval, data_noregroup)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction))
    if regroup_criteria == 'None':
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup_criteria={regroup_criteria}.csv'

    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def kcore_regroup_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """

    if regroup == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore/'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore_regroup/'

    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by KNN partition"
    G = nx.from_numpy_array(A)
    A_reduction, net_arguments_reduction, x_eff = kcore_shell(G, A, xs_multi, regroup)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction))
    if regroup == 'None':
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup={regroup}.csv'

    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None


def evolution(network_type, N, d, weight_list, seed, dynamics, arguments, attractor_value):
    """TODO: Docstring for evolution.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :weight: TODO
    :dynamics: TODO
    :arguments: TODO
    :: TODO
    :returns: TODO

    """

    if network_type == 'SBM_ER':
        A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, 1, seed)
    else:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A)
    core_number = np.array(list(nx.core_number(G).values())) 
    N_actual = len(A)
    k = np.sum(A>0, 0)

    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi_list = np.zeros((len(weight_list), N_actual))
    for i, weight in enumerate(weight_list):
        if network_type == 'SBM_ER':
            A, A_interaction, index_i, index_j, cum_index = SBM_ER(N, d, weight, seed)
        else:
            A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

        net_arguments = (index_i, index_j, A_interaction, cum_index)
        xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_multi_list[i] = xs_multi

    des = '../data/' + dynamics + '/' + network_type + f'/xs_multi/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data=  np.vstack(( np.hstack((0, k)), np.hstack((0, core_number)), np.hstack((weight_list.reshape(len(weight_list), 1), xs_multi_list))))
    df = pd.DataFrame(data)
    df.to_csv(des_file, index=None, header=None, mode='a')

    return data




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


weight_list = np.arange(0.01, 0.5, 0.01)
network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
seed = [0, 0]
seed_list = [[0, 0]]
number_groups_list = [5]
space = 'log'


for seed in seed_list:
    for weight in weight_list:
        #A, xs_multi, A_reduction, x_eff, xs_reduction, alpha_list, beta_list, R_list, xs_spectral, beta_1D, x_eff_1D, xs_reduction_1D = xs_compare(N_group, p, weight, seed, dynamics, arguments, attractor_value, number_groups)
        #degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space)
        pass

N_group = [100, 100, 100]
p = np.ones((3, 3)) * 0.0005
np.fill_diagonal(p, [0.9, 0.5, 0.1]) 

network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
weight_list = np.arange(0.01, 0.5, 0.01)
seed = [0, 0]
regroup = [[0, 1]]
regroup = 'None' 
regroup_criteria = 0.2
regroup_criteria = 'None'
degree_interval = 1.5
space = 'log'
core_N = 1

network_type = 'SBM_ER'
number_groups_list = [3]
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
N = [100, 100]
d = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
seed = 0
seed_list = [0]
degree_interval = 20
space = 'linear'





for weight in weight_list:
    #KNN_kcore_regroup_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup)
    #KNN_kcore_degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup, degree_interval, space, core_N)
    #KNN_kcore_degree_parallel_partition_regroup_criteria(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup_criteria, degree_interval, space)
    #kcore_regroup_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup)
    pass

network_type = 'SBM_ER'
N = [100, 100]
d = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]])
seed = 0
weight_list = np.arange(0.1, 0.25, 0.001)
#data = evolution(network_type, N, d, weight_list, seed, dynamics, arguments, attractor_value)
