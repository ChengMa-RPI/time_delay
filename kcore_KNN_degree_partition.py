from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng, dde_RK45

import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
import networkx as nx
import multiprocessing as mp
import time
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
import os

"""this is for networks with identical edge weights"""



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

def mutual_1D(x, t, beta, arguments):
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

def SIS_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x 
    sum_g = A_interaction * (1-x[index_i]) * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
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

def find_core_group(G):
    """TODO: Docstring for group_stable.

    :arg1: TODO
    :returns: TODO

    """
    core_number = np.array(list(nx.core_number(G).values())) 
    core_group = np.where(core_number == core_number.max())[0]
    return core_group

def neighborshell_given_core(G, A, core_group):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
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
    return KNN

"network partition approach"

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

def reducednet_degree(A, x, number_groups, space):
    """TODO: Docstring for reduced_network.

    :arg1: TODO
    :returns: TODO

    """
    k = np.sum(A>0, 0)
    N_actual = len(A)
    group_index, rearange_index = group_partition_degree(k, number_groups, N_actual, space)
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index

def kcore_KNN(G, A, core_group, x, regroup):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    KNN = neighborshell_given_core(G, A, core_group)
    if regroup == 'None':
        group_index = KNN
    else:
        group_index = [] 
        for i in regroup:
            group_index.append(np.hstack(([KNN[j] for j in i])).tolist())
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index

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
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index

def kcore_KNN_degree(G, A, core_group, x, regroup_criteria, space, degree_interval, data_noregroup):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    KNN = neighborshell_given_core(G, A, core_group)
    k = np.sum(A>0, 0)
    KNN_degree = []
    for i in KNN:
        N_i = len(i)
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
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index

def kcore_degree(G, A, x, space, degree_interval):
    """TODO: Docstring for neighbors_shell.

    :G: TODO
    :returns: TODO

    """
    core_number = np.array(list(nx.core_number(G).values())) 
    core_number_unique = np.sort(np.unique(core_number))[::-1]
    kcore_group = []
    for i in core_number_unique:
        group_i = np.where(core_number == i)[0]
        kcore_group.append(group_i)
    k = np.sum(A>0, 0)
    kcore_degree = []
    for i in kcore_group:
        N_i = len(i)
        k_i = k[i]
        group_index_i, rearange_index_i = group_partition_degree_interval(k_i, degree_interval, N_i, space)
        for j in group_index_i:
            kcore_degree.append(i[j])
    group_index = kcore_degree
    A_reduction, net_arguments, x_eff = reducednet_effstate(A, x, group_index)
    return A_reduction, net_arguments, x_eff, group_index


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

def group_partition_degree_interval(k, degree_interval, N_actual, space):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    k_unique = np.unique(k)
    k_max = np.max(k)
    k_min = np.min(k)
    if space == 'log':
        m = max(1, np.int(np.ceil(np.log(k_max / k_min) / degree_interval)))
        k_separate = [k_min * np.exp(i * degree_interval) for i in range(m+1)]
    elif space == 'linear':
        m = max(1, np.int(np.ceil((k_max - k_min) / degree_interval)))
        k_separate = [k_min + (i * degree_interval) for i in range(m+1)]
    k_separate[-1] = k_separate[-1] * 1.1
    k_separate[0] = k_separate[0] *0.9
    group_index = []
    for k_i, k_j in zip(k_separate[:-1], k_separate[1:]):
        index = np.where((k < k_j)  & (k >= k_i ))[0]
        if len(index):
            group_index.append(index)
    group_index = group_index[::-1]
    rearange_index = np.hstack((group_index))
    if len(rearange_index) != N_actual:
        print(k_separate, len(rearange_index))
        print('groups wrong')
    return group_index, rearange_index


"network partition, reduction state, and data saving"

def degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + f'/bifurcation_group_decouple_' + space + '/' 
    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by degree partition"
    for number_groups in number_groups_list:
        A_reduction_deg_part, net_arguments_reduction_deg_part, x_eff_deg_part, group_index = reducednet_degree(A, xs_multi, number_groups, space)
        initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
        xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
        "save data"
        data=  np.hstack((weight, np.ravel(A_reduction_deg_part), x_eff_deg_part, xs_reduction_deg_part))
        des_file = des + f'N={N}_d=' + str(d) + f'_group_num={number_groups}_seed={seed}.csv'
        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def kcore_KNN_regroup_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup):
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
    w = np.sum(A, 0)
    A_reduction, net_arguments_reduction, x_eff, group_index = kcore_KNN(G, A, core_group, xs_multi, regroup)
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

def kcore_KNN_degree_partition_regroup_criteria(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, regroup_criteria, degree_interval, space):
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
    A_reduction, net_arguments_reduction, x_eff, group_index = kcore_KNN_degree(G, A, core_group, xs_multi, regroup_criteria, space, degree_interval, data_noregroup)
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
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by KNN partition"
    G = nx.from_numpy_array(A)
    A_reduction, net_arguments_reduction, x_eff, group_index = kcore_shell(G, A, xs_multi, regroup)
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

def kcore_degree_partition(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, degree_interval, space):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore_degree_' + space + f'={degree_interval}/'
    if not os.path.exists(des):
        os.makedirs(des)

    "the original network"
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)

    N_actual = len(A)
    w = np.sum(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by kcore_degree partition"
    G = nx.from_numpy_array(A)
    A_reduction, net_arguments_reduction, x_eff, group_index = kcore_degree(G, A, xs_multi, space, degree_interval)
    initial_condition_reduction = np.ones(len(A_reduction)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction, t, args=(arguments, net_arguments_reduction))[-1]

    "save data"
    data=  np.hstack((weight, np.ravel(A_reduction), x_eff, xs_reduction))
    des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

