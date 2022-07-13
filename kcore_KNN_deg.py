import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng, dde_RK45
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

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
from sklearn.cluster import KMeans



"""this is for networks with identical edge weights"""


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
h = 2
a = 5
b = 1

fs = 22
ticksize = 16
legendsize = 14
alpha = 0.8
lw = 3



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

def feature_from_network_topology(A, G, space, tradeoff_para, method):
    """TODO: Docstring for feature_from_network_topology.

    :off: TODO
    :returns: TODO

    """
    core_number = np.array(list(nx.core_number(G).values())) 
    k = np.sum(A>0, 0)
    N_actual = len(k)
    if method == 'kcore':
        feature = core_number/core_number.max()
    elif method == 'degree':
        if space == 'log':
            feature = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature = k/k.max()
    elif method == 'kcore_degree':
        feature1 = core_number/core_number.max()
        if space ==  'log':
            feature2 = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature2 = k/k.max()
        feature = feature1 * tradeoff_para + feature2 * (1-tradeoff_para)
    elif method == 'kcore_KNN_degree':
        core_group = np.where(core_number == core_number.max())[0]
        KNN = neighborshell_given_core(G, A, core_group)
        KNN_num = np.zeros((N_actual))
        for i, KNN_i in enumerate(KNN):
            for j in KNN_i:
                KNN_num[j] = len(KNN) - i
        feature1 = KNN_num / KNN_num.max()
        if space ==  'log':
            feature2 = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature2 = k/k.max()
        feature = feature1 * tradeoff_para + feature2 * (1-tradeoff_para)
    return feature

def group_index_from_feature_Kmeans(x, number_groups):
    """TODO: Docstring for group_index_from_feature.

    :arg1: TODO
    :returns: TODO

    """
    x_unique = np.unique(x)
    if number_groups > np.size(x_unique):
        first_group_index = []
        for x_i in np.sort(x_unique):
            first_group_index.append(np.where(x == x_i)[0])
        while len(first_group_index) < number_groups:
            group_index = []
            for i, group_i in enumerate(first_group_index):
                if len(group_index) + len(first_group_index[i:]) == number_groups:
                    group_index.extend(first_group_index[i:])
                    break 
                if len(group_i) > 1:
                    group_index.extend(np.array_split(group_i,2))
                else:
                    group_index.append(group_i)
            first_group_index = group_index


        print('number of groups is too large')

    else:
        kmeans = KMeans(n_clusters=number_groups, random_state=0).fit(x.reshape(-1, 1))
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        group_index = []
        for i in np.argsort(centers[:, 0]):
            group_index.append(np.where(labels == i)[0])
    return group_index[::-1]

def group_partition_sa(network_type, N, d, seed, number_groups, N_trial, T_info):
    """TODO: Docstring for group_partition_log_degree.
    :returns: TODO

    """
    T0, T_num, T_decay = T_info
    des_file = '../data/network_partition/' + network_type  + '/' + f'N={N}_d={d}_seed={seed}_number_groups={number_groups}_trial={N_trial}_T=[{T0}, {T_num}, {T_decay}].txt'
    data = []
    group_index = []
    with open(des_file, 'r') as f:
        data.append(f.readlines())
    for i in data[0]:
        if not i == '\n':
            group_index.append(np.array(list(map(int, i.strip().split(',')))))
    return group_index

def partition_state(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space, tradeoff_para, sa_info):
    """TODO: Docstring for degree_partition.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/' + method + '_fixed_number_groups_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des += '/'
    elif method == 'degree':
        des += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    elif method == 'sa':
        N_trial, T_info = sa_info
        des += f'_N_trial={N_trial}_T_info={T_info}/'

    if not os.path.exists(des):
        os.makedirs(des)
    "the original network"
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    "the reduced system by different methods"
    if method == 'node_state':
        feature = xs_multi / xs_multi.max()
    elif method != 'sa':
        feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    for number_groups in number_groups_list:
        if not method == 'sa':
            group_index = group_index_from_feature_Kmeans(feature, number_groups)
        else:
            group_index = group_partition_sa(network_type, N, d, seed, number_groups, N_trial, T_info)
        A_reduction_deg_part, net_arguments_reduction_deg_part, x_eff_deg_part = reducednet_effstate(A, xs_multi, group_index)
        initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
        xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
        "save data"
        data=  np.hstack((weight, np.ravel(A_reduction_deg_part), x_eff_deg_part, xs_reduction_deg_part))
        des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={number_groups}_seed={seed}.csv'
        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None










seed1 = np.arange(10).tolist()
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
seed_ER = seed1

N = 1000
attractor_value = 0.1


















attractor_value = 0.1
weight_list = np.arange(0.01, 0.5, 0.01)
dynamics = 'BDP'
arguments = (B_BDP, )

attractor_value = 0.1
dynamics = 'PPI'
arguments = (B_PPI, F_PPI)
weight_list = np.arange(0.01, 0.5, 0.01)

attractor_value = 0.1
dynamics = 'SIS'
arguments = (B_SIS, )
weight_list = np.arange(0.01, 0.2, 0.01)

attractor_value = 0.01
dynamics = 'CW'
arguments = (a, b)
weight_list = np.arange(0.01, 1.2, 0.01)


attractor_value = 0.1
dynamics = 'genereg'
arguments = (B_gene,)
weight_list = np.arange(0.01, 0.3, 0.01)

attractor_value = 0.1
dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
weight_list = np.arange(0.01, 0.5, 0.01)


network_list = ['SF', 'SF', 'SBM_ER', 'SBM_ER']
N_list = [1000, 1000, [100, 100, 100], [100, 100]]
d_list = [[2.5, 999, 3], [2.1, 999, 2], np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist(), np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()]
seed_list = [[0, 0], [0, 0],  0, 0]
space_list = ['log', 'log', 'linear', 'linear']
number_groups_list_list = [[5], [5], [3], [2]]








tradeoff_para = 0.5
tradeoff_para_list = [0.3, 0.5, 0.7]
method_list = ['kcore_KNN_degree']
method_list = ['node_state', 'degree', 'kcore', 'kcore_degree']

network_list = ['SF', 'SBM_ER']
N_list = [1000, [100, 100, 100]]
d_list = [[2.5, 999, 3], np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()]
seed_list = [[0, 0], 0]
space_list = ['log', 'linear']
number_groups_list_list = [[3, 5, 10], [3, 5, 10]]

network_list = ['SF', 'SBM_ER']
N_list = [1000, [100, 100, 100]]
d_list = [[2.5, 999, 3], np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()]
seed_list = [[0, 0], 0]
space_list = ['log', 'linear']
number_groups_list_list = [[3, 5, 10], [3, 5, 10]]


network_list = ['SBM_ER']
N_list = [[100, 100, 100]]
d_list = [np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()]
seed_list = [0]
space_list = [ 'linear']
number_groups_list_list = [[20, 30]]






for method in method_list:
    for network_type, N, d, seed, space, number_groups_list in zip(network_list, N_list, d_list, seed_list, space_list, number_groups_list_list):
        for weight in weight_list:
            if method == 'degree' or method == 'kcore' or method == 'node_state':
                #partition_state(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space, tradeoff_para)
                pass
            else:
                for tradeoff_para in tradeoff_para_list:
                    #partition_state(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space, tradeoff_para)
                    pass


method = 'sa'
T0 = 1e-4
T_num = 40
T_decay = 0.9
N_trial = 100000
number_groups_list = [5]
T_info = [T0, T_num, T_decay]
sa_info = [N_trial, T_info]

network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
seed = 0

network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
seed = [0, 0]




for weight in weight_list:
    #partition_state(network_type, N, d, weight, seed, dynamics, arguments, attractor_value, number_groups_list, space, tradeoff_para, sa_info)
    pass

