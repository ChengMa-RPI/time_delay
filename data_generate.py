import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp
from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

cpu_number = 8

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


fs = 24
ticksize = 20
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))



def xs_group_partition_bifurcation(dynamics, arguments, network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, des_save):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    dynamics_multi = globals()[dynamics + '_multi']
    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
        for j, m in enumerate(m_list):
            group_index = group_index_from_feature_Kmeans(feature, m)
            A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(N_actual), group_index)
            initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
            xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
            data = np.hstack((weight, xs_reduction_deg_part))
            des_file = des_save + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
            df = pd.DataFrame(data.reshape(1, len(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def xs_multi_bifurcation(dynamics, arguments, network_type, N, seed, d, weight_list, attractor_value, des_save):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        des_file = des + f'N={N}_d=' + str(d_record) + f'_seed={seed}.csv'
        data = np.hstack((weight, xs_multi))
        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def xs_group_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, m_list, attractor_value, space, tradeoff_para, method):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des_save = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des_save += '/'
    elif method == 'degree':
        des_save += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des_save += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    if not os.path.exists(des_save):
        os.makedirs(des_save)

    p = mp.Pool(cpu_number)
    p.starmap_async(xs_group_partition_bifurcation, [(dynamics, arguments,  network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, des_save) for seed, d in zip(seed_list, d_list) ]) .get()
    p.close()
    p.join()
    return None

def xs_multi_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, attractor_value):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des_save = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/' 
    if not os.path.exists(des_save):
        os.makedirs(des_save)
    p = mp.Pool(cpu_number)
    p.starmap_async(xs_multi_bifurcation, [(dynamics, arguments, network_type, N, seed, d, weight_list, attractor_value, des_save) for seed, d in zip(seed_list, d_list) ]) .get()
    p.close()
    p.join()
    return None





dynamics = 'genereg'
arguments = (B_gene, )
attractor_value = 20


dynamics = 'CW'
arguments = (a, b)
attractor_value = 0

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1


network_type = 'real'

N = 1044
seed_list = [12]
d_list = [0]
space = 'log'

N = 456
seed_list = [11]
d_list = [0]
space = 'linear'

N = 97
seed_list = [8]
d_list = [0]
space = 'log'


N = 270
seed_list = [6]
d_list = [0]
space = 'log'

N = 91
seed_list = [5]
d_list = [0]
space = 'log'
space = 'linear'



beta = 1
betaeffect = 0 











network_type = 'ER'
N = 1000
d = 2000
seed = 0
d_list = [4000, 8000]
seed_list = [0, 1]
space = 'linear'


network_type = 'SBM_ER'
N = [100, 100, 100]
d_list = [np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()]
d_list = [np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist()]
d_list = [np.array([[0.2, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.05]]).tolist(), np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist(), np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()]
seed = 0
seed_list = [0, 1]
space = 'linear'


network_type = 'star'
space = 'log'
N = 50
seed_list = [0]
d_list = [0]

network_type = 'degree_seq'
space = 'linear'
N = 100
seed_list = [0]
degree_sequence = np.random.randint(1, 3, N)
degree_sequence[0] = np.random.randint(50, 90, 1)
degree_sequence[1:5] = np.random.randint(4, 10, 4)
d_list = [degree_sequence]

network_type = 'SF'
space = 'log'
N = 1000
gamma_list = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4]
kmin_list = [3, 4, 5]

d_list = sum([[[gamma, N-1, kmin]] * 50 for gamma in gamma_list for kmin in kmin_list], [])
seed_list = [[i, i] for i in range(0, 50)] * 30


m_list = np.arange(1, 11, 1)
weight_list = np.round(np.arange(0.01, 1.01, 0.01), 5)
tradeoff_para = 0.5
method = 'degree'



xs_group_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, m_list, attractor_value, space, tradeoff_para, method)

xs_multi_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, attractor_value)
