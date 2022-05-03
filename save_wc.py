import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

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
import scipy.stats as stats
import scipy

cpu_number = 4

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


def save_A(network_type, N, d, seed, save_des):
    """TODO: Docstring for A_to_save.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """
    save_file = save_des + f'N={N}_d={d}_seed={seed}_A.npz'
    if not os.path.exists(save_file):
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        scipy.sparse.save_npz(save_file, scipy.sparse.csr_matrix(A) )
    return None

def save_A_parallel(network_type, N, d_list, seed_list):
    """TODO: Docstring for A_to_save.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """
    save_des = '../data/A_matrix/' + network_type + '/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    p = mp.Pool(cpu_number)
    p.starmap_async(save_A, [(network_type, N, d, seed, save_des) for d, seed in zip(d_list, seed_list)]).get()
    p.close()
    p.join()
    return None
                
def network_critical_point(dynamics, network_type, N, seed, d, critical_type, threshold_value, survival_threshold, wc_file, weight_list=None):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_A).toarray()
    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d={d}_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    if not weight_list:
        weight_list = np.sort(np.unique(data[:, 0]))
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_multi = data[index, 1:]
    y_multi = betaspace(A, xs_multi)[-1]
    if critical_type == 'survival_ratio':
        index = np.where(np.sum(xs_multi > survival_threshold, 1) / N > threshold_value) [0]
    else:
        index = np.where(y_multi > threshold_value)[0]
    if len(index):
        critical_weight = weight_list[index[0]]
    else:
        critical_weight = None
    df = pd.DataFrame(np.array([d, seed, critical_weight], dtype='object').reshape(1, 3))
    df.to_csv(wc_file, index=None, header=None, mode='a')
    return None

def network_wc_parallel(dynamics, network_type, N, seed_list, d_list, critical_type, threshold_value, survival_threshold):
    """TODO: Docstring for network_wc_parallel.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :seed_list: TODO
    :d_list: TODO
    :critical_type: TODO
    :threshold_value: TODO
    :survival_threshold: TODO
    :returns: TODO

    """
    save_des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/wc_multi/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    wc_file = save_des + 'critical_type=' + critical_type + f'_threshold_value={threshold_value}.csv'
    p = mp.Pool(cpu_number)
    p.starmap_async(network_critical_point, [(dynamics, network_type, N, seed, d, critical_type, threshold_value, survival_threshold, wc_file) for seed, d in zip(seed_list, d_list)]).get()
    p.close()
    p.join()
    return None

def group_critical_point(dynamics, network_type, N, seed, d, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold, wc_file, weight_list=None):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_A).toarray()
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, m)
    des_reduction = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans_space=' + space + '/'
    des_file = des_reduction + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    if not weight_list:
        weight_list = np.sort(np.unique(data[:, 0]))
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_reduction_multi = np.zeros((len(weight_list), N_actual))
    xs_i = data[index, 1:]
    for i, group_i in enumerate(group_index):
        xs_reduction_multi[:, group_i] = np.tile(xs_i[:, i], (len(group_i), 1)).transpose()
    y_reduction = betaspace(A, xs_reduction_multi)[-1]
    if critical_type == 'survival_ratio':
        index = np.where(np.sum(xs_reduction_multi > survival_threshold, 1) / N > threshold_value)[0]
    else:
        index = np.where(y_reduction > threshold_value)[0]
    if len(index):
        critical_weight = weight_list[index[0]]
    else:
        critical_weight = None
    df = pd.DataFrame(np.array([d, seed, critical_weight], dtype='object').reshape(1, 3))
    df.to_csv(wc_file, index=None, header=None, mode='a')
    return None

def group_wc_parallel(dynamics, network_type, N, seed_list, d_list, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold):
    """TODO: Docstring for network_wc_parallel.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :seed_list: TODO
    :d_list: TODO
    :critical_type: TODO
    :threshold_value: TODO
    :survival_threshold: TODO
    :returns: TODO

    """
    save_des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/wc_group/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    wc_file = save_des + 'critical_type=' + critical_type + f'_threshold_value={threshold_value}_m={m}.csv'
    p = mp.Pool(cpu_number)
    p.starmap_async(group_critical_point, [(dynamics, network_type, N, seed, d, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold, wc_file) for seed, d in zip(seed_list, d_list)]).get()
    p.close()
    p.join()
    return None






dynamics = 'mutual'
network_type = 'SF'
N = 1000
gamma_list = [round(i, 1) if i%1 > 1e-5 else int(i) for i in np.arange(2.1, 5.1, 0.1)]
d_list = sum([[[gamma, N-1, kmin]] * 50 for gamma in gamma_list for kmin in [3, 4, 5]], [])
seed_list = [[i, i] for i in range(50)] * 90
save_A_parallel(network_type, N, d_list, seed_list)

critical_type = 'ygl'
threshold_value = 2.5
survival_threshold = 5
#network_wc_parallel(dynamics, network_type, N, seed_list, d_list, critical_type, threshold_value, survival_threshold)
tradeoff_para = 0.5
method = 'degree'
space = 'log'
m = 5

#group_wc_parallel(dynamics, network_type, N, seed_list, d_list, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold)
