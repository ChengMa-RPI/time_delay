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
from scipy.optimize import fsolve
import scipy
import time

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
        des_file = des_save + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
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


def helper_stable(x1, x2, tol=1e-6):
    """TODO: Docstring for helper_stable.

    :xs: TODO
    :tol: TODO
    :returns: TODO

    """
    diff = np.abs(x1 - x2).max() 
    return diff<tol

def helper_wc_binary_search(dynamics_func, net_arguments, arguments, attractor_value, wl, wr, xc, R, des_file, tol=1e-1):
    """TODO: Docstring for helper_wc_binary_search.
    :returns: TODO

    """
    xl = xs_w(dynamics_func, net_arguments, wl, arguments, attractor_value, des_file)
    xr = xs_w(dynamics_func, net_arguments, wr, arguments, attractor_value, des_file)
    while np.sum(xl > xc) > R:
        wl /= 2
        xl = xs_w(dynamics_func, net_arguments, wl, arguments, attractor_value, des_file)
    while np.sum(xr > xc) <= R :
        wr *= 2
        xr = xs_w(dynamics_func, net_arguments, wr, arguments, attractor_value, des_file)
    while wr - wl > tol * (wr + wl):
        wm = (wr + wl)/ 2
        xm = xs_w(dynamics_func, net_arguments, wm, arguments, attractor_value, des_file)
        if np.sum(xm > xc) <= R:
            wl = wm
        else:
            wr = wm
    return wl, wr


def xs_w(dynamics_func, net_arguments, weight, arguments, attractor_value, des_file):
    """TODO: Docstring for wc_m_1.

    :dynamics: TODO
    :arguments: TODO
    :returns: TODO

    """
    net_arguments = (net_arguments[0], net_arguments[1], net_arguments[2] * weight, net_arguments[3])
    N = np.max(net_arguments[0]) + 1
    T = 1000
    t_increment = 50
    dt = 0.01
    t_start = 0
    initial_condition = np.ones(N) * attractor_value
    stable = False
    while stable == False:
        t = np.arange(t_start, t_start + t_increment + dt/2, dt)
        xs = odeint(dynamics_func, initial_condition, t,  args=(arguments, net_arguments))[-1]
        stable = helper_stable(initial_condition, xs)
        initial_condition = xs
        t_start += t_increment

    data = np.hstack((weight, xs))
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return  xs

def xs_rdw(network_type, N, d, seed, dynamics, arguments, attractor_value, wl, wr, xc1, xc2, m, space, des_file):
    """TODO: Docstring for wc_find.

    :dynamics: TODO
    :arguments: TODO
    :wl: TODO
    :wr: TODO
    :attractor_value: TODO
    :: TODO
    :returns: TODO

    """
    if 'high' in dynamics:
        dynamics_func = globals()[dynamics[:dynamics.find('_high')] + '_multi']
    else:
        dynamics_func = globals()[dynamics + '_multi']
    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A_unit = scipy.sparse.load_npz(file_A).toarray()
    A_index = np.where(A_unit>0)
    A_interaction = A_unit[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A_unit>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    if not m == N:
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        group_index = group_index_from_feature_Kmeans(feature, m)
        A_reduction_deg_part, net_arguments, _ = reducednet_effstate(A_unit, np.zeros(len(A_unit)), group_index)

    wc_R1, _ = helper_wc_binary_search(dynamics_func, net_arguments, arguments, attractor_value, wl, wr, xc1, 0, des_file)
    _, wc_R2 = helper_wc_binary_search(dynamics_func, net_arguments, arguments, attractor_value, wc_R1, wr, xc2, m-1, des_file)
    weight_list = []
    for i in range(500):
        w_cand = wc_R1 * 1.01 ** i 
        if w_cand <= wc_R2:
            weight_list.append(w_cand)
        else:
            break
    weight_list = np.round(weight_list, 5)
    if i == 499:
        print('range too large', wc_R1, wc_R2)
        weight_list = np.round(np.linspace(wc_R1, wc_R2, 500), 5)

    for weight in weight_list:
        xs_w(dynamics_func, net_arguments, weight, arguments, attractor_value, des_file)
    return None

def xs_rdw_parallel(dynamics, arguments, network_type, N, seed_list, d_list, m_list, attractor_value, wl, wr, xc1, xc2, space):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi_rdw=0.01/' 
    des_group = '../data/' + dynamics + '/' + network_type + f'/xs_bifurcation/degree_kmeans_space={space}_rdw=0.01/' 
    for des in [des_multi, des_group]:
        if not os.path.exists(des):
            os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(xs_rdw, [(network_type, N, d, seed, dynamics, arguments, attractor_value, wl, wr, xc1, xc2, m, space, [des_group + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv', des_multi + f'N={N}_d={d}_seed={seed}.csv'][int(m==N)]) for seed, d in zip(seed_list, d_list) for m in m_list]) .get()
    p.close()
    p.join()
    return None

def xs_beta(network_type, N, d, seed, dynamics, arguments, attractor_value, beta_list, m, space, des_file):
    """TODO: Docstring for wc_find.

    :dynamics: TODO
    :arguments: TODO
    :wl: TODO
    :wr: TODO
    :attractor_value: TODO
    :: TODO
    :returns: TODO

    """
    if 'high' in dynamics:
        dynamics_func = globals()[dynamics[:dynamics.find('_high')] + '_multi']
    else:
        dynamics_func = globals()[dynamics + '_multi']
    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A_unit = scipy.sparse.load_npz(file_A).toarray()
    A_index = np.where(A_unit>0)
    A_interaction = A_unit[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A_unit>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    beta_cal = betaspace(A_unit, [0] )[0]
    weight_list = beta_list / beta_cal 
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    if not m == N:
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        group_index = group_index_from_feature_Kmeans(feature, m)
        A_reduction_deg_part, net_arguments, _ = reducednet_effstate(A_unit, np.zeros(len(A_unit)), group_index)
    for weight in weight_list:
        xs_w(dynamics_func, net_arguments, weight, arguments, attractor_value, des_file)
    return None

def xs_beta_parallel(dynamics, arguments, network_type, N, seed_list, d_list, m_list, attractor_value, beta_list, space):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi_beta/' 
    des_group = '../data/' + dynamics + '/' + network_type + f'/xs_bifurcation/degree_kmeans_space={space}_beta/' 
    for des in [des_multi, des_group]:
        if not os.path.exists(des):
            os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(xs_beta, [(network_type, N, d, seed, dynamics, arguments, attractor_value, beta_list, m, space, [des_group + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv', des_multi + f'N={N}_d={d}_seed={seed}.csv'][int(m==N)]) for seed, d in zip(seed_list, d_list) for m in m_list]) .get()
    p.close()
    p.join()
    return None


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


m_list = np.arange(1, 11, 1)
weight_list = np.round(np.arange(0.01, 1.01, 0.01), 5)
tradeoff_para = 1
method = 'degree'


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1

network_type = 'SF'
space = 'log'
N = 1000
seed_list = [[i, i] for i in range(1, 10)]
d_list = [[3, 999, 4]] * 9
weight_list = np.round( np.arange(0.01, 0.6, 0.01), 5)
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )

 


network_type = 'ER'
space = 'linear'
N = 100
seed_list = [0]
d_list = [1600]
weight_list = np.round( np.arange(0.01, 1.001, 0.01), 5)
m_list = np.arange(1, 51, 1)

network_type = 'SBM_ER'
space = 'linear'
N = [33, 33, 34]
d_list = [np.array([[0.9, 0.001, 0.001], [0.001, 0.5, 0.001], [0.001, 0.001, 0.05]]).tolist()]
seed_list = [0]
weight_list = np.round( np.arange(0.01, 1.001, 0.01), 5)
m_list = np.arange(1, 51, 1)

network_type = 'SF'
space = 'log'
N = 100
seed_list = [[15, 15]]
d_list = [[2.1, N-1, 1]]
weight_list = np.round( np.arange(0.01, 0.61, 0.01), 5)
m_list = np.arange(1, N, 1)




#xs_group_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, m_list, attractor_value, space, tradeoff_para, method)

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1

dynamics = 'CW'
arguments = (a, b)
attractor_value = 100

network_type = 'SF'
space = 'log'
N = 1000
seed_list = [[i, i] for i in range(0, 10)]  * 2
d_list = [[2.5, 999, 3]] * 10 + [[3, 999, 4]] * 10
network_type = 'ER'
space = 'ER'
N = 1000
seed_list = [i for i in range(10)] * 2
d_list = [4000] * 10 + [8000] * 10
#weight_list = np.round( np.arange(0.01, 0.3, 0.05), 5)
weight_list = [0.6]

xs_multi_parallel(dynamics, arguments, network_type, N, seed_list, d_list, weight_list, attractor_value)


network_type = 'SF'
space = 'log'
N = 1000
gamma_list = [round(i, 1) if i%1 < 1e-5 else int(i) for i in np.arange(2.1, 5.01, 0.1)]
kmin_list = [3, 4, 5]

d_list = sum([[[gamma, N-1, kmin]] * 10 for gamma in gamma_list for kmin in kmin_list], [])
seed_list = [[i, i] for i in range(0, 10)] * len(gamma_list) * len(kmin_list)





dynamics = 'CW'
arguments = (a, b)
attractor_value = 0
wl, wr, xc1, xc2 = 0.01, 10, 1, 1

dynamics = 'genereg'
arguments = (B_gene, )
attractor_value = 100
wl, wr, xc1, xc2 = 0.01, 1, 1, 1

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1
wl, wr, xc1, xc2 = 0.01, 1, 5, 5


m_list = [1, 5, 10, N]
#xs_rdw_parallel(dynamics, arguments, network_type, N, seed_list, d_list, m_list, attractor_value, wl, wr, xc1, xc2, space)

beta_list = [1, 2, 3, 4, 5, 6, 7]
#xs_beta_parallel(dynamics, arguments, network_type, N, seed_list, d_list, m_list, attractor_value, beta_list, space)
