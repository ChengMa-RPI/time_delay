import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')


import numpy as np 
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp
import time
from scipy.integrate import odeint
import scipy
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

fs = 24
ticksize = 20
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8




def extract_network_data(dynamics, network_type):
    """TODO: Docstring for extract_network_data.

    :dynamics: TODO
    :network_list: TODO
    :returns: TODO

    """
    N_list, d_list, seed_list = [], [], []
    des_bifurcation = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    des_multi = des_bifurcation + 'xs_multi/'
    des_file = os.listdir(des_multi)
    for file_i in des_file:
        if network_type == 'ER':
            N, d, seed = re.findall('\d+', file_i)
            N, d, seed = int(N), int(d), int(seed)
        elif network_type == 'SF':
            extract_number = re.findall('\d+', file_i)
            if len(extract_number) == 7:
                N, gamma_interger, gamma_decimal, kmax, kmin, seed1, seed2  = list(map(int, extract_number))
                gamma = gamma_interger + 10 ** (-len(str(gamma_decimal))) * gamma_decimal
            else:
                N, gamma, kmax, kmin, seed1, seed2  = list(map(int, extract_number))
            d = [gamma, kmax, kmin]
            seed = [seed1, seed2]
        N_list.append(N)
        d_list.append(d)
        seed_list.append(seed)
    return N_list, d_list, seed_list

def A_to_save(dynamics, network_type, N_list=None, d_list=None, seed_list=None):
    """TODO: Docstring for A_to_save.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """

    if not N_list and not d_list and not seed_list:
        N_list, d_list, seed_list = extract_network_data(dynamics, network_type) 
    save_des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    for N, d, seed in zip(N_list, d_list, seed_list):
        save_file = save_des + f'N={N}_d={d}_seed={seed}_A.npz'
        print(N, d, seed)
        if not os.path.exists(save_file):
            A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
            scipy.sparse.save_npz(save_file, scipy.sparse.csr_matrix(A) )
    return None
                

def network_critical_point(dynamics, network_type, N, seed, d, critical_type, threshold_value, survival_threshold, weight_list=None):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """

    file_des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/' +  f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_des).toarray()
    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    if weight_list:
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    else:
        weight_list = np.sort(np.unique(data[:, 0]))
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_multi = data[index, 1:]
    y_multi = betaspace(A, xs_multi)[-1]
    #y_multi = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])
    if critical_type == 'survival_ratio':
        transition_index = np.where(np.sum(xs_multi > survival_threshold, 1) / N > threshold_value) [0][0]
    else:
        transition_index = np.where(y_multi > threshold_value)[0][0]
    critical_weight = weight_list[transition_index]
    return y_multi, critical_weight

def group_critical_point(dynamics, network_type, N, seed, d, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold, weight_list=None):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """

    file_des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/' +  f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_des).toarray()
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, m)
    des_reduction = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans_space=' + space + '/'
    des_file = des_reduction + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
    if not os.path.exists(des_file):
        return None, None
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    if weight_list:
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    else:
        weight_list = np.sort(np.unique(data[:, 0]))
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]

    xs_reduction_multi = np.zeros((len(weight_list), N_actual))
    xs_i = data[index, 1:]
    for i, group_i in enumerate(group_index):
        xs_reduction_multi[:, group_i] = np.tile(xs_i[:, i], (len(group_i), 1)).transpose()

    y_reduction = betaspace(A, xs_reduction_multi)[-1]
    #y_multi = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])
    if critical_type == 'survival_ratio':
        index = np.where(np.sum(xs_reduction_multi > survival_threshold, 1) / N > threshold_value)[0]
        if len(index):
            transition_index = index[0]
        else:
            return None, None
    else:
        index = np.where(y_reduction > threshold_value)[0]
        if len(index):
            transition_index = index[0]
        else:
            return None, None
    critical_weight = weight_list[transition_index]
    return y_reduction, critical_weight

def critical_region_scatter(dynamics, network_type_list, m, hetero_type, critical_type, threshold_value, survival_threshold):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """

    cmaps = ['Blues', 'Reds', 'Greens', 'Purples']
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    delta_wc = []
    h1_list = []
    h2_list =[]
    kmean_list =[]
    kmin_list =[]
    for network_type in network_type_list:
        if network_type == 'SF':
            space = 'log'
        elif network_type == 'ER':
            space = 'linear'
        N_list, d_list, seed_list = extract_network_data(dynamics, network_type) 

        for N, d, seed in zip(N_list, d_list, seed_list):
            file_des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/' +  f'N={N}_d={d}_seed={seed}_A.npy'
            A = np.load(file_des)
            degrees = np.sum(A, 0)
            beta_cal = np.mean(degrees ** 2) / np.mean(degrees)
            kmean = np.mean(degrees)
            h1 = beta_cal - kmean
            h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / kmean

            y_reduction, wc_reduction = group_critical_point(network_type, N, seed, d, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold)
            if not y_reduction is None:
                y_multi, wc_multi = network_critical_point(network_type, N, seed, d, critical_type, threshold_value, survival_threshold)
                delta_wc.append( np.abs(wc_multi - wc_reduction) / wc_multi)
                h1_list.append(h1)
                h2_list.append( h2)
                kmean_list.append( kmean)
                kmin_list.append(d[-1])
                #delta_wc[i] =  wc_multi
    if hetero_type == 'h2':
        hetero_list = h2_list
        ylabel = '$h_2$'
    elif hetero_type == 'h1':
        hetero_list = h1_list
        ylabel = '$h_1$'

    xlabel = '$\\langle k \\rangle $'

    hetero_list = np.array(hetero_list)
    kmean_list = np.array(kmean_list)
    kmin_list = np.array(kmin_list)
    delta_wc = np.array(delta_wc)
    kmin_unique = np.sort(np.unique(kmin_list))
    for i, kmin_u in enumerate(kmin_unique):
        index = np.where(kmin_list == kmin_u)[0]
        kmean_u = kmean_list[index]
        hetero_u = hetero_list[index]
        wc_u = delta_wc[index]
        #plt.scatter(kmean_u, hetero_u, s=10 ** (wc_u) / np.max(10**delta_wc) * 50, c=wc_u, cmap=cmaps[i], vmin= delta_wc.min(), vmax=delta_wc.max(), label='$k_{min}=$' + f'{kmin_u}')
        plt.scatter(kmean_u, hetero_u, s=(wc_u/delta_wc.max()) * 50, c=wc_u, cmap=cmaps[i], vmin= delta_wc.min(), vmax=delta_wc.max(), label='$k_{min}=$' + f'{kmin_u}')
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    ax = plt.gca()
    legend = ax.get_legend()
    for i in range(len(kmin_unique)):
        legend.legendHandles[i].set_color(colors[i])
    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    #plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + hetero_type + '_' + critical_type + f'={threshold_value}.png'
    plt.savefig(save_des, format='png')
    plt.close()
    #plt.show()

    return kmean_list, hetero_list, delta_wc




                
#dynamics = 'genereg'
survival_threshold = 1e-3

dynamics = 'mutual'
network_type = 'SF'
N_list = [1000] * 10
d_list = [[3.8, 999, 5]] * 10
seed_list = [[i, i] for i in range(0, 10)]
network_type = 'ER'
N_list = [1000] * 30
d_list= [2000] * 10 + [4000] * 10 + [8000] * 10
seed_list = [i for i in range(10)] * 3
A_to_save(dynamics, network_type, N_list, d_list, seed_list)

space = 'log'
tradeoff_para = 0.5
method = 'degree'

#N_list, d_list, seed_list = extract_network_data(dynamics, network_type) 

threshold_value = 0.5
m = 1

hetero_type = 'h2'
network_type_list = ['SF']
critical_type = 'survival_ratio'
threshold_value_list = [0.1, 0.3, 0.5, 0.7, 0.9]

critical_type = 'y_gl'
threshold_value_list = [0.1, 0.3, 0.5, 0.7, 0.9]
m = 1
for threshold_value in threshold_value_list:
    #critical_region_scatter(network_type_list, m, hetero_type, critical_type, threshold_value, survival_threshold)
    pass
