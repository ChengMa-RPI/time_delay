import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import network_generate

import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import itertools
import time 


fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-', '--']))




def data_plot_resilience_function(des_file, xaxis, save_des):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    data = np.array(pd.read_csv(des_file, header=None).iloc[:50, :])
    length_groups = int(np.sqrt(np.size(data, 1) ) - 1)
    weight = data[:, 0]
    A_reduction = data[:, 1: length_groups**2+1]
    A_reduction_diagonal = np.vstack(([A_reduction[:, i*length_groups+i] for i in range(length_groups)])).transpose()
    x_eff = data[:, length_groups**2+1:length_groups**2+length_groups+1]
    xs_reduction = data[:, length_groups**2+length_groups+1:length_groups**2+2*length_groups+1]

    if xaxis == 'w' or xaxis == 'A_reduction':
        y_theory = xs_reduction
        y_simulation = x_eff
        if xaxis == 'w':
            x = np.tile(weight, (np.size(y_theory, 1), 1)).transpose()
            xlabel ='$w$'
        elif xaxis == 'A_reduction':
            xlabel = "$A'_{\\mathrm{diag}}$"
            x = A_reduction_diagonal
        for i in range(np.size(y_theory, 1)):
            #plt.plot(x[:, i], y_theory[:, i], '-', linewidth=lw, label='theory' + str(i+1))
            #plt.plot(x[:, i], y_simulation[:, i], '--', linewidth=lw, label='simulation' + str(i+1))
            plt.plot(x[:, i], y_theory[:, i], '-', linewidth=lw, label='theory')
            plt.plot(x[:, i], y_simulation[:, i], '--', linewidth=lw, label='simulation')
    elif xaxis == 'beta_reduction':
        interaction_in_out_list = []
        beta_reduction = []
        x_eff_reduction = []
        xs_reduction_reduction = []
        for j, i in enumerate(A_reduction):
            A_reduction_one_w = i.reshape(length_groups, length_groups)
            k_out_reduction = np.sum(A_reduction_one_w, 0)
            k_in_reduction = np.sum(A_reduction_one_w, 1)
            interaction_in_out = np.sum(A_reduction_one_w * k_out_reduction.reshape(len(k_out_reduction), 1), 1) / np.sum(k_out_reduction)
            interaction_in_out_list.append(interaction_in_out)
            beta_reduction.append(np.sum(k_in_reduction * k_out_reduction) / np.sum(k_out_reduction)) 
            x_eff_reduction.append(np.sum(x_eff[j] * k_out_reduction) / np.sum(k_out_reduction))
            xs_reduction_reduction.append(np.sum(xs_reduction[j] * k_out_reduction) / np.sum(k_out_reduction))
        interaction_in_out_list = np.vstack((interaction_in_out_list))
        xlabel = '$\\beta$'
        x = beta_reduction
        plt.plot(beta_reduction, xs_reduction_reduction, '-', linewidth=lw, label='theory')
        plt.plot(beta_reduction, x_eff_reduction, '--', linewidth=lw, label='simulation')

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.75, wspace=0.25, hspace=0.25, bottom=0.18, top=0.80)
    plt.locator_params('x', nbins=5)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$x_s$', fontsize=fs)
    if xaxis == 'beta_reduction':
        plt.legend(fontsize=13, frameon=False) 
    else:
        #plt.legend(fontsize=13, frameon=False, loc=1, bbox_to_anchor=(1.5,1.2) )
        plt.legend(fontsize=13, frameon=False) 
    plt.savefig(save_des)
    plt.close()
    #plt.show()
    return None

def xs_resilience_partition_method(network_type, N, d, seed, dynamics, number_groups, xaxis, space, method, sa_info):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    #des = '../data/' + dynamics + '/' + network_type + '/' + method + '_fixed_number_groups_kmeans' 
    des = '../data/' + dynamics + '/' + network_type + '/' + method + '_fixed_number_groups' 
    save_des = '../report/report102021/' + dynamics + '_' + network_type + '_' + method + '_fixed_number_groups_kmeans'
    if method == 'kcore' or method == 'node_state':
        des += '/'
    elif method == 'degree':
        des += '_space=' + space + '/'
        save_des += '_space=' + space
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
        save_des += '_space=' + space + f'_tradeoffpara={tradeoff_para}'
    elif method == 'sa':
        N_trial, T_info = sa_info
        des += f'_N_trial={N_trial}_T_info={T_info}/'
        save_des += f'_N_trial={N_trial}_T_info={T_info}'



    des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={number_groups}_seed={seed}.csv'
    save_file = save_des + f'_N={N}_d=' + str(d) + f'_number_groups={number_groups}_seed={seed}_' + xaxis + '.png'

    data_plot_resilience_function(des_file, xaxis, save_file)
    return None





dynamics = 'genereg'
dynamics = 'CW'
dynamics = 'BDP'
dynamics = 'SIS'
dynamics = 'PPI'
dynamics = 'mutual'

seed = 0
N_group = [100, 100]
p = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
number_groups = 2

N_group = [100, 100, 100]
p = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
number_groups = 5

xaxis='w'
xaxis='A_reduction'
seed = 0 
number_groups = 3
N_group = [100, 100, 100]
p = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()


network_type = 'SBM_ER'
N = [100, 100]
d = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
seed = 0
regroup = [[0], [1], [2, 3]]
regroup = 'None'

network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
seed = 0
regroup = [[0, 1, 2, 3, 4]]
regroup = [[0], [1], [2, 3, 4]]
regroup = 'None'






network_type = 'SF'
N = 1000
d = [2.1, 999, 3]
seed =[0, 0]
regroup = 'None'
regroup_criteria = 'None'
number_groups = 5
space = 'log'
degree_interval = 1.5

network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
N = [100, 100]
d = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
number_groups = 5
space = 'linear'
degree_interval = 20
seed = 0

number_groups_list = [1, 5]


network_list = ['SBM_ER']
N_list = [[100, 100, 100]]
d_list = [ np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()]
seed_list = [0]
space_list = ['linear']
number_groups_list_list = [[20, 30]]

network_list = ['SF', 'SBM_ER']
N_list = [1000, [100, 100, 100]]
d_list = [[2.5, 999, 3], np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()]
seed_list = [[0, 0], 0]
space_list = ['log', 'linear']
number_groups_list_list = [[3, 5, 10], [3, 5, 10]]



xaxis_list = ['w', 'beta_reduction']
tradeoff_para_list = [0.3, 0.5, 0.7]
method_list = ['degree', 'node_state', 'kcore', 'kcore_degree', 'kcore_KNN_degree']

        
for method in method_list:
    for tradeoff_para in tradeoff_para_list:
        for xaxis in xaxis_list:
            for network_type, N, d, seed, space, number_groups_list in zip(network_list, N_list, d_list, seed_list, space_list, number_groups_list_list):
                for number_groups in number_groups_list:
                    #xs_resilience_partition_method(network_type, N, d, seed, dynamics, number_groups, xaxis, space, method)
                    pass

method = 'sa'
method = 'kcore'
T0 = 1e-4
T_num = 40
T_decay = 0.9
N_trial = 100000
number_groups = 1
T_info = [T0, T_num, T_decay]
sa_info = [N_trial, T_info]




network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
seed = [0, 0]

network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
seed = 0


xaxis_list = ['w', 'beta_reduction']

for xaxis in xaxis_list:
    xs_resilience_partition_method(network_type, N, d, seed, dynamics, number_groups, xaxis, space, method, sa_info)
