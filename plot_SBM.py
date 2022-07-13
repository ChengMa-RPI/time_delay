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



def data_read(N_group, p, seed, dynamics, number_groups):
    """TODO: Docstring for compare_xs_multi_group.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :group_num: TODO
    :seed: TODO
    :returns: TODO

    """
    network_type = 'SBM_ER'
    des = '../data/' + dynamics + '/' + network_type + f'/xs_compare_multi_community_spectral/'
    des_file = des + f'N={N_group}_p=' + str(p) + f'_group_num={number_groups}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:70, :])
    weight = data[:, 0]
    A_reduction = data[:, 1: number_groups**2+1]
    A_reduction_diagonal = np.vstack(([A_reduction[:, i*number_groups+i] for i in range(number_groups)])).transpose()
    x_eff = data[:, number_groups**2+1: number_groups**2+number_groups+1]
    xs_reduction = data[:, number_groups**2+number_groups+1: number_groups**2+number_groups*2+1]
    alpha_list = data[:, number_groups**2+number_groups*2+1: number_groups**2+number_groups*3+1]
    beta_list = data[:, number_groups**2+number_groups*3+1: number_groups**2+number_groups*4+1]
    R_list = data[:, number_groups**2+number_groups*4+1: number_groups**2+number_groups*5+1]
    xs_spectral = data[:, number_groups**2+number_groups*5+1: number_groups**2+number_groups*6+1]

    beta_1D = data[:, number_groups**2+number_groups*6+1: number_groups**2+number_groups*6+2]
    x_eff_1D = data[:, number_groups**2+number_groups*6+2: number_groups**2+number_groups*6+3]
    xs_reduction_1D = data[:, number_groups**2+number_groups*6+3: number_groups**2+number_groups*6+4]
    return weight, A_reduction, A_reduction_diagonal, x_eff, xs_reduction, alpha_list, beta_list, R_list, xs_spectral, beta_1D, x_eff_1D, xs_reduction_1D

def xs_resilience_community_spectral(N_group, p, seed, dynamics, number_groups, reduction_method, xaxis):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    weight, A_reduction, A_reduction_diagonal, x_eff, xs_reduction, alpha_list, beta_list, R_list, xs_spectral, beta_1D, x_eff_1D, xs_reduction_1D = data_read(N_group, p, seed, dynamics, number_groups)
    if reduction_method == 'reduction_community':
        y_theory = xs_reduction
        y_simulation = x_eff
        reduction_parameter = A_reduction_diagonal
        xlabel_A_reduction = "$A'_{\\mathrm{diag}}$"
        xlabel_A_reduction = "$A'_{\\mathrm{diag}}$"
    elif reduction_method == 'reduction_spectral':
        y_theory = xs_spectral
        y_simulation = R_list
        reduction_parameter = alpha_list
        xlabel_A_reduction = '$\\alpha$'
    elif reduction_method == 'degree_weighted_1D':
        y_theory = xs_reduction_1D
        y_simulation = x_eff_1D
        reduction_parameter = beta_1D
        xlabel_A_reduction = '$\\beta$'

    if xaxis == 'w':
        x = np.tile(weight, (np.size(y_theory, 1), 1)).transpose()
        xlabel ='$w$'
    elif xaxis == 'A_reduction':
        xlabel = xlabel_A_reduction
        x = reduction_parameter
    for i in range(np.size(y_theory, 1)):
        plt.plot(x[:, i], y_theory[:, i], '-', linewidth=lw, label='theory' + str(i+1))
        plt.plot(x[:, i], y_simulation[:, i], '--', linewidth=lw, label='simulation' + str(i+1))

    #plt.plot(A_reduction_deg_part_diagonal, xs_reduction_deg_part, 'b')
    #plt.plot(A_reduction_deg_part_diagonal, x_eff_deg_part, 'red')

    #plt.plot(weight, xs_spectral, 'b')
    #plt.plot(weight, R_list, 'red')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.subplots_adjust(left=0.2, right=0.75, wspace=0.25, hspace=0.25, bottom=0.18, top=0.80)
    plt.locator_params('x', nbins=5)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$x_s$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc=1, bbox_to_anchor=(1.52,1.0) )
    plt.savefig('../report/report082621/' + dynamics + f'_N_={N_group}_p={p}_seed={seed}_' + reduction_method + '_' + xaxis + '.png')
    plt.close('all')
    #plt.show()
    return None


def data_plot_resilience_function(des_file, xaxis, save_des):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    data = np.array(pd.read_csv(des_file, header=None).iloc[:120, :])
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
            plt.plot(x[:, i], y_theory[:, i], '-', linewidth=lw, label='theory' + str(i+1))
            plt.plot(x[:, i], y_simulation[:, i], '--', linewidth=lw, label='simulation' + str(i+1))
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
        plt.legend(fontsize=13, frameon=False, loc=1, bbox_to_anchor=(1.5,1.2) )
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()
    return None

def xs_resilience_degree_partition(network_type, N, d, seed, dynamics, number_groups, xaxis, space):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + f'/bifurcation_group_decouple_' + space + '/'
    des_file = des + f'N={N}_d=' + str(d) + f'_group_num={number_groups}_seed={seed}.csv'

    save_des = '../report/report092221/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_degree_' + space + f'_number_groups={number_groups}_' + xaxis + '.png'
    data_plot_resilience_function(des_file, xaxis, save_des)
    return None

def xs_resilience_KNN_kcore_regroup_partition(network_type, N, d, seed, dynamics, xaxis, regroup):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    if regroup == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
        save_des = '../report/report092221/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_KNN_kcore_' + xaxis + '.png'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_regroup/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup={regroup}.csv'
        save_des = '../report/report091521/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_KNN_kcore_regroup={regroup}_' + xaxis + '.png'

    data_plot_resilience_function(des_file, xaxis, save_des)
    return None

def xs_resilience_KNN_kcore_degree_partition(network_type, N, d, seed, dynamics, xaxis, regroup_criteria, space, degree_interval):
    """TODO: Docstring for xs_resilience_KNN_kcore_degree_parallel_partition.

    :arg1: TODO
    :returns: TODO

    """
    if regroup_criteria == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_degree_parallel_' + space + f'={degree_interval}/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
        save_des = '../report/report092221/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_KNN_kcore_' + space + f'={degree_interval}_parallel_' + xaxis + '.png'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_KNN_kcore_degree_regroup_criteria_' + space + f'={degree_interval}/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup_criteria={regroup_criteria}.csv'
        save_des = '../report/report091521/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_KNN_kcore_' + space + f'={degree_interval}_regroup_criteria={regroup_criteria}_' + xaxis + '.png'
    data_plot_resilience_function(des_file, xaxis, save_des)
    return None

def xs_resilience_kcore(network_type, N, d, seed, dynamics, xaxis, regroup):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    if regroup == 'None':
        des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
        save_des = '../report/report092221/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_kcore_' + xaxis + '.png'
    else:
        des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore_regroup/'
        des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}_regroup={regroup}.csv'
        save_des = '../report/report091521/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_kcore_regroup={regroup}_' + xaxis + '.png'

    data_plot_resilience_function(des_file, xaxis, save_des)

    return None

def xs_resilience_kcore_degree_partition(network_type, N, d, seed, dynamics, xaxis, space, degree_interval):
    """TODO: Docstring for xs_resilience_KNN_kcore_degree_parallel_partition.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + f'/xs_kcore_degree_' + space + f'={degree_interval}/'
    des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    save_des = '../report/report092921/' + dynamics + '_' + network_type + f'_N_={N}_d={d}_seed={seed}_kcore_degree_' + space + f'={degree_interval}_' + xaxis + '.png'
    data_plot_resilience_function(des_file, xaxis, save_des)
    return None

def wc_k(network_type, N, d, seed, dynamics, xaxis):
    """TODO: Docstring for xs_resilience.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_multi/'
    des_file = des + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    weight = data[2:, 0]
    k = data[0, 1:]
    core_number = data[1, 1:]
    xs_multi_list = data[2:, 1:]
    wc_list = []
    for xs_multi in xs_multi_list.transpose():
        xs_multi_diff = np.diff(np.diff(xs_multi))
        index = np.argmax(xs_multi_diff)
        wc = weight[index]
        wc_list.append(wc)

    if xaxis == 'k':
        xlabel ='$k$'
        plt.semilogx(k, wc_list, '.')
    elif xaxis == 'core_number':
        xlabel = 'core'
        plt.plot(core_number, wc_list, '.')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.75, wspace=0.25, hspace=0.25, bottom=0.18, top=0.80)
    #plt.locator_params('x', nbins=5)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$w_c$', fontsize=fs)
    plt.legend(fontsize=13, frameon=False, loc=1, bbox_to_anchor=(1.5,1.2) )
    #plt.savefig('../report/report090821/' + dynamics + '_' + network_type + f'_N_={N}_p={d}_seed={seed}_' + 'KNN' + f'_parallel_degree_interval={degree_interval}_regroup_criteria={regroup_criteria}_' + xaxis + '.png')
    #plt.close('all')
    #plt.show()
    return None





dynamics = 'mutual'
dynamics = 'SIS'
dynamics = 'PPI'
dynamics = 'BDP'
dynamics = 'genereg'
dynamics = 'CW'

seed = 0
N_group = [100, 100]
p = np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()
number_groups = 2

N_group = [100, 100, 100]
p = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
number_groups = 5


reduction_method_list = ['reduction_community', 'reduction_spectral', 'degree_weighted_1D']
xaxis_list = ['A_reduction', 'w']
for reduction_method in reduction_method_list:
    for xaxis in xaxis_list:
        #xs_resilience_community_spectral(N_group, p, seed, dynamics, number_groups, reduction_method, xaxis)
        pass

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

network_list = ['SF', 'SF', 'SBM_ER', 'SBM_ER']
N_list = [1000, 1000, [100, 100, 100], [100, 100]]
d_list = [[2.5, 999, 3], [2.1, 999, 2], np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist(), np.array([[0.5, 0.001], [0.001, 0.1]]).tolist()]
seed_list = [[0, 0], [0, 0],  0, 0]
degree_interval_list = [1.5, 1.5, 20, 20]
space_list = ['log', 'log', 'linear', 'linear']
number_groups_list_list = [[5], [5], [3], [2]]



xaxis_list = ['w', 'beta_reduction']


for xaxis in xaxis_list:
    for network_type, N, d, seed, space, degree_interval, number_groups_list in zip(network_list, N_list, d_list, seed_list, space_list, degree_interval_list, number_groups_list_list):
        for number_groups in number_groups_list:
            xs_resilience_degree_partition(network_type, N, d, seed, dynamics, number_groups, xaxis, space)
            pass
        #xs_resilience_KNN_kcore_regroup_partition(network_type, N, d, seed, dynamics, xaxis, regroup)
        #xs_resilience_KNN_kcore_degree_partition(network_type, N, d, seed, dynamics, xaxis, regroup_criteria, space, degree_interval)
        #xs_resilience_kcore(network_type, N, d, seed, dynamics, xaxis, regroup)
        #xs_resilience_kcore_degree_partition(network_type, N, d, seed, dynamics, xaxis, space, degree_interval)
        pass
xaxis = 'k'

#wc_k(network_type, N, d, seed, dynamics, xaxis)
