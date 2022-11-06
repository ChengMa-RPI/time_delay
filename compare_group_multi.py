import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import network_generate



import numpy as np 
import pandas as pd
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

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#fc8d62', '#66c2a5', '#8da0cb', '#a6d854',  '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3']) 



def compare_xs_multi_group(dynamics, network_type, N, d, beta, betaeffect, group_num, seed):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'


    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    degree = data[:, 0]
    xs_multi_degree = data[:, 1]
    xs_multi_group = data[:, 2]
    plt.plot(xs_multi_degree, xs_multi_group , 'o', color='#8da0cb')
    x = [np.min(np.hstack((xs_multi_degree, xs_multi_group))), np.max(np.hstack((xs_multi_degree, xs_multi_group)))]
    plt.plot(x, x, '--', color='#fc8d62')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$x_i$', fontsize=fs)
    plt.ylabel('$x_{\\mathrm{group}}$', fontsize=fs)
    #plt.savefig('../report/report060121/' + dynamics + '_' + network_type + '_' + str(d) + f'_group_num={group_num}_seed={seed}' + '_x_group_x_i.png')
    #plt.show()
    #plt.close('all')
    return None

def compare_xs_multi_group_degree_weighted(dynamics, network_type, N, d, beta, betaeffect, group_num, seed):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_degree_weighted/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_index = data[:, 1]
    xs_multi_degree = data[:, 2]
    xs_multi_group = data[:, 3]
    plt.plot(xs_multi_degree, xs_multi_group , 'o', color='#8da0cb')
    x = [np.min(np.hstack((xs_multi_degree, xs_multi_group))), np.max(np.hstack((xs_multi_degree, xs_multi_group)))]
    plt.plot(x, x, '--', color='#fc8d62')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$x_i$', fontsize=fs)
    plt.ylabel('$x_{\\mathrm{group}}$', fontsize=fs)
    #plt.savefig('../report/report060121/' + dynamics + '_' + network_type + '_' + str(d) + f'_group_num={group_num}_seed={seed}' + '_x_group_x_i.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_multi_group_num(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            degree = data[:, 0]
            N_actual = len(degree)
            xs_multi_degree = data[:, 1]
            xs_multi_group = data[:, 2]
            xs_multi_group_unique = np.unique(xs_multi_group)
            xs_multi_mean = []
            group_each_length = []
            for xs_group_i in xs_multi_group_unique:
                index = np.where(xs_multi_group == xs_group_i)[0]
                xs_multi_mean.append(np.mean(xs_multi_degree[index]))
                group_each_length.append(len(index))
            xs_multi_mean = np.array(xs_multi_mean)
            delta_x = np.sum(np.abs(xs_multi_mean - xs_multi_group_unique)/xs_multi_mean * group_each_length) / N_actual
            delta_x_list[i, j] = delta_x
    if network_type == 'ER':
        labels = f'$\\langle k \\rangle ={int(2*d/N)}$'
    elif network_type =='SF':
        labels = f'$\\gamma = {d[0]}$'
    plt.plot(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label=labels)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('Error', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.show()

def xs_multi_degree_weighted_group_num(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_degree_weighted/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        w = np.sum(A, 0)
        N_actual = np.size(A, 0)
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            group_number = data[:, 0]
            node_index = data[:, 1]
            xs_multi_degree = data[:, 2]
            xs_multi_group = data[:, 3]
            group_all = np.unique(group_number)
            x_eff = []
            group_each_length = []
            xs_multi_group_unique = []
            for group_i in group_all:
                index = np.where(group_number == group_i)[0]
                group_each_length.append(len(index))
                node_i = np.array(node_index[index], dtype=int)
                w_i = w[node_i]
                x_eff.append((xs_multi_degree[index] * w_i).sum() / w_i.sum())
                xs_multi_group_unique.append(xs_multi_group[index[0]])
            group_each_length = np.array(group_each_length)
            x_eff = np.array(x_eff)
            xs_multi_group_unique = np.array(xs_multi_group_unique)
            delta_x = np.sum((np.abs(xs_multi_group_unique - x_eff)/ x_eff) * group_each_length)/ N_actual
            #delta_x = np.sum(np.abs(xs_multi_group - xs_multi_degree)/ xs_multi_degree)/ N_actual
            delta_x_list[i, j] = delta_x
    if network_type == 'ER':
        labels = f'$\\langle k \\rangle ={int(2*d/N)}$'
    elif network_type =='SF':
        labels = f'$\\gamma = {d[0]}$'
    plt.plot(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label=labels)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('Error', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.show()

def xs_multi_degree(dynamics, network_type, N, d, beta, betaeffect, seed):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group/'
    group_num = 1
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    degree = data[:, 0]
    xs_multi_degree = data[:, 1]
    xs_multi_group = data[:, 2]

    if network_type == 'ER':
        labels = f'$\\langle k \\rangle ={int(2*d/N)}$'
    elif network_type =='SF':
        labels = f'$\\gamma = {d[0]}$'

    plt.plot(degree, xs_multi_degree, 'o', label=labels)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$k_i$', fontsize=fs)
    plt.ylabel('$x_i$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.show()


seed1 = np.arange(10).tolist()
network_type = 'SF'
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
network_type = 'ER'
seed_ER = seed1

N = 1000
beta = 1
betaeffect = 0



dynamics = 'BDP'
dynamics = 'PPI'
dynamics = 'mutual'
dynamics = 'CW'


network_type = 'star'
d_list = np.arange(100, 120, 10)
d_list = [999]
seed = 0



network_type = 'SF'
d = [3.8, 999, 5]
d_list = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_list = [[3, 999, 4]]
seed_list = seed_SF
seed = [0, 0]

network_type = 'ER'
d = 2000
d_list = [2000, 4000, 8000]
d_list = [4000]
seed = 4
seed_list = seed_ER

network_type = 'RGG'
d_list = [0.04, 0.05, 0.07]
d_list = [0.05]
seed = 0
seed_list = seed_ER





group_num = 10
group_num_list = [1]
for d in d_list:
    for group_num in group_num_list:
        #compare_xs_multi_group(dynamics, network_type, N, d, beta, betaeffect, group_num, seed)
        compare_xs_multi_group_degree_weighted(dynamics, network_type, N, d, beta, betaeffect, group_num, seed)
        pass

group_num_list = np.arange(1, 11, 1)
for d in d_list:
    #xs_multi_group_num(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list)
    #xs_multi_degree_weighted_group_num(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list)
    #xs_multi_degree(dynamics, network_type, N, d, beta, betaeffect, seed)
    pass

plt.show()
