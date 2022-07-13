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

#mpl.rcParams['axes.prop_cycle'] = cycler(color=['#fc8d62', '#66c2a5', '#8da0cb', '#ffd92f', '#a6d854',  '#e78ac3', '#e5c494', '#b3b3b3']) 



def xs_multi_group_decouple_space(dynamics, network_type, N, d_list, beta, betaeffect, group_num_list, seed_list, space):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    for d in d_list:
        delta_x_list = np.zeros((len(seed_list), len(group_num_list)))

        for i, seed in enumerate(seed_list):
            A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
            N_actual = np.size(A, 0)
            w = np.sum(A, 0)
            for j, group_num in enumerate(group_num_list):
                if betaeffect:
                    des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                else:
                    des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
                data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
                node_group_number = data[:, 0]
                node_index = np.array(data[:, 1], dtype=int)
                xs_multi = data[:, 2]

                xs_multi_decouple = data[:, 3]
                #xs_multi_group = data[:, 4]
                delta_x = np.sum(np.abs(xs_multi - xs_multi_decouple)/ xs_multi)/ N_actual
                #delta_x = np.sum(np.abs(xs_multi - xs_multi_decouple))/ N_actual
                delta_x_list[i, j] = delta_x
        if space == 'linear':
            labels = 'constant'
        elif space == 'log':
            labels = 'log'
        elif space == 'equal_node':
            labels = 'even'
        plt.semilogy(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label=labels)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('Error', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_multi_decouple_hist_fix_degree(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k):
    """TODO: Docstring for xs_multi_decouple_hist_fix_degree.

    :dynamics: TODO
    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    k_index = np.where(abs(w - k * beta)<1e-10)[0]
    k_xs_multi = xs_multi[k_index]
    k_xs_decouple = xs_decouple[k_index]
    k_xs_multi_decouple = np.vstack((k_xs_multi, k_xs_decouple)).transpose()
    nbins = 10
    #bins = [0, 1, 5, np.max(k_xs_multi_decouple)]
    plt.hist(k_xs_multi_decouple, nbins, histtype='bar', label = ['multi', 'decouple'], color=['#fc8d62', '#66c2a5'])
    plt.xlabel('$x_s$', fontsize=fs)
    plt.ylabel('$n$', fontsize=fs)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.legend(fontsize=legendsize, frameon=False)

    plt.show()
    return None

def xs_multi_hist_degrees(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k_list):
    """TODO: Docstring for xs_multi_decouple_hist_fix_degree.

    :dynamics: TODO
    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    k_xs_multi_list = []
    for k in k_list:
        k_index = np.where(abs(w - k * beta)<1e-10)[0]
        k_xs_multi = xs_multi[k_index]
        k_xs_multi_list.append(k_xs_multi)
    nbins = 10
    #bins = [0, 1, 5, np.max(k_xs_multi_decouple)]
    plt.hist(k_xs_multi_list, nbins, histtype='bar', label=['$k=$'+str(k) for k in k_list])
    plt.yscale('log')

    plt.xlabel('$x_s$', fontsize=fs)
    plt.ylabel('$n$', fontsize=fs)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.legend(fontsize=legendsize, frameon=False)

    plt.show()
    return None

def xs_error_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, absolute_error):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_group = data[:, 4]
            N_actual = len(xs_multi)
            if absolute_error == 'absolute':
                delta_x = np.sum(np.abs(xs_multi - xs_decouple))/ N_actual
                error_type = '$E_2$'
            elif absolute_error == 'relative':
                delta_x = np.sum(np.abs(xs_multi - xs_decouple)/xs_multi)/ N_actual
                error_type = '$E_1$'
            delta_x_list[i, j] = delta_x
    plt.semilogy(group_num_list, np.mean(delta_x_list, 0), linestyle='-', linewidth=lw, alpha=0.9, label='$l=0$')
    #plt.loglog(group_num_list, delta_x_list.transpose(), linestyle='--', linewidth=lw, alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    #plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel(error_type, fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_two_cluster_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step, absolute_error):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), iteration_step))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

            xs_multi = data[:, 0]
            w = data[:, 1]
            xs_beta = data[:, 2]
            xs_group_decouple = data[:, 3]
            for k in range(iteration_step):
                node_group_number = data[:, 3+k * 5+ 1]
                node_index = np.array(data[:, 3+ k * 5+ 2], dtype=int)
                xs_nn = data[:, 3+ k*5+ 3]
                xs_groups = data[:, 3+ k*5+ 4]
                xs_group_decouple_iteration = data[:, 3 + k*5 + 5]
                N_actual = len(xs_multi)
                if absolute_error == 'absolute':
                    delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration))/ N_actual
                    error_type = '$E_2$'
                elif absolute_error == 'relative':
                    delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration)/xs_multi[node_index])/ N_actual
                    error_type = '$E_1$'
                delta_x_list[i, j, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:, 1+l], linestyle='--', linewidth=lw, label=f'$l={l+1}$', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel(error_type, fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right', bbox_to_anchor=(1.35, -0.05))
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_two_cluster_iteration_adaptive_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step, absolute_error):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), iteration_step))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

            xs_multi = data[:, 0]
            w = data[:, 1]
            xs_beta = data[:, 2]
            for k in range(iteration_step):
                node_group_number = data[:, 2+k * 5+ 1]
                node_index = np.array(data[:, 2+ k * 5+ 2], dtype=int)
                xs_nn = data[:, 2+ k*5+ 3]
                xs_groups = data[:, 2+ k*5+ 4]
                xs_group_decouple_iteration = data[:, 2 + k*5 + 5]
                N_actual = len(xs_multi)
                if absolute_error == 'absolute':
                    delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration))/ N_actual
                    error_type = '$E_2$'
                elif absolute_error == 'relative':
                    delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration)/xs_multi[node_index])/ N_actual
                    error_type = '$E_1$'
                delta_x_list[i, j, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:,l], linestyle='--', linewidth=lw, label=f'$l={l+1}$', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel(error_type, fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right', bbox_to_anchor=(1.35, -0.05))
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_iteration(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step, absolute_error):
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
    group_num = 1
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), iteration_step))
    for i, seed in enumerate(seed_list):
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        xs_multi = data[:, 0]
        w = data[:, 1]
        xs_beta = data[:, 2]
        xs_group_decouple = data[:, 3]
        for k in range(iteration_step):
            node_group_number = data[:, 3+k * 5+ 1]
            node_index = np.array(data[:, 3+ k * 5+ 2], dtype=int)
            xs_nn = data[:, 3+ k*5+ 3]
            xs_groups = data[:, 3+ k*5+ 4]
            xs_group_decouple_iteration = data[:, 3 + k*5 + 5]
            N_actual = len(xs_multi)
            if absolute_error == 'absolute':
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_nn))/ N_actual
                error_type = '$E_2$'
            elif absolute_error == 'relative':
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_nn)/xs_multi[node_index])/ N_actual
                error_type = '$E_1$'
            delta_x_list[i, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    plt.semilogy(np.arange(iteration_step) + 1, delta_ave_seed, linestyle='--', linewidth=lw, label='parallel', alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$l$', fontsize=fs)
    plt.ylabel(error_type, fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_adaptive(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step, absolute_error):
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
    group_num = 1
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), iteration_step))
    for i, seed in enumerate(seed_list):
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

        xs_multi = data[:, 0]
        w = data[:, 1]
        xs_beta = data[:, 2]
        for k in range(iteration_step):
            node_group_number = data[:, 2+k * 5+ 1]
            node_index = np.array(data[:, 2+ k * 5+ 2], dtype=int)
            xs_nn = data[:, 2+ k*5+ 3]
            xs_groups = data[:, 2+ k*5+ 4]
            xs_group_decouple_iteration = data[:, 2 + k*5 + 5]
            N_actual = len(xs_multi)
            if absolute_error == 'absolute':
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_nn))/ N_actual
                error_type = '$E_2$'
            elif absolute_error == 'relative':
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_nn)/xs_multi[node_index])/ N_actual
                error_type = '$E_1$'
            delta_x_list[i, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    plt.semilogy(np.arange(iteration_step) + 1, delta_ave_seed, 'o',  linestyle='--', linewidth=lw, label='serial', alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$l$', fontsize=fs)
    plt.ylabel(error_type, fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_decouple_core(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_core_neighbor_sum_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_group = data[:, 4]
            N_actual = len(xs_multi)
            delta_x = np.sum(np.abs(xs_multi - xs_decouple)/xs_multi)/ N_actual
            delta_x = np.sum(np.abs(xs_multi - xs_decouple))/ N_actual
            delta_x_list[i, j] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    plt.semilogy(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label='average', color='#66c2a5')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$E_1$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_group_alpha(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_group = data[:, 4]
            N_actual = len(xs_multi)
            w = data[:, 5]
            group_length = np.size(np.unique(node_group_number))
            x_alpha = []
            xs_group_unique = []
            for k in range(group_length):
                group_index = np.where(node_group_number == k)[0]
                x_alpha.append(np.sum(xs_multi[group_index] * w[group_index]) / np.sum(w[group_index]))
                xs_group_unique.append(xs_group[group_index[0]])

            xs_group_unique = np.array(xs_group_unique)
            x_alpha = np.array(x_alpha)
            delta_x = np.sum(np.abs(xs_group_unique - x_alpha)/x_alpha)/ group_length
            delta_x_list[i, j] = delta_x
    plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    plt.semilogy(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label='average', color='#66c2a5')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$E_2$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_group = data[:, 4]
            w = data[:, 5]
            N_actual = len(xs_multi)
            delta_x = len(np.where(np.abs(xs_multi - xs_decouple) > 4)[0])
            delta_x_list[i, j] = delta_x
    #plt.plot(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    plt.plot(group_num_list, np.mean(delta_x_list, 0), linestyle='-', linewidth=lw, label='$l=0$')
    #plt.loglog(group_num_list, delta_x_list.transpose(), linestyle='--', linewidth=lw, alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    #plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_two_cluster_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), iteration_step))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

            xs_multi = data[:, 0]
            w = data[:, 1]
            xs_beta = data[:, 2]
            xs_group_decouple = data[:, 3]
            for k in range(iteration_step):
                node_group_number = data[:, 3+k * 5+ 1]
                node_index = np.array(data[:, 3+ k * 5+ 2], dtype=int)
                xs_nn = data[:, 3+ k*5+ 3]
                xs_groups = data[:, 3+ k*5+ 4]
                xs_group_decouple_iteration = data[:, 3 + k*5 + 5]
                N_actual = len(xs_multi)
                delta_x = len(np.where(np.abs(xs_multi[node_index] - xs_group_decouple_iteration) > 4)[0])
                delta_x_list[i, j, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:, 1+l],  linestyle='--', linewidth=lw, label=f'$l={l+1}$', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right', bbox_to_anchor=(1.35, -0.05))
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_two_cluster_iteration_adaptive_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), iteration_step))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

            xs_multi = data[:, 0]
            w = data[:, 1]
            xs_beta = data[:, 2]
            for k in range(iteration_step):
                node_group_number = data[:, 2+k * 5+ 1]
                node_index = np.array(data[:, 2+ k * 5+ 2], dtype=int)
                xs_nn = data[:, 2+ k*5+ 3]
                xs_groups = data[:, 2+ k*5+ 4]
                xs_group_decouple_iteration = data[:, 2 + k*5 + 5]
                N_actual = len(xs_multi)
                delta_x = len(np.where(np.abs(xs_multi[node_index] - xs_group_decouple_iteration) > 4)[0])
                delta_x_list[i, j, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:, l],  linestyle='--', linewidth=lw, label=f'$l={l+1}$', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right', bbox_to_anchor=(1.35, -0.05))
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_adaptive(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step):
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
    group_num = 1
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), iteration_step))
    for i, seed in enumerate(seed_list):
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        xs_multi = data[:, 0]
        w = data[:, 1]
        xs_beta = data[:, 2]
        for k in range(iteration_step):
            node_group_number = data[:, 2+k * 5+ 1]
            node_index = np.array(data[:, 2+ k * 5+ 2], dtype=int)
            xs_nn = data[:, 2+ k*5+ 3]
            xs_groups = data[:, 2+ k*5+ 4]
            xs_group_decouple_iteration = data[:, 2 + k*5 + 5]
            N_actual = len(xs_multi)
            delta_x = len(np.where(np.abs(xs_multi[node_index] - xs_nn) > 4)[0])
            delta_x_list[i, k] = delta_x
    delta_ave_seed = np.mean(delta_x_list, 0)
    plt.semilogy(np.arange(iteration_step) + 1, delta_ave_seed, 'o', linestyle='--', linewidth=lw, label='serial', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$l$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_iteration(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step):
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
    group_num = 1
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), iteration_step))
    for i, seed in enumerate(seed_list):
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        xs_multi = data[:, 0]
        w = data[:, 1]
        xs_beta = data[:, 2]
        xs_group_decouple = data[:, 3]
        for k in range(iteration_step):
            node_group_number = data[:, 3+k * 5+ 1]
            node_index = np.array(data[:, 3+ k * 5+ 2], dtype=int)
            xs_nn = data[:, 3+ k*5+ 3]
            xs_groups = data[:, 3+ k*5+ 4]
            xs_group_decouple_iteration = data[:, 3 + k*5 + 5]
            N_actual = len(xs_multi)
            delta_x = len(np.where(np.abs(xs_multi[node_index] - xs_nn) > 4)[0])
            delta_x_list[i, k] = delta_x
    delta_ave_seed = np.mean(delta_x_list, 0)
    plt.semilogy(np.arange(iteration_step) + 1, delta_ave_seed, 'o', linestyle='--', linewidth=lw, label=f'parallel', alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.80, wspace=0.25, hspace=0.25, bottom=0.18, top=0.88)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$l$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_decouple_core(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space):
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
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_core_neighbor_sum_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list)))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_group = data[:, 4]
            w = data[:, 5]
            N_actual = len(xs_multi)
            delta_x = len(np.where(np.abs(xs_multi - xs_decouple) > 4)[0])
            delta_x_list[i, j] = delta_x
    #plt.plot(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    plt.plot(group_num_list, np.mean(delta_x_list, 0), 'o', linestyle='--', linewidth=lw, label='average', color='#66c2a5')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def compare_xs_group_alpha(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space):
    """TODO: Docstring for betaspace.

    :A: TODO
    :x: TODO
    :group_num: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    group_length = np.size(np.unique(node_group_number))
    x_alpha = []
    xs_group_unique = []
    for i in range(group_length):
        group_index = np.where(node_group_number == i)[0]
        x_alpha.append(np.sum(xs_multi[group_index] * w[group_index]) / np.sum(w[group_index]))
        xs_group_unique.append(xs_group[group_index[0]])
    plt.plot(x_alpha, xs_group_unique, '.', markersize=10)
    x = [np.min(np.hstack((x_alpha, xs_group_unique))), np.max(np.hstack((x_alpha, xs_group_unique)))]
    plt.plot(x, x, '--', color='#b3b3b3')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$\\mathcal{L}_{\\alpha}(x_{\mathrm{multi}})$', fontsize=fs)
    plt.ylabel('$x_{\\mathrm{group}}$', fontsize=fs)
    plt.show()
    return None

def compare_xs_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space):
    """TODO: Docstring for betaspace.

    :A: TODO
    :x: TODO
    :group_num: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    plt.plot(xs_multi, xs_decouple, '.', markersize=10)
    x = [np.min(np.hstack((xs_multi, xs_decouple))), np.max(np.hstack((xs_multi, xs_decouple)))]
    plt.plot(x, x, '--', color='#b3b3b3')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$x_{\mathrm{multi}}$', fontsize=fs)
    plt.ylabel('$x_{\\mathrm{decouple}}$', fontsize=fs)
    plt.show()
    return None

def compare_xs_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space):
    """TODO: Docstring for betaspace.

    :A: TODO
    :x: TODO
    :group_num: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_nn_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    plt.plot(xs_multi, xs_decouple, '.', markersize=10)
    x = [np.min(np.hstack((xs_multi, xs_decouple))), np.max(np.hstack((xs_multi, xs_decouple)))]
    plt.plot(x, x, '--', color='#b3b3b3')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params(nbins=5)
    plt.xlabel('$x_{\mathrm{multi}}$', fontsize=fs)
    plt.ylabel('$x_{\\mathrm{decouple}}$', fontsize=fs)
    plt.show()
    return None

def xs_error_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), 10))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_all = data[:, 3: 14].transpose()
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_nn = data[:, 4:4+10].transpose()
            xs_group = data[:, -2]
            w = data[:, -1]
            N_actual = len(xs_multi)
            delta_x = np.sum(np.abs(xs_multi - xs_nn)/xs_multi, 1)/ N_actual
            delta_x = np.sum(np.abs(xs_multi - xs_nn), 1)/ N_actual
            delta_x_list[i, j, :] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:, 1+l] , 'o', linestyle='--', linewidth=lw, label=f'l={l}', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$E_1$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right')
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), iteration_step))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])

            xs_multi = data[:, 0]
            w = data[:, 1]
            xs_beta = data[:, 2]
            xs_group_decouple = data[:, 3]
            for k in range(iteration_step):
                node_group_number = data[:, 3+k * 5+ 1]
                node_index = np.array(data[:, 3+ k * 5+ 2], dtype=int)
                xs_nn = data[:, 3+ k*5+ 3]
                xs_groups = data[:, 3+ k*5+ 4]
                xs_group_decouple_iteration = data[:, 3 + k*5 + 5]
                N_actual = len(xs_multi)
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration)/xs_multi[node_index])/ N_actual
                delta_x = np.sum(np.abs(xs_multi[node_index] - xs_group_decouple_iteration))/ N_actual
                delta_x_list[i, j, k] = delta_x
    #plt.semilogy(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(9):
        plt.semilogy(group_num_list, delta_ave_seed[:, 1+l] , 'o', linestyle='--', linewidth=lw, label=f'l={l}', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$E_1$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right')
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def xs_error_multi_three_node(dynamics, network_type, N, d, beta, betaeffect, seed_list, group_num_list):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_three_level/'
    delta_x_list = np.zeros((len(seed_list)))
    for i, seed in enumerate(seed_list):
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_seed={seed}.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        node_index = np.array(data[:, 0], dtype=int)
        xs_multi = data[:, 1]
        xs_i = data[:, 2]
        xs_i_nn = data[:, 3].transpose()
        xs_beta = data[:, 4]
        w = data[:, -1]
        N_actual = len(xs_multi)
        delta_x = np.sum(np.abs(xs_multi - xs_i)/xs_multi)/ N_actual
        delta_x_list[i] = delta_x
    delta_ave_seed = np.mean(delta_x_list)
    plt.semilogy(group_num_list, delta_ave_seed * np.ones(len(group_num_list)) , 'o', linestyle='--', linewidth=lw, label='three-node', alpha=0.7)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$E_1$', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, loc='lower right')
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_error_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_' + space + '/'
    delta_x_list = np.zeros((len(seed_list), len(group_num_list), 10))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_nn = data[:, 4:4+10].transpose()
            xs_group = data[:, -2]
            w = data[:, -1]
            N_actual = len(xs_multi)
            delta_x = np.sum((np.abs(xs_multi - xs_nn) > 4), 1)
            delta_x_list[i, j, :] = delta_x
    #plt.plot(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    delta_ave_seed = np.mean(delta_x_list, 0)
    for l in range(10):
        plt.plot(group_num_list, delta_ave_seed[:, l] , 'o', linestyle='--', linewidth=lw, label=f'l={l}')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes of wrong states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None

def num_high_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step):
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
    des = '../data/' + dynamics + '/' + network_type + f'/xs_group_iteraction_{iteration_step}_' + space + '/'
    num_high_list = np.zeros((len(seed_list), len(group_num_list), 10+ 2))
    for i, seed in enumerate(seed_list):
        for j, group_num in enumerate(group_num_list):
            if betaeffect:
                des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            else:
                des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
            node_group_number = data[:, 0]
            node_index = np.array(data[:, 1], dtype=int)
            xs_all = data[:, 2 : 14].transpose()
            xs_multi = data[:, 2]
            xs_decouple = data[:, 3]
            xs_nn = data[:, 4:4+10].transpose()
            xs_group = data[:, -2]
            w = data[:, -1]
            N_actual = len(xs_multi)
            num_high = np.sum(xs_all > 5, 1)
            num_high_list[i, j, :] = num_high
    #plt.plot(group_num_list, delta_x_list.transpose(), 'o', markersize=3, linestyle='-', linewidth=2, color='#fc8d62')
    num_high_ave = np.mean(num_high_list, 0)
    plt.plot(group_num_list, num_high_ave[:, 1] , 'o', linestyle='--', linewidth=lw, label=f'l={0}', alpha=0.7)
    for l in range(8):
        plt.plot(group_num_list, num_high_ave[:, l+2] , 'o', linestyle='--', linewidth=lw, label=f'l={l+1}', alpha=0.7)
    plt.plot(group_num_list, num_high_ave[:, 0] , 'o', linestyle='--', linewidth=lw, label='multi', alpha=0.7)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.locator_params('x', nbins=5)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$n$ nodes in high states', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report062321/' + dynamics + '_' + network_type + '_error_m.png')
    #plt.show()
    #plt.close('all')
    return None


def state_high_low_degrees(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space):
    """TODO: Docstring for state_high_low_degrees.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_three_level/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_index = np.array(data[:, 0], dtype=int)
    xs_multi = data[:, 1]
    w = data[:, -1]
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    neighbors_num =  np.sum(A>0, 0)
    G = nx.convert_matrix.from_numpy_matrix(A)
    core_number = np.array(list(nx.core_number(G).values())) 
    core_neighbor = [core_number[np.where(A[i] > 0)[0]] for i in range(N_actual)]
    core_neighbor_sum = np.array([np.sum(i) for i in core_neighbor])
    core_neighbor_ave = core_neighbor_sum / neighbors_num
    neighbors_weights = [w[np.where(A[i] > 0)[0]] for i in range(N_actual)]
    neighbors_weights_sum = np.array([np.sum(i) for i in neighbors_weights])
    neighbors_weights_ave = neighbors_weights_sum / neighbors_num
    #colors = list(mpl.cm.rainbow(xs_multi/xs_multi.max()))
    index_high = np.where(xs_multi > 5)[0]
    index_low = np.where(xs_multi < 1)[0]
    y = core_number
    y = neighbors_weights_ave
    y = neighbors_weights_sum
    y = core_neighbor_ave
    y = core_neighbor_sum 
    x = w
    delta_y = np.diff(np.unique(np.sort(y))).min() * 1
    print(delta_y)
    if seed == [0, 0]:
        sc = plt.scatter(x[index_high], y[index_high] + np.random.random(np.size(index_high)) * delta_y * 0.1, color='tab:green', alpha = 0.5, s = 10, label='active')
        sc.set_facecolor('none')
        sc = plt.scatter(x[index_low], y[index_low] + np.random.random(np.size(index_low)) * delta_y* 0.1, color='tab:red', alpha = 0.5, s = 10, label='dead')
        sc.set_facecolor('none')
    else:
        sc = plt.scatter(x[index_high], y[index_high] + np.random.random(np.size(index_high)) * delta_y* 0.1, color='tab:green', alpha = 0.5, s = 10)
        sc.set_facecolor('none')
        sc = plt.scatter(x[index_low], y[index_low] + np.random.random(np.size(index_low)) * delta_y * 0.1, color='tab:red', alpha = 0.5, s = 10)
        sc.set_facecolor('none')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xlabel('$k$', fontsize=fs)
    plt.ylabel('$sum(k^{(nn)})$', fontsize=fs)
    plt.ylabel('$\\langle k^{(nn)} \\rangle$', fontsize=fs)
    plt.ylabel('core', fontsize=fs)
    plt.legend(fontsize=legendsize, frameon=False, markerscale=2)

    return neighbors_weights_sum, neighbors_weights_ave

def state_high_low_features(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k):
    """TODO: Docstring for state_high_low_degrees.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_group_decouple_' + space + '/'
    if betaeffect:
        des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
    else:
        des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    node_group_number = data[:, 0]
    node_index = np.array(data[:, 1], dtype=int)
    xs_multi = data[:, 2]
    xs_decouple = data[:, 3]
    xs_group = data[:, 4]
    w = data[:, 5]
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    G = nx.convert_matrix.from_numpy_matrix(A)
    core_number = np.array(list(nx.core_number(G).values())) 
    core_neighbor = [core_number[np.where(A[i] > 0)[0]] for i in range(N_actual)]
    core_neighbor_sum = np.array([np.sum(i) for i in core_neighbor])

    neighbors_weights = np.array([w[np.where(A[i] > 0)[0]] for i in range(N_actual)])
    neighbors_index = np.array([np.where(A[i] > 0)[0] for i in range(N_actual)])
    neighbors_num =  np.sum(A>0, 0)
    neighbors_weights_sum = np.array([np.sum(i) for i in neighbors_weights])
    neighbors_weights_ave = neighbors_weights_sum / neighbors_num

    index_k = np.where(np.abs(w - beta * k) < 1e-9)[0]
    xs_k = xs_multi[index_k]
    xs_k_low_index = np.where(xs_k < 1)[0]
    xs_low_index = index_k[xs_k_low_index]

    xs_k_high_index = np.where(xs_k > 1)[0]
    xs_high_index = index_k[xs_k_high_index]
    plt.semilogx(w, xs_multi, '.')
    #plt.semilogx(core_neighbor_sum, xs_multi, '.')
    plt.semilogx(np.log(core_neighbor_sum) * w, xs_multi, '.')


    return xs_multi, xs_low_index, xs_high_index, neighbors_weights, neighbors_index



seed1 = [0, 2, 3, 4, 5, 7, 8]
seed1 = np.arange(10).tolist()
seed1 = [0]
network_type = 'SF'
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
network_type = 'ER'
seed_ER = seed1

N = 1000
beta = 1
betaeffect = 0



dynamics = 'PPI'
dynamics = 'CW'
dynamics = 'BDP'
dynamics = 'mutual'

network_type = 'star'
d_list = np.arange(100, 120, 10)
d_list = [999]



network_type = 'RGG'
d_list = [0.04, 0.05, 0.07]
d_list = [0.07]
seed = 0
seed_list = seed_ER
group_num_list = np.arange(1, 10, 1)

network_type = 'ER'
d_list = [2000, 4000, 8000]
d_list = [2000]
seed_list = seed_ER
seed = 0
group_num_list = np.arange(1, 10, 1)


network_type = 'SF'
d_list = [[2.1, 999, 2], [2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_list = [[2.5, 999, 3]]
d = [2.5, 999, 3]
seed_list = seed_SF
seed = [0, 0]
group_num = 10
space = 'log'
beta = 0.15
k = 7
iteration_step = 10

#xs_multi_decouple_hist_fix_degree(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k)
k_list = np.arange(2, 10, 1)
#xs_multi_hist_degrees(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k_list)
group_num_list = np.hstack((np.arange(1, 30, 1), np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 900])))
group_num_list = np.arange(1, 20, 1)
absolute_error = 'absolute'
absolute_error = 'relative'
#xs_error_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, absolute_error)
#xs_error_multi_two_cluster_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step, absolute_error)
#xs_error_multi_two_cluster_iteration_adaptive_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step, absolute_error)
#xs_error_multi_adaptive(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step, absolute_error)
#xs_error_multi_iteration(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step, absolute_error)

#num_error_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space)
#num_error_multi_two_cluster_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
#num_error_multi_two_cluster_iteration_adaptive_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
num_error_multi_adaptive(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step)
num_error_multi_iteration(dynamics, network_type, N, d, beta, betaeffect, seed_list, space, iteration_step)

#xs_error_multi_iteration_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
#xs_error_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
#xs_error_multi_three_node(dynamics, network_type, N, d, beta, betaeffect, seed_list, group_num_list)
#num_error_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
#xs_error_group_alpha(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space)
#compare_xs_group_alpha(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space)
#compare_xs_multi_decouple(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space)
#compare_xs_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space)
#num_high_multi_decouple_nn(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space, iteration_step)
#xs_error_multi_decouple_core(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space)
#num_error_multi_decouple_core(dynamics, network_type, N, d, beta, betaeffect, group_num_list, seed_list, space)
#for seed in seed_list:
    #a, b = state_high_low_degrees(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space)
k = 3
seed = [0, 0]
#xs_multi, xs_low_index, xs_high_index, neighbors_weights, neighbors_index = state_high_low_features(dynamics, network_type, N, d, beta, betaeffect, group_num, seed, space, k)
