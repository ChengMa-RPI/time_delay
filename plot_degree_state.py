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
import time
from netgraph import Graph
from matplotlib.legend_handler import HandlerTuple
import scipy

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
ticksize = 18
labelsize = 17
anno_size = 14
subtitlesize = 15
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ]


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()




def read_xs(dynamics, network_type, N, space, d, seed, m, weight_list):
    """TODO: Docstring for read_xs.

    :network_type: TODO
    :N: TODO
    :space: TODO
    :d: TODO
    :seed: TODO
    :m: TODO
    :returns: TODO

    """
    A_des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/'
    save_file = A_des + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(save_file).toarray()

    des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/'
    if m == N:
        xs_multi_file = des + 'xs_multi/' + f'N={N}_d={d}_seed={seed}.csv'
        data_multi = np.array(pd.read_csv(xs_multi_file, header=None))
        weights_multi = data_multi[:, 0]
        index = [np.where(np.abs(weights_multi - weight) < 1e-05 )[0][0]  for weight in weight_list]
        xs = data_multi[index, 1:]
    else:
        xs_group_file = des + f'degree_kmeans_space={space}/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data_group = np.array(pd.read_csv(xs_group_file, header=None))
        weights_group = data_group[:, 0]
        index = [np.where(np.abs(weights_group - weight) < 1e-05 )[0][0]  for weight in weight_list]
        xs = data_group[index, 1:]
    degree = np.sum(A, 1)
    return xs, A, degree


def compare_plot(dynamics, arguments, xs, group_index, degree, m, ax, f_or_g, color, legend, markerstyle):
    reduce_f = lambda x: np.array([np.mean(x[group_i] * degree[group_i]) / np.mean(degree[group_i]) for group_i in group_index])
    def reduce_g(x):
        reduce_j = np.vstack(([np.mean(x[:, group_i] * degree[group_i], axis=-1) / np.mean(degree[group_i]) for group_i in group_index])).transpose()
        reduce_ij = np.vstack(([ np.mean(reduce_j[group_i].transpose() * degree[group_i], axis=-1) / np.mean(degree[group_i]) for group_i in group_index]))
        return reduce_ij
    if markerstyle == 'o':
        markersize = 8
    elif markerstyle == 's':
        markersize = 6
    simpleaxis(ax)
    if dynamics == 'mutual':
        B, C, D, E, H, K = arguments
        f_i = lambda x: B + x * (1 - x/K) * ( x/C - 1)
        g_ij = lambda xi, xj: xi.reshape(len(xi), 1) * xj / (D + E * xi.reshape(len(xi), 1) + H * xj)

    if f_or_g == 'f':
        L_f = reduce_f(f_i(xs))
        f_L = f_i(reduce_f(xs))
        ax.plot(f_L, L_f, markerstyle, markersize=markersize, color=color, label=legend, alpha=0.6)
        data = np.hstack((f_L, L_f))
    else:
        g_original = g_ij(xs, xs)
        L_g = reduce_g(g_original).flatten()
        g_x = reduce_f(xs)
        g_L = g_ij(g_x, g_x).flatten()
        ax.plot(g_L, L_g, markerstyle, markersize=markersize, color=color, label=legend, alpha=0.6)
        data = np.hstack((L_g, g_L))

    xmin = np.min(data)
    xmax = np.max(data)
    #ax.plot([xmin, xmax], [xmin, xmax], '--', color='grey')
    return xmin, xmax

    
def degree_state(network_type_list, space_list, N, d_list_list, seed_list_list, weight_list_list, dynamics):
    """TODO: Docstring for main_fig.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :dynamics: TODO
    :returns: TODO

    """
    colors = ['Reds', 'Blues', 'Greens']
    letters = list('abcdefghijklmnopqrstuvwxyz')
    fig = plt.figure(figsize=(16, 7))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    #markers = ['+', '*', 'o']
    for (i, network_type), space, d_list, seed_list, weight_list in zip(enumerate(network_type_list), space_list, d_list_list, seed_list_list, weight_list_list):
        for j, weight in enumerate(weight_list):
            ax = fig.add_subplot(gs[i, j])
            simpleaxis(ax)
            title_letter = letters[i * len(weight_list) + j]
            ax.annotate('(' + title_letter + ')', xy=(-0.2, 1.03), xycoords="axes fraction", size=17)
            ax.set_title('$w=$' + f'{weight}', fontsize=18)

            for k, d in enumerate(d_list):
                if network_type == 'ER':
                    label = '$\\langle k \\rangle = $' + f'{int(d*2/N)}'
                else:
                    label = '$\\gamma=$' + f'{d[0]}'
                degree_list = []
                xs_list = []
                for seed in seed_list:
                    xs, A, degree = read_xs(dynamics, network_type, N, space, d, seed, m, weight_list)
                degree_list.extend(degree.tolist())
                xs_list.extend(xs[j].tolist())
                ax.scatter(degree_list, xs_list, marker='o', alpha=0.5, s=10, label=label)
                ax.tick_params(axis='both', which='major', labelsize=14)
                if j == len(weight_list)-1:
                    ax.legend(markerscale=2, handlelength=0.1, fontsize=legendsize, frameon=False, loc=(0.6, 0.1))

    fig.text(0.02, 0.5, '$x_s$', size=25, )
    fig.text(0.51, 0.05, '$k$', size=25, )
    plt.subplots_adjust(left=0.10, right=0.95, wspace=0.26, hspace=0.30, bottom=0.15, top=0.95)
    return None

def approx_justify(network_type_list, space_list, N, d_list_list, seed_list_list, weight_list_list, m_list, dynamics, arguments):
    """TODO: Docstring for main_fig.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :dynamics: TODO
    :returns: TODO

    """
    letters = list('abcdefghijklmnopqrstuvwxyz')
    cols = len(m_list)
    rows = 2
    fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))
    markerstyles = ['o', 's']
    min_f_list = np.ones(len(m_list)) * 1000
    max_f_list = np.ones(len(m_list)) * (-1000)
    min_g_list = np.ones(len(m_list)) * 1000
    max_g_list = np.ones(len(m_list)) * (-1000)
    for (j, network_type), space, d_list, seed_list, weight_list in zip(enumerate(network_type_list), space_list, d_list_list, seed_list_list, weight_list_list):
        color=colors[j]
        for k, weight in enumerate(weight_list):
            for l, d in enumerate(d_list):
                label = network_type
                for seed in seed_list:
                    xs_list, A, degree = read_xs(dynamics, network_type, N, space, d, seed, N, weight_list)
                    xs = xs_list[k]
                    feature = feature_from_network_topology(A, 'None', space, 0, 'degree')
                    for i, m in enumerate(m_list):
                        group_index = group_index_from_feature_Kmeans(feature, m)
                        for row_i, f_or_g in enumerate(['f', 'g']):
                            ax = axes[row_i, i]
                            if k == len(weight_list)-1 and l == len(d_list)-1 and seed == seed_list[-1]:
                                min_i, max_i = compare_plot(dynamics, arguments, xs, group_index, degree, m, ax, f_or_g, color, network_type_list[j], markerstyles[j])
                            else:
                                min_i, max_i = compare_plot(dynamics, arguments, xs, group_index, degree, m, ax, f_or_g, color, '', markerstyles[j])
                            ax.tick_params(axis='both', which='major', labelsize=14)
                            if f_or_g == 'f':
                                min_f_list[i] = min(min_f_list[i], min_i)
                                max_f_list[i] = max(max_f_list[i], max_i)
                            else:
                                min_g_list[i] = min(min_g_list[i], min_i)
                                max_g_list[i] = max(max_g_list[i], max_i)

    for i_m, m in enumerate(m_list):
        axes[0, i_m].set_title(f'$m={m}$', size=22)
        for i_f, f_or_g in enumerate(['f', 'g']):
            title_letter = letters[i_f * len(m_list) + i_m]
            axes[i_f, i_m].annotate('(' + title_letter + ')', xy=(-0.2, 1.03), xycoords="axes fraction", size=21)
            if f_or_g == 'f':
                min_i = min_f_list[i_m]
                max_i = max_f_list[i_m]
            else:
                min_i = min_g_list[i_m]
                max_i = max_g_list[i_m]
            axes[i_f, i_m].plot([min_i, max_i], [min_i, max_i], '--', color='grey')
            

    axes[-1, -1].legend(markerscale=2, handlelength=0.1, fontsize=legendsize, frameon=False, loc=(0.9, 0.1))

    fig.text(0.03, 0.3, '$\\mathcal{L}_G$', size=25, )
    fig.text(0.03, 0.75, '$\\mathcal{L}_F$', size=25, )
    fig.text(0.50, 0.05, '$G_{\\mathcal{L}}$', size=25, )
    fig.text(0.50, 0.53, '$F_{\\mathcal{L}}$', size=25, )
    plt.subplots_adjust(left=0.10, right=0.92, wspace=0.26, hspace=0.50, bottom=0.15, top=0.95)
    return None

def approx_error(dynamics_list, argument, network_type_list, space_list, N, d_list_list, seed_list_list, weight_list_list, m_list):
    letters = list('abcdefghijklmnopqrstuvwxyz')
    cols = len(dynamics_list)
    rows = 2
    fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))
    for (i, dynamics), arguments in zip(enumerate(dynamics_list), arguments_list):
        for (j, network_type), space, d_list, seed_list, weight_list in zip(enumerate(network_type_list), space_list, d_list_list, seed_list_list, weight_list_list):
            color=colors[j]
            for k, weight in enumerate(weight_list):
                for l, d in enumerate(d_list):
                    label = network_type
                    for seed in seed_list:
                        xs_list, A, degree = read_xs(dynamics, network_type, N, space, d, seed, N, weight_list)
                        xs = xs_list[k]
                        feature = feature_from_network_topology(A, 'None', space, 0, 'degree')
                        for i, m in enumerate(m_list):
                            group_index = group_index_from_feature_Kmeans(feature, m)
                            for row_i, f_or_g in enumerate(['f', 'g']):
                                ax = axes[row_i, i]






dynamics = 'CW'
arguments = (a, b)
dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)

network_type = 'ER'
N = 1000
space = 'linear'
d = 4000
seed = 0
m = 1000
weight_list = np.round(np.arange(0.01, 1.0, 0.01), 5)


network_type_list = ['ER', 'SF']
space_list = ['linear', 'log']
N = 1000
d_list_list = [[2000, 4000, 8000], [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]]
seed_list_list = [[i for i in range(1)], [[i, i] for i in range(1)]]
weight_list_list = [[0.1, 0.2, 0.6, 0.9], [0.01, 0.1, 0.3, 0.6]] 
#degree_state(network_type_list, space_list, N, d_list_list, seed_list_list, weight_list_list, dynamics)

network_type_list = ['SF'] 
space_list = ['log']
N = 1000
d_list_list = [[[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]]
seed_list_list = [ [[i, i] for i in range(1)]]
weight_list_list = [[0.3, 0.6]] 

network_type_list = ['ER', 'SF']
space_list = ['linear', 'log']
N = 1000
d_list_list = [[4000, 8000], [[2.5, 999, 3], [3, 999, 4], ]]
seed_list_list = [[i for i in range(1)], [[i, i] for i in range(1)]]
weight_list_list = [[0.6], [0.3]] 

m_list = [ 1, 2, 5, 10]
approx_justify(network_type_list, space_list, N, d_list_list, seed_list_list, weight_list_list, m_list, dynamics, arguments)
