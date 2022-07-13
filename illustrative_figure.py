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

from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

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

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#8da0cb', '#e78ac3','#a6d854',  '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))



def ER_SBM_partition(arg1):
    """TODO: Docstring for ER_SBM_partition.

    :arg1: TODO
    :returns: TODO

    """
    network_type = 'SBM_ER'
    N = [30, 30, 30]
    beta = 1
    betaeffect = 0 
    seed = 2
    d = [[0.9, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.1]]

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    G = nx.from_numpy_matrix(A)
    k = np.sum(A, 0)


    space = 'linear'
    tradeoff_para = 0.5
    method = 'degree'
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    number_groups = 3
    group_index = group_index_from_feature_Kmeans(feature, number_groups)
    val_map = dict()
    color_val = ['tab:red', 'tab:green', 'tab:blue']

    color_val = ['#1b9e77', '#d95f02', '#7570b3']
    for i, group_i in enumerate(group_index):
        for j in group_i:
            val_map[j] = color_val[i]

    node_color = [val_map.get(node) for node in range(sum(N))]

    pos=nx.nx_agraph.graphviz_layout(G, prog='neato')
    pos_map = dict()
    pos_group = [[0, 0], [100, 100], [100, -100]]
    for i, group_i in enumerate(group_index):
        for j in group_i:
            pos_map[j] = (np.random.random(2) * 70 + pos_group[i])

    node_size = [v*15 for v in k]

    nx.draw(G, pos=pos_map, node_color=node_color, node_size=node_size, edgecolors='k')
    plt.savefig('../manuscript/dimension_reduction_v1_111021/figure/network_' + network_type + '_' + method +  '_space=' + space + f'_N={N}_d={d}_group_num={number_groups}_seed={seed}'   + '.svg', format="svg") 

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    plt.show()

def network_partition(network_type, N, seed, d, beta, betaeffect, number_groups, space):
    """TODO: Docstring for ER_SBM_partition.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    G = nx.from_numpy_matrix(A)
    k = np.sum(A, 0)
    N_actual = len(A)

    tradeoff_para = 0.5
    method = 'degree'
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, number_groups)
    val_map = dict()

    color_val = [ '#66c2a5', '#fc8d62', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ]
    for i, group_i in enumerate(group_index):
        for j in group_i:
            val_map[j] = color_val[i]

    node_color = [val_map.get(node) for node in range(N_actual)]

    pos=nx.nx_agraph.graphviz_layout(G, prog='neato')

    node_size = [v*15 for v in k]
    nx.draw(G, pos=pos, node_color=node_color, node_size=node_size, edgecolors='k')
    plt.savefig('/home/mac/RPI/research/timedelay/manuscript/dimension_reduction_v1_111021/SF_network.svg', format="svg") 

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    plt.close()

    pos_map = dict()
    pos_group = [[0, 0], [100, 100], [100, -100]]
    for i, group_i in enumerate(group_index):
        for j in group_i:
            pos_map[j] = (np.random.random(2) * 70 + pos_group[i])
    group_belongs = dict()
    for i, group_i in enumerate(group_index):
        for node in group_i:
            group_belongs[node] = i
    edges = list(G.edges())
    edge_within = []
    edge_between = []
    for edge in edges:
        if group_belongs[edge[0]] == group_belongs[edge[1]]:
            edge_within.append(edge)
        else:
            edge_between.append(edge)



    nx.draw(G, pos=pos_map, edgelist=edge_within, node_color=node_color, node_size=node_size, edgecolors='k')
    #nx.draw(G, pos=pos_map, edgelist=edge_between, width=0.0001, node_color=node_color, node_size=node_size, edgecolors='k')
    plt.savefig('/home/mac/RPI/research/timedelay/manuscript/dimension_reduction_v1_111021/network_partition.svg', format="svg") 
    plt.close()



def plot_xs_bifurcation_m(network_type, N, seed, d, w, m_list, attractor_value, space, tradeoff_para, method):
    """TODO: Docstring for plot_xs_bifurcation_m.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des += '/'
    elif method == 'degree':
        des += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    xs_low = np.zeros(len(m_list))
    xs_high = np.zeros(len(m_list))
    for i, m in enumerate(m_list):
        des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        weight_list = data[:, 0]
        plot_index = np.where(np.abs(weight_list-w)<1e-8)[0][0]
        xs_i = data[plot_index, 1:]
        xs_low[i] = np.min(xs_i)
        xs_high[i] = np.max(xs_i)
        plt.semilogx(np.ones(len(xs_i)) * m, xs_i, '.', color='#66c2a5', markersize=18 / np.log( m  +1))

    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    weight_list = data[:, 0]
    plot_index = np.where(np.abs(weight_list-w)<1e-8)[0][0]
    xs_multi = data[plot_index, 1:]
    plt.loglog(np.ones(len(xs_multi)) * len(xs_multi), xs_multi, '.', color='#66c2a5', markersize=18 / np.log( m  +1))
    xs_low = np.append(xs_low, np.min(xs_multi))
    xs_high = np.append(xs_high, np.max(xs_multi))
    #plt.plot(np.hstack((m_list, len(xs_multi) )), xs_low, linewidth=lw, alpha=alpha, color='#fc8d62') 
    #plt.plot(np.hstack((m_list, len(xs_multi) )), xs_high, linewidth=lw, alpha=alpha, color='#8da0cb') 
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$x^{(i)}$', fontsize=fs)
    save_des = '../report/report012722/' + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_weight={w}_xs_m.svg'
    plt.savefig(save_des, format='svg')
    plt.close()
    return None
        
def plot_wc_bifurcation_m(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, high_criteria):
    """TODO: Docstring for plot_xs_bifurcation_m.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des += '/'
    elif method == 'degree':
        des += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    for i, m in enumerate(m_list):
        des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        weight_list = data[:, 0]
        xs_i = data[:, 1:]
        wc_i = [weight_list[np.where(xs_i[:, i] > high_criteria)[0][0]] for i in range(m)]
        plt.semilogx(np.ones(len(wc_i)) * m, wc_i, '.', color='#66c2a5', markersize=18 / np.log( m + 1), alpha=0.6)

    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    weight_list = data[:, 0]
    xs_multi = data[:, 1:]
    wc_multi = [weight_list[np.where(xs_multi[:, i] > high_criteria)[0][0]] for i in range(len(xs_multi[0]))]
    plt.semilogx(np.ones(len(wc_multi)) * len(xs_multi[0]), wc_multi, '.', color='#66c2a5', markersize=18 / np.log( m  +1))
    """
    xs_low = np.append(xs_low, np.min(xs_multi))
    xs_high = np.append(xs_high, np.max(xs_multi))
    plt.plot(np.hstack((m_list, len(xs_multi) )), xs_low, linewidth=lw, alpha=alpha, color='#fc8d62') 
    plt.plot(np.hstack((m_list, len(xs_multi) )), xs_high, linewidth=lw, alpha=alpha, color='#8da0cb') 
    """

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$w_c^{(i)}$', fontsize=fs)
    #plt.legend(fontsize=13, frameon=False) 
    save_des = '../report/report012722/' + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_wc_m.svg'
    plt.savefig(save_des, format='svg')
    plt.close()
    return None


def data_xs(network_type, N, seed, d, weight_list, m_list, space, tradeoff_para, method):
    """TODO: Docstring for plot_xs_bifurcation_m.

    :arg1: TODO
    :returns: TODO

    """

    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_multi = data[index, 1:]

    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des += '/'
    elif method == 'degree':
        des += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)

    A = A_unit * 1
    N_actual = len(A)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    xs_reduction_multi = np.zeros((len(m_list), len(weight_list), N_actual))
    for m_i, m in enumerate(m_list):

        group_index = group_index_from_feature_Kmeans(feature, m)
        des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
        xs_i = data[index, 1:]
        for i, group_i in enumerate(group_index):
            xs_reduction_multi[m_i][:, group_i] = np.tile(xs_i[:, i], (len(group_i), 1)).transpose()
    return A, xs_multi, xs_reduction_multi




def compare_xs_multi(xs_multi, xs_reduction_multi, weight, m_list, des):
    """TODO: Docstring for plot_error.

    :arg1: TODO
    :returns: TODO

    """

    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']
    markers = ['o', '*', '^', 'v', '>', '<']
    markersizes = [60, 80, 60, 60, 60, 60]
    for i, m in enumerate(m_list):
        xs_reduction_multi_plot = xs_reduction_multi[i]
        plt.scatter(xs_multi, xs_reduction_multi_plot, marker=markers[i], color=colors[i], alpha=0.6, label=f'$m={m}$', s=markersizes[i])
    plt.plot([0.8 * min(np.min(xs_multi), np.min(xs_reduction_multi) ) , max(np.max(xs_multi), np.max(xs_reduction_multi) ) * 1.1], [0.8 * min(np.min(xs_multi), np.min(xs_reduction_multi) ) , max(np.max(xs_multi), np.max(xs_reduction_multi) ) * 1.1], '--', linewidth = lw, color = 'tab:grey', alpha = alpha)
    plt.locator_params(nbins=5)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$x_s^{\\mathrm{multi}}$', fontsize=fs)
    plt.ylabel('$x_s^{\\mathrm{reduction}}$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_compare_xs_w={weight}.svg'
    plt.savefig(save_des, format='svg', transparent=True)
    plt.close()
    return None

def plot_yglobal_m(y_multi, y_reduction, weight_list, m_list, des):
    """TODO: Docstring for plot_xs_bifurcation_m.

    :arg1: TODO
    :returns: TODO

    """
    linestyles = ['solid', 'solid', 'dashed', 'dashdot', 'dashed', 'dashdot', 'dashed']

    fig, ax = plt.subplots()
    for i, m in enumerate(m_list):
        ax.plot(weight_list, y_reduction[i], linestyle=linestyles[i], linewidth=lw, alpha=alpha, label=f'm={m}')
    ax.plot(weight_list, y_multi, linestyle='dashdot', linewidth=lw, alpha=alpha, label='$m=N$')
    ax.set_yscale('symlog')
    plt.locator_params(axis='x', nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(np.hstack((0, np.logspace(0, np.ceil(np.log10(np.max(y_reduction))), int(np.ceil(np.log10(np.max(y_reduction)))), endpoint=False) )), fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$y^{(gl)}$', fontsize=fs)
    plt.legend(fontsize=15, frameon=False) 
    plt.tight_layout()
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_yglobal'
    plt.savefig(save_des + '.svg' , format='svg', transparent=True)
    plt.savefig(save_des + '.png' , format='png', transparent=True)
    plt.close()
    return None

def plot_yi_group(xs_reduction_multi, weight_list, m_list, des):
    """TODO: Docstring for plot_error.

    :arg1: TODO
    :returns: TODO

    """
    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']
    fig, ax = plt.subplots()
    for i, m in enumerate(m_list):
        xs_reduction_plot = xs_reduction_multi[i]
        y_i = np.vstack(([np.unique(xs_reduction_plot[i]) for i in range(len(weight_list))]))
        for j in range(m):
            """
            if j == 0:
                plt.plot(weight_list, y_i[:, j], linewidth=lw, alpha=alpha, label='$m=$' + f'{m}', color=colors[i])
            else:
                plt.plot(weight_list, y_i[:, j], linewidth=lw, alpha=alpha, color=colors[i])
            """
            ax.plot(weight_list, y_i[:, j], linewidth=lw, alpha=alpha, color=colors[j], label=f'group={j}')
    ax.set_yscale('symlog')

    plt.locator_params(axis='x', nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(np.hstack((0, np.logspace(0, np.ceil(np.log10(np.max(xs_reduction_multi))), int(np.ceil(np.log10(np.max(xs_reduction_multi)))), endpoint=False) )), fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$y^{(i)}$', fontsize=fs)
    plt.legend(fontsize=15, frameon=False) 
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_yi_m_{m_list}.svg'
    plt.savefig(save_des, format='svg', transparent=True)
    plt.close()
    return None

def plot_error_m(xs_multi, xs_reduction_multi, y_multi, y_reduction, weight_list, m_list, des):
    """TODO: Docstring for plot_error.

    :arg1: TODO
    :returns: TODO

    """
    if error_method == 'weighted':
        #error = np.abs(y_multi_weighted -y_reduction_weighted) / y_multi_weighted
        error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))
    elif error_method == 'unweighted':
        #error = np.abs(y_multi_unweighted -y_reduction_unweighted) / y_multi_unweighted
        error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))

    elif error_method == 'individual_unweighted':
        error = np.mean(np.abs( np.round(xs_reduction_multi - xs_multi, 12)) / ((np.abs(xs_multi) + np.abs(xs_reduction_multi))), 2)
    fig, ax = plt.subplots()
    for i, w in enumerate(weight_list):
        #plt.plot(m_list, error[:, i], '.-', linewidth=lw, alpha=alpha, label=f'$w={round(w, 4)}$')
        ax.plot(m_list, error[:, i], '.-', linewidth=lw, alpha=alpha, label=f'$w={round(w, 4)}$')
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    #plt.locator_params(axis='x', nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks([0, 0.1, 1], fontsize=ticksize)
    xaxis = plt.gca().xaxis
    xaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(2, 10)))
    #yaxis = plt.gca().yaxis
    #yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(2, 10)))
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('Error', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_y_error_' + error_method + '.svg'
    plt.savefig(save_des, format='svg', transparent=True)
    plt.close()
    return None

def heatmap_error_m(xs_multi, xs_reduction_multi, y_multi, y_reduction, weight_list, m_list, des):
    """TODO: Docstring for heatmap_error_m.

    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    "assume that ticks for m axis is 1, 10, 20, 30, 40, 50"
    m_ticks = [1, 10, 20, 30, 40, 50]
    if max(weight_list) == 0.4:
        weight_ticks = [0.01, 0.1, 0.2, 0.3, 0.4]
    elif max(weight_list) == 0.5:
        weight_ticks = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif max(weight_list) == 1:
        weight_ticks = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    elif max(weight_list) == 4:
        weight_ticks = [weight_list[0], 1, 2, 3, 4]


    if error_method == 'weighted':
        #error = np.abs(y_multi_weighted -y_reduction_weighted) / y_multi_weighted
        error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))
    elif error_method == 'unweighted':
        #error = np.abs(y_multi_unweighted -y_reduction_unweighted) / y_multi_unweighted
        error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))
    elif error_method == 'individual_unweighted':
        error = np.mean(np.abs(np.round(xs_reduction_multi - xs_multi, 12)) / ((np.abs(xs_multi) + np.abs(xs_reduction_multi))), 2)
    x_interval = int(len(weight_list) / 4)
    y_interval = int(len(m_list) / 4)
    ax = sns.heatmap(error[::-1], vmin=0, vmax=np.max(1), linewidths=0, yticklabels=m_ticks[::-1], xticklabels=weight_ticks)
    ax.set_yticks(np.array(m_ticks)-0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)

    cax = plt.gcf().axes[-1]
    ax.tick_params(labelsize=0.7 * fs)
    cax.tick_params(labelsize=0.7 * fs)
    ax.collections[0].colorbar.set_label("Error", fontsize=fs*0.8)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$m$', fontsize=fs)
    plt.tight_layout()
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_heatmap_error_' + error_method 
    plt.savefig(save_des + '.svg', format='svg', transparent=True)
    plt.savefig(save_des + '.png', format='png', transparent=True)
    plt.close()


def plot_y_m(data, weight_list, m_list, des):
    """TODO: Docstring for plot_error.

    :arg1: TODO
    :returns: TODO

    """

    fig, ax = plt.subplots()
    for i, w in enumerate(weight_list):
        ax.loglog(m_list, data[:, i], '.-', linewidth=lw, alpha=alpha, label=f'$w={round(w, 4)}$')
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    #plt.locator_params(axis='x', nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks([0, 1], fontsize=ticksize)
    xaxis = plt.gca().xaxis
    xaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(2, 10)))
    #yaxis = plt.gca().yaxis
    #yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=np.arange(2, 10)))
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$m$', fontsize=fs)
    plt.ylabel('$y^{(gl)}$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    plt.tight_layout()
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_y_' + method_weight
    plt.savefig(save_des +  '.svg', format='svg', transparent=True)
    plt.savefig(save_des +  '.png', format='png', transparent=True)
    plt.close()
    return None

def heatmap_y_m(data, weight_list, m_list, des):
    """TODO: Docstring for heatmap_error_m.

    :network_type: TODO
    :: TODO
    :returns: TODO

    """

    "assume that ticks for m axis is 1, 10, 20, 30, 40, 50"
    m_ticks = [1, 10, 20, 30, 40, 50]
    if max(weight_list) == 0.4:
        weight_ticks = [0.01, 0.1, 0.2, 0.3, 0.4]
    elif max(weight_list) == 0.5:
        weight_ticks = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif max(weight_list) == 1:
        weight_ticks = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    elif max(weight_list) == 4:
        weight_ticks = [weight_list[0], 1, 2, 3, 4]

    #x_interval = int(len(weight_list) / 4)
    #y_interval = int(len(m_list) / 4)
    #ax = sns.heatmap(data[::-1], vmin=np.min(data), vmax=np.max(data), linewidths=0, yticklabels=m_list[::-1][::y_interval], xticklabels=weight_list[::x_interval]))
    data = data.copy()
    data[data<1e-5] = 1e-1
    ax = sns.heatmap(data[::-1], vmin=10 ** (np.floor(np.log10(max(0.01, np.min(data))))), vmax=10 ** (np.ceil(np.log10(np.max(data)))), linewidths=0, yticklabels=m_ticks[::-1], xticklabels=weight_ticks, norm=mpl.colors.LogNorm(), cmap='coolwarm')
    ax.set_yticks(np.array(m_ticks)-0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    cax = plt.gcf().axes[-1]
    ax.tick_params(labelsize=0.7 * fs)
    cax.tick_params(labelsize=0.7 * fs)
    cax.minorticks_off()
    ax.collections[0].colorbar.set_label("$y^{(\\mathrm{gl})}$", fontsize=fs*0.8)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$m$', fontsize=fs)
    plt.tight_layout()
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_heatmap_y_' + method_weight
    plt.savefig(save_des  + '.svg', format='svg', transparent=True)
    plt.savefig(save_des  + '.png', format='png', transparent=True)
    plt.close()

def critical_m_reduction(data, weight_list, m_list, des):
    """TODO: Docstring for plot_critical_m.

    :arg1: TODO
    :returns: TODO

    """

    diff = np.abs(np.diff(data, axis = 0)) / data[:-1]
    argmax = np.argmax(diff, axis=0)
    m_critical = np.zeros((len(argmax)))
    for i, argmax_i in enumerate(argmax):
        diff_critical = np.abs(np.round(np.abs(data[argmax_i + 1, i] - data[argmax_i, i]), 5)) / ( np.abs(data[argmax_i + 1, i]) + np.abs(data[argmax_i, i]) ) 
        if diff_critical > 0.5:
            m_critical[i] = m_list[argmax_i] + 1
        else:
            m_critical[i] = 1

    fig, ax = plt.subplots()
    ax.plot(weight_list, m_critical, linewidth=lw, alpha=alpha)
    plt.ylim(0, max(m_list))

    plt.locator_params(axis='x', nbins=4)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$m^{opt}$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_mcritical_' + method_weight + '.svg'
    plt.savefig(save_des, format='svg', transparent=True)
    plt.close()
    return None

def critical_m_two(y_reduction, y_multi, weight_list, m_list, original_threshold, reduction_threshold, des):
    """TODO: Docstring for plot_critical_m.

    :arg1: TODO
    :returns: TODO

    """
    m_critical_reduction = np.zeros((len(weight_list)))
    m_critical_multi = np.zeros((len(weight_list)))

    y_reduction_diff = np.abs(np.diff(y_reduction, axis = 0)) 
    error = np.abs(np.round(y_reduction[1:] - y_reduction[:-1], 5)) / (y_reduction[1:] + y_reduction[:-1])
    for i, w in enumerate(weight_list):
        index = np.where(error[:, i] > reduction_threshold)[0]
        if len(index):
            m_critical_reduction[i] = m_list[index[0]+1]
        else:
            m_critical_reduction[i] = m_list[0]

    """
    argmax = np.argmax(y_reduction_diff, axis=0)
    for i, argmax_i in enumerate(argmax):
        diff_critical = np.abs(np.round(np.abs(y_reduction[argmax_i + 1, i] - y_reduction[argmax_i, i]), 5)) / ( np.abs(y_reduction[argmax_i + 1, i]) + np.abs(y_reduction[argmax_i, i]) ) 
        if diff_critical > 0.5:
            m_critical_reduction[i] = m_list[argmax_i] + 1
        else:
            m_critical_reduction[i] = 1
    """

    error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))
    for i, w in enumerate(weight_list):
        index = np.where(error[:, i] < original_threshold)[0]
        if len(index):
            m_critical_multi[i] = m_list[index[0]]
        else:
            m_critical_multi[i] = m_list[-1]

    fig, ax = plt.subplots()
    ax.plot(weight_list, m_critical_reduction, linewidth=lw, alpha=alpha, label='reduction', color='#8da0cb')
    ax.plot(weight_list, m_critical_multi, linewidth=lw, alpha=alpha, label='original', linestyle='--', color='#e5c494')
    #plt.text(0.37 * weight_list[-1], 35, '$\\Delta y^{(c)}=$' + f'{reduction_threshold}' + '_$\\mathrm{error}^{(c)}=$' + f'{original_threshold}' , fontsize=fs*0.7)
    plt.text(0.6 * weight_list[-1], 35, '$\\epsilon_1=$' + f'{original_threshold}' + '_$\\epsilon_2=$' + f'{reduction_threshold}', fontsize=fs*0.7)
    plt.ylim(0, max(m_list))
    plt.locator_params(axis='x', nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$m^{opt}$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_mcritical_two_orithresh={original_threshold}_redthresh={reduction_threshold}' + method_weight 
    plt.savefig(save_des + '.png', format='png', transparent=True)
    plt.close()
    return None

def critical_m_multi_threshold(y_reduction, y_multi, weight_list, m_list, original_threshold_list, reduction_threshold_list, des):
    """TODO: Docstring for plot_critical_m.

    :arg1: TODO
    :returns: TODO

    """
    m_critical_reduction_list = []
    m_critical_multi_list = []
    for original_threshold in original_threshold_list:
        for reduction_threshold in reduction_threshold_list:
            m_critical_reduction = np.zeros((len(weight_list)))
            m_critical_multi = np.zeros((len(weight_list)))

            y_reduction_diff = np.abs(np.diff(y_reduction, axis = 0)) 
            error = np.abs(np.round(y_reduction[1:] - y_reduction[:-1], 5)) / (y_reduction[1:] + y_reduction[:-1])
            for i, w in enumerate(weight_list):
                index = np.where(error[:, i] > reduction_threshold)[0]
                if len(index):
                    m_critical_reduction[i] = m_list[index[0]+1]
                else:
                    m_critical_reduction[i] = m_list[0]

            error = np.abs(np.round(y_multi -y_reduction, 12)) / (np.abs(y_multi) + np.abs(y_reduction))
            for i, w in enumerate(weight_list):
                index = np.where(error[:, i] < original_threshold)[0]
                if len(index):
                    m_critical_multi[i] = m_list[index[0]]
                else:
                    m_critical_multi[i] = m_list[-1]
            m_critical_reduction_list.append(m_critical_reduction)
            m_critical_multi_list.append(m_critical_multi)

    m_critical_reduction_list = np.vstack((m_critical_reduction_list))
    m_critical_multi_list = np.vstack((m_critical_multi_list))
    fig, ax = plt.subplots()
    m1 = ax.scatter(np.tile(weight_list, (len(original_threshold_list) * len(reduction_threshold_list), 1)).flatten(), m_critical_reduction_list.flatten(), marker='o', s=60, alpha=0.5, color='#8da0cb')
    m2 = ax.scatter(np.tile(weight_list, (len(original_threshold_list) * len(reduction_threshold_list), 1)).flatten(), m_critical_multi_list.flatten(), marker='*', s=70, alpha=0.5, color='#e5c494')
    l1, =ax.plot(weight_list, np.mean(m_critical_reduction_list, 0), linestyle=(0, (3, 2)), linewidth=lw, alpha=alpha, color='#8da0cb')
    l2, = ax.plot(weight_list, np.mean(m_critical_multi_list, 0), linestyle=(0, (3, 2, 1, 2)), linewidth=lw, alpha=alpha, color='#e5c494')
    plt.ylim(0, max(m_list)+1)
    plt.locator_params(axis='x', nbins=4)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$w$', fontsize=fs)
    plt.ylabel('$m^{opt}$', fontsize=fs)
    plt.legend( [(l1, m1), (l2, m2)], ['reduction', 'original'], fontsize=17, frameon=False) 
    plt.tight_layout()
    save_des = des + dynamics + '_' + network_type + '_' + method + f'_N={N}_d={d}_seed={seed}_mcritical_two_multithresholds' + method_weight 
    plt.savefig(save_des + '.svg', format='svg', transparent=True)
    plt.savefig(save_des + '.png', format='png', transparent=True)
    plt.close()
    return None

def critical_m_threshold(y_reduction, y_multi, weight_list, m_list, original_threshold_list, reduction_threshold_list, des):
    """TODO: Docstring for critical_m_threshold.

    :y_reduction: TODO
    :y_multi: TODO
    :weight_list: TODO
    :m_list: TODO
    :original_threshold_list: TODO
    :reduction_threshold_list: TODO
    :des: TODO
    :returns: TODO

    """
    for original_threshold in original_threshold_list:
        for reduction_threshold in reduction_threshold_list:
            critical_m_two(y_reduction, y_multi, weight_list, m_list, original_threshold, reduction_threshold, des)


def figure_plot(network_type, N, seed, d, weight_list, m_list, compare_weight, error_weight_list, compare_m_list, yi_m, space, tradeoff_para, method, method_weight, error_method, original_threshold_list, reduction_threshold_list, original_threshold_si, reduction_threshold_si):
    """TODO: Docstring for figure_plot.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '/' 
    des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres=20/' 
    if not os.path.exists(des):
        os.makedirs(des)
    A, xs_multi, xs_reduction_multi = data_xs(network_type, N, seed, d, weight_list, m_list, space, tradeoff_para, method)
    y_reduction_weighted = np.vstack(([np.array([betaspace(A, xs_reduction_multi[m, k])[-1] for k in range(len(weight_list))]) for m in range(len(m_list))]))
    y_reduction_unweighted = np.vstack(( [np.array([np.mean(xs_reduction_multi[m, k]) for k in range(len(weight_list))]) for m in range(len(m_list))] ))
    y_multi_weighted = np.array([betaspace(A, xs_multi[i])[-1] for i in range(len(weight_list))])
    y_multi_unweighted = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])

    if method_weight == 'weighted':
        y_reduction = y_reduction_weighted
        y_multi = y_multi_weighted
    elif method_weight == 'unweighted':
        y_reduction = y_reduction_unweighted
        y_multi = y_multi_unweighted

    """
    "Fa"
    compare_weight_index = np.where(np.abs(compare_weight - weight_list) < 1e-8)[0][0]
    compare_m_index = [np.where(np.abs(m_list - m_i) < 1e-8)[0][0] for m_i in compare_m_list]
    compare_xs_multi(xs_multi[compare_weight_index], xs_reduction_multi[compare_m_index, compare_weight_index], compare_weight, compare_m_list, des)

    "Fb"
    plot_yglobal_m(y_multi, y_reduction[compare_m_index], weight_list, compare_m_list, des)

    "Fc"
    yi_m_index = [np.where(np.abs(m_list - yi_m[0]) < 1e-8)[0][0] ]
    plot_yi_group(xs_reduction_multi[yi_m_index], weight_list, yi_m, des)

    "Fd"
    error_weight_index = [np.where(np.abs(w_i - weight_list) < 1e-8)[0][0] for w_i in error_weight_list]
    plot_error_m(xs_multi[error_weight_index], xs_reduction_multi[:, error_weight_index], y_multi[error_weight_index], y_reduction[:, error_weight_index], error_weight_list, m_list, des)

    "Fe"
    heatmap_error_m(xs_multi, xs_reduction_multi, y_multi, y_reduction, weight_list, m_list, des)

    "Ff"
    heatmap_y_m(y_reduction, weight_list, m_list, des)
    "Fg"
    plot_y_m(y_reduction[:, error_weight_index], weight_list[error_weight_index], m_list, des)
    "Fh"
    critical_m_reduction(y_reduction, weight_list, m_list, des)

    "Fi"
    critical_m_multi_threshold(y_reduction, y_multi, weight_list, m_list, original_threshold_list, reduction_threshold_list, des)
    """

    "Fj"
    critical_m_threshold(y_reduction, y_multi, weight_list, m_list, original_threshold_si, reduction_threshold_si, des)


    return None


def tippingpoint_m(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, high_criteria):
    """TODO: Docstring for one_dimension_comparison.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :d: TODO
    :: TODO
    :returns: TODO

    """
    A, xs_multi, xs_reduction_multi = data_xs(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method)

    y_reduction_weighted = np.vstack(([np.array([betaspace(A, xs_reduction_multi[m, k])[-1] for k in range(len(weight_list))]) for m in range(len(m_list))]))
    y_reduction_unweighted = np.vstack(( [np.array([np.mean(xs_reduction_multi[m, k]) for k in range(len(weight_list))]) for m in range(len(m_list))] ))
    y_multi_weighted = np.array([betaspace(A, xs_multi[i])[-1] for i in range(len(weight_list))])
    y_multi_unweighted = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])

    wc_multi = weight_list[np.where(y_multi_weighted > high_criteria)[0][0]]
    wc_reduction = [weight_list[np.where(y_reduction_weighted[i] > high_criteria)[0][0]] for i in range(len(m_list))]
    #plt.loglog(m_list, (wc_reduction-wc_multi))
    #plt.loglog(m_list, (wc_reduction-wc_multi)/(wc_reduction[0] - wc_multi))
    w_compare_list = np.arange(0.1, 0.41, 0.05)
    for w_compare in w_compare_list:
        index = np.where(np.abs(weight_list - w_compare)<1e-8)[0][0]
        y_reduction_weighted_select = y_reduction_weighted[:, index]
        y_multi_weighted_select = y_multi_weighted[index]
        plt.loglog(m_list[1:], y_reduction_weighted_select[1:] - y_reduction_weighted_select[:-1], label=f'w={w_compare}')
    plt.legend()
    plt.show()

def degree_dis(network_type, N, seed, d):
    """TODO: Docstring for one_dimension_comparison.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :d: TODO
    :: TODO
    :returns: TODO

    """

    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    k = np.sum(A_unit, 0)
    bins = np.linspace(k.min(), k.max(), 20)
    plt.hist(k, bins)
    k_prob, bins = np.histogram(k, bins)
    #plt.loglog((bins[:-1] + bins[1:]) / 2, k_prob)

    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel('$k$', fontsize=fs)
    plt.ylabel('$N(k)$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False) 
    save_des = '../manuscript/dimension_reduction_v1_111021/genereg/' + network_type + '_' + f'_N={N}_d={d}_seed={seed}_p_k.svg'
    plt.savefig(save_des, format='svg')
    plt.close()
    return None


dynamics = 'mutual'

dynamics = 'genereg'

dynamics = 'CW'

dynamics = 'CW_high'



network_type = 'ER'
N = 1000
d = 2000
seed = 0
d_list = [4000, 8000]
seed_list = [0, 1]
space = 'linear'

network_type = 'SF'
N = 1000
seed = [0, 0]
d = [2.5, 999, 3]
seed_list = [[i, i] for i in range(0, 1)]
d_list = [[2.1, 999, 2]]
space = 'log'

network_type = 'SBM_ER'
N = [100, 100, 100]
d_list = [np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()]
d_list = [np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist()]
d_list = [np.array([[0.2, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.05]]).tolist(), np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist(), np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()]
seed = 0
seed_list = [0, 1]
space = 'linear'


#network_partition(network_type, N, seed, d, beta, betaeffect, number_groups, space)
m_list = np.arange(1, 50, 1)
weight_list = np.round(np.arange(0.01, 0.5, 0.01), 5)
tradeoff_para = 0.5
method = 'degree'
error_method = 'weighted'
method_weight = 'weighted'






network_type = 'ER'
N = 1000
d = 8000
seed = 0
space = 'linear'

network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.2, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.05]]).tolist()
d = np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist()
d = np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()
seed = 1





"real networks"
network_type = 'real'
d = 0

N = 91
seed = 5
d = 0
space = 'linear'
weight_list = np.round(np.arange(0.01, 0.2, 0.01), 5)
error_weight_list = [0.12, 0.13, 0.14, 0.15]
compare_weight = 0.13


N = 456
seed = 11
d = 0
space = 'linear'
weight_list = np.sort(np.hstack(( np.round(np.arange(0.01, 0.03, 0.01), 5), np.round(np.arange(0.016, 0.017, 0.0001), 5) )))
error_weight_list = [0.0165, 0.0166, 0.0167, 0.0168]
compare_weight = 0.0168

N = 85
seed = 7
space = 'linear'
weight_list = np.round(np.arange(0.01, 0.2, 0.01), 5)
error_weight_list = [0.15, 0.16, 0.17, 0.18]
compare_weight = 0.16

N = 270
seed = 6
d = 0
space = 'log'
weight_list = np.round(np.arange(0.01, 0.2, 0.01), 5)
error_weight_list = [0.05, 0.06, 0.07, 0.08]
compare_weight = 0.06

N = 97
seed = 8
d = 0
space = 'log'
weight_list = np.round(np.arange(0.01, 0.3, 0.01), 5)
error_weight_list = [0.15, 0.16, 0.17, 0.18]
compare_weight = 0.18

N = 1044
seed = 12
d = 0
space = 'log'
weight_list = np.round(np.arange(0.01, 0.03, 0.001), 5)
error_weight_list = [0.016, 0.017, 0.018, 0.019]
compare_weight = 0.018


"SBM ER networks"
compare_m_list = [1, 3, 10]
yi_m = [3]
network_type = 'SBM_ER'
N = [100, 100, 100]
d = np.array([[0.2, 0.001, 0.001], [0.001, 0.1, 0.001], [0.001, 0.001, 0.05]]).tolist()
d = np.array([[0.2, 0.005, 0.005], [0.005, 0.1, 0.005], [0.005, 0.005, 0.05]]).tolist()
d = np.array([[0.2, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.05]]).tolist()
seed = 0
seed = 1
space = 'linear'
weight_list = np.round(np.arange(0.01, 0.5, 0.01), 5)
error_weight_list = [0.3, 0.32, 0.34, 0.36]
compare_weight = 0.34


"SF networks"
compare_m_list = [1, 5, 10]
yi_m = [5]

network_type = 'SF'
space = 'log'
N = 20
d = [2.1, 19, 2]
seed = [0, 0]
m_list = np.arange(1, 20, 1)
weight_list = np.round(np.arange(0.01, 1.5, 0.01), 5)
error_weight_list = [1, 1.1, 1.2, 1.3]
compare_weight = 1.2


"SF networks with beta preserved"
network_type = 'SF'
space = 'log'
N = 1000
m_list = np.arange(1, 101, 1)
m_list = np.arange(1, 51, 1)
weight_list = np.round(np.arange(0.02, 1.01, 0.02), 5)
compare_m_list = [1, 2, 3, 10] 
compare_m_list = [1, 5, 10] 
compare_weight = 0.2

dynamics = 'CW'
weight_list = np.round(np.arange(0.1, 4.01, 0.1), 5)
error_weight_list = [1, 1.5, 2, 2.5]


dynamics = 'CW_high'
weight_list = np.round(np.arange(0.02, 1.01, 0.02), 5)
error_weight_list = [0.3, 0.5, 0.7, 0.9]

dynamics = 'genereg'
dynamics = 'mutual'
weight_list = np.round(np.arange(0.01, 0.401, 0.01), 5)
error_weight_list = [0.15, 0.2, 0.25, 0.3]


original_threshold_si = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
reduction_threshold_si = [0.5, 0.6, 0.7, 0.8, 0.9]

original_threshold_list = np.arange(0.2, 0.4, 0.02)
reduction_threshold_list = np.arange(0.5, 0.9, 0.05)


beta_pres = 20
des = '../data/beta_pres_networks/'   
des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
index = np.arange(75)
index = [6]
index = [6, 17, 22, 43, 60, 73]
net_data = np.array(pd.read_csv(des_file, header=None).iloc[index, :])
gamma_list, seed_list, kmin_list, kmax_list, kmean_list, h1_list, h2_list, beta_cal_list = net_data.transpose()

for gamma, kmin, seed in zip(gamma_list, kmin_list, seed_list):
    d = [gamma, N-1, int(kmin)]
    seed = [int(seed), int(seed)]
    figure_plot(network_type, N, seed, d, weight_list, m_list, compare_weight, error_weight_list, compare_m_list, yi_m, space, tradeoff_para, method, method_weight, error_method, original_threshold_list, reduction_threshold_list, original_threshold_si, reduction_threshold_si)



"""
N = 1000
method_weight = 'weighted'

network_type = 'SF'
d = [2.5, 999, 3]
seed = [1, 1]
space = 'log'

y_weight_list = [0.2, 0.25, 0.3, 0.35]
y_weight_list = [0.06, 0.08, 0.1, 0.12]
y_weight_list = [0.04, 0.06, 0.08, 0.1]
y_weight_list = [0.1, 0.12, 0.14, 0.18]

network_type = 'ER'
d = 8000
seed = 0
space = 'linear'

weight_list = np.round( np.arange(0.01, 1.5, 0.01), 4)
y_weight_list = [0.9, 1, 1.1, 1.2]
y_weight_list = [0.6, 0.65, 0.7, 0.75]
y_weight_list = [0.36, 0.38, 0.4, 0.42]


#degree_dis(network_type, N, seed, d)

high_criteria = 2.5
m_list = np.arange(1, 20, 1)
#tippingpoint_m(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, high_criteria)
"""
