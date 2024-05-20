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
h = 2
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


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def draw_brace(ax, xspan, color, linewidth, text, rotation=False):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution//2+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.5*y - .45)*yspan # adjust vertical position

    ax.autoscale(False)
    if rotation == False:
        ax.plot(x, y, color=color, lw=linewidth)
        ax.text((xmax+xmin)/2., ymin+.07*yspan, text, ha='center', va='bottom')
    else:
        ax.plot(y, x, color=color, lw=linewidth)
        ax.text(( ymin+.07*yspan, xmax+xmin)/2., text, ha='center', va='bottom')



def plot_xs_weight(ax, xlabel, ylabel, A, m, weight, xs, color, linewidth, alpha, color_bias, title_letter, xs_multi=None, A_unit=None, group_index=None):
    """TODO: Docstring for plot_xs_weight.

    :data: TODO
    :returns: TODO

    """
    ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.7)
    k = np.sum(A, 0)
    cmap = sns.color_palette(color, as_cmap=True)
    simpleaxis(ax)


    for (i, xs_i), k_i in zip(enumerate(xs.transpose()), k):
        color_i = np.log(k_i / k.max() + 1)
        color_i = k_i / k.max() + color_bias
        ax.plot(weight, xs_i, linewidth=linewidth, alpha=alpha, color=cmap(color_i))

    if xs_multi is not None:
        x_eff = np.zeros((len(weight), m ))
        for i, xs_multi_i in enumerate(xs_multi):
            _, _, x_eff_i = reducednet_effstate(A_unit, xs_multi_i, group_index)
            x_eff[i] = x_eff_i
        ax.plot(weight, x_eff, linewidth=linewidth*0.8, alpha=0.7, color='tab:grey')


    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)

def plot_ygl_weight(ax, xlabel, ylabel, A, m, weight, xs, color, linewidth, alpha, color_bias, title_letter, xs_multi=None, A_unit=None, group_index=None):
    """TODO: Docstring for plot_xs_weight.

    :data: TODO
    :returns: TODO

    """
    ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.7)
    k = np.sum(A, 0)
    cmap = sns.color_palette(color, as_cmap=True)
    simpleaxis(ax)

    xs_group = np.zeros(( len(weight), len(A_unit)) )
    for j, group_i in enumerate(group_index):
        xs_group[:, group_i] = xs[:, j:j+1]
    ygl_group = betaspace(A_unit, xs_group)[-1]
    ax.plot(weight, ygl_group, linewidth=linewidth, alpha=alpha, color=cmap(1.0))
        #ax.plot(weight, ygl_group, linewidth=linewidth, alpha=0.8, color=cmap(0))
        #ax.plot(weight, ygl_group, linewidth=linewidth, alpha=0.8, color=cmap(0.5))

    if xs_multi is not None:
        ygl = np.zeros((len(weight) ))
        for i, xs_multi_i in enumerate(xs_multi):
            ygl_i = betaspace(A_unit, xs_multi_i)[-1]
            ygl[i] = ygl_i
        ax.plot(weight, ygl, linewidth=linewidth*0.8, alpha=0.7, color='tab:grey')


    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)

def plot_network_topology(ax, network_type, N, A_unit, color, color_bias, node_size_bias, title_letter):
    """TODO: Docstring for plot_network_topology.

    :ax: TODO
    :A_unit: TODO
    :returns: TODO

    """
    simpleaxis(ax)
    ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.7)
    cmap = sns.color_palette(color, as_cmap=True)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)

    if network_type == 'SF':
        edge_width = 1.0
        node_size = k ** 0.25  * 2.1
    elif network_type == 'ER':
        edge_width = 0.2
        node_size = k ** 0.2  * 2
    else:
        node_size = k ** 0.2  * 2
        edge_width = 2.5
    node_size_dict = {u:v for u, v in zip(np.arange(len(k)), node_size)}
    node_color_dict = {u: cmap(k_i / k.max()+color_bias ) for u, k_i in zip(np.arange(len(k)), k)}
    G = nx.from_numpy_array(A_unit)
    if network_type == 'SBM_ER':
        space = 'linear'
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        group_index = group_index_from_feature_Kmeans(feature, len(N))
        groups_dict = {node_i: group_i for group_i, nodes in enumerate(group_index) for node_i in nodes}
        g = Graph(A_unit, edge_width=edge_width, arrows=True, node_size=node_size_dict, edge_layout='straight', node_layout='community', node_layout_kwargs={'node_to_community':groups_dict}, node_color=node_color_dict, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.8, ax=ax, edge_zorder=1)
    else:
        g = Graph(A_unit, edge_width=edge_width, arrows=True, node_size=node_size_dict, edge_layout='straight', node_layout=nx.spring_layout(G), node_color=node_color_dict, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.8, ax=ax, edge_zorder=1)
        #nx.draw(G, pos=nx.spring_layout(G), node_size=node_size, alpha=0.8, node_color=node_color, edge_color='tab:grey', width=edge_width, ax=ax)

def plot_reduced_network_topology(ax, network_type, A_reduced, m, color, color_bias, node_size_bias, title_letter):
    """TODO: Docstring for plot_network_topology.

    :ax: TODO
    :A_unit: TODO
    :returns: TODO

    """
    ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.7)
    cmap = sns.color_palette(color, as_cmap=True)

    G = nx.from_numpy_array(A_reduced)
    k = np.sum(A_reduced, 1)

    node_size = np.log(k+node_size_bias) 
    node_size_dict = {u:v for u, v in zip(np.arange(len(k)), node_size)}
    node_color_dict = {u: cmap(k_i / k.max()+color_bias ) for u, k_i in zip(np.arange(len(k)), k)}
    edgelists = [ (ni, nj) for ni in range(m) for nj in range(m) if A_reduced[ni, nj]]
    if m == 1:
        node_layout = {0:(0.7, 0.5)}
        node_size_dict = {0:15}
        radius = 0.2
        edge_width = 5
        node_edge_width=3
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        node_layout = 'spring'
        node_sizes = np.log(k + 1) / m**0.3 * 2.5
        node_size_dict=  {u: node_sizes[u] for u in np.arange(len(k))}
        radius = 0.07 / m ** 0.25
        widths = np.log(A_reduced / A_reduced.max() + 1.3)  * 2
        widths = np.log(A_reduced / A_reduced.max() + 1) * 80 / m**1.5
        edge_width = {(u, v) :  widths[v, u] * 1 if u == v else widths[v, u] for (u, v) in edgelists}
        node_edge_width = 1


    if  m == 3:
        node_layout = {0:(0.3, 0.2), 1:(0.7, 0.2), 2:(0.5, 0.6)}
        node_layout ={key: (x* (np.random.random()*0.1+1), y* (np.random.random()*0.1+1) ) for key, (x, y) in node_layout.items()}
        node_sizes = np.log(k+1) * 2
        node_size_dict=  {u: node_sizes[u] for u in np.arange(len(k))}
        radius = 0.08
        widths = np.log(A_reduced / A_reduced.max() + 1.5) * 10
        edge_width = {(u, v) :  widths[v, u] * 0.5 if u == v else widths[v, u] for (u, v) in edgelists}

    elif  m == 5:
        if network_type == 'SF':
            node_layout = {0:(0.5, 0.45), 1:(0.2, 0.25), 2:(0.6, 0.2), 3:(0.6, 0.75), 4:(0.3, 0.55)}
        else:
            node_layout = {0:(0.3, 0.4), 1:(0.7, 0.4), 2:(0.5, 0.6), 3:(0.4, 0.2), 4:(0.6, 0.2)}
        node_layout ={key: (x* (np.random.random()*0.1+1), y* (np.random.random()*0.1+1) ) for key, (x, y) in node_layout.items()}
        #node_layout=nx.spring_layout(G)
        node_sizes = np.log(k+3) * 1.8
        node_size_dict=  {u: node_sizes[u] for u in np.arange(len(k))}
        radius = 0.06
        widths = np.log(A_reduced / A_reduced.max() + 1.3) * 5
        edge_width = {(u, v) :  widths[v, u] * 0.5 if u == v else widths[v, u] for (u, v) in edgelists}

    g = Graph(edgelists, edge_width=edge_width, arrows=True, node_layout = node_layout, node_size=node_size_dict, edge_layout='straight', node_color=node_color_dict, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.5, ax=ax, edge_layout_kwargs={'selfloop_radius': radius})
    xy_position = np.vstack((g.node_positions.values() )) 
    x_max, x_min = xy_position[:, 0].max() + radius*2, xy_position[:, 0].min() - radius*2.0
    y_max, y_min = xy_position[:, 1].max() + radius*2, xy_position[:, 1].min() - radius*2.0
    #y_max, y_min = xy_position[:, 1].max() + (0.3 * xy_position[:, 1].mean()), xy_position[:, 1].min() - (0.3 * xy_position[:, 1].mean())


def read_xs(network_type, N, space, d, seed, m, weight_list):
    """TODO: Docstring for read_xs.

    :network_type: TODO
    :N: TODO
    :space: TODO
    :d: TODO
    :seed: TODO
    :m: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/'
    if m == N:
        xs_multi_file = des + 'xs_multi/' + f'N={N}_d={d}_seed={seed}.csv'
        data_multi = np.array(pd.read_csv(xs_multi_file, header=None))
        weights_multi = data_multi[:, 0]
        index = [np.where(np.abs(weights_multi - weight) < 1e-05 )[0][0]  for weight in weight_list]
        xs = data_multi[index, 1:]
        #y = betaspace(A_unit, xs)[-1]
    else:
        xs_group_file = des + f'degree_kmeans_space={space}/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data_group = np.array(pd.read_csv(xs_group_file, header=None))
        weights_group = data_group[:, 0]
        index = [np.where(np.abs(weights_group - weight) < 1e-05 )[0][0]  for weight in weight_list]
        xs = data_group[index, 1:]
        #y = betaspace(A, xs)[-1]
    return xs



def main_fig(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics):
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
    fig = plt.figure(figsize=(13, 20))
    gs = mpl.gridspec.GridSpec(nrows=8, ncols=9, height_ratios=[0.9, 1, 1, 1, 1.15, 1, 1, 1], width_ratios=[1.05, 0.25, 0.75, 0.10, 0.75, 0.25, 0.75, 0.85, 1])
    #plt.rcParams.update({"text.usetex": True,})
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    dxdt = r'$\frac{dx_i}{dt} = F(x_i) + w \sum_{j=1}^{N} A_{ij} G(x_i, x_j)$'
    t = ax.text(0.3, 0.7, dxdt, ha="center", va="center", rotation=0, size=15, bbox=dict(boxstyle="round,pad=0.3", fc="tab:grey", ec="k", lw=1, alpha=0.5))

    dxdt = r'$\frac{dx}{dt} = F(x) + w \beta G(x, x)$'
    t = ax.text(0.91, 0.7, dxdt, ha="center", va="center", rotation=0, size=15, bbox=dict(boxstyle="round,pad=0.3", fc="tab:grey", ec="k", lw=1, alpha=0.5))
    ax.annotate( 'One-dimensional Reduction', xy = (1.2, 0.8),  xytext=(0.55, 0.7), xycoords='axes fraction', fontsize=14, color='tab:grey', weight='bold')


    #plt.rcParams.update({"text.usetex": False,})
    for (i, network_type), N, d, seed, space in zip(enumerate(network_type_list), N_list, d_list, seed_list, space_list):
        if network_type == 'ER':
            color_bias = 0
            node_size_bias = 1
        else:
            color_bias = 0.1
            node_size_bias = 5


        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        if i == 2:
            xlabel = '$w$'
            xk_label = '$k$'
        else:
            xlabel = ''
            xk_label = ''
        if i == 1:
            ylabel = '$y^{(\\mathrm{gl})}_s$'
            ylabel_xs= '$x_s$'
            yk_label = '$P(k)$'
        else:
            ylabel_xs = ''
            ylabel = ''
            yk_label = ''

        ax = fig.add_subplot(gs[i+1, 0:2])
        ax.annotate('$A_{ij}$', xy = (-0.7, 0.5),  xytext=(0.9, 0.9), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
        title_letter = f'({letters[i+0]}1)'
        title_letter = f'({letters[0]}{i+1})'
        plot_network_topology(ax, network_type, N, A_unit, colors[i], color_bias, node_size_bias, title_letter)
        ax.annotate( network_type, xy = (-0.7, 0.5),  xytext=(-0.85 - 0.08* len(network_type) , 0.45), xycoords='axes fraction', fontsize=15, color=sns.color_palette(colors[i])[-1], alpha=.5, weight='bold')
        if i == 0:
            ax.annotate('Topology', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')

        xs_multi = read_xs(network_type, N, space, d, seed, N, weight_list)
        ax = fig.add_subplot(gs[i+1, 2])
        title_letter = f'({letters[i+0]}2)'
        title_letter = f'({letters[1]}{i+1})'
        ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.8)
        simpleaxis(ax)
        k = np.sum(A_unit >0 , 0)
        if network_type == 'SF':
            sns.histplot(k, bins=20, stat='density', ax=ax, color = sns.color_palette(colors[i])[-1], alpha=0.5)
            ax.set_yscale('log')
        else:
            sns.histplot(k, bins=20, stat='density', ax=ax, color = sns.color_palette(colors[i])[-1], alpha=0.5)
        ax.set_xlabel(xk_label, fontsize=labelsize)
        ax.set_ylabel(yk_label, fontsize=labelsize)

        ax = fig.add_subplot(gs[i+1, 4:6])
        title_letter = f'({letters[i+0]}3)'
        title_letter = f'({letters[2]}{i+1})'
        linewidth = 2
        alpha = 0.8
        plot_xs_weight(ax, xlabel, ylabel_xs, A_unit, len(A_unit), weight_list, xs_multi, colors[i], linewidth, alpha, color_bias, title_letter)
        if i == 0:
            ax.annotate('Dynamics', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')

        ax = fig.add_subplot(gs[i+1, 6])
        ax.set_axis_off()
        ax.annotate(' ' * 10 , xy=(0.4, 0.50), xytext=(0.75, 0.5), xycoords='axes fraction', ha='center', va='bottom', bbox=dict(boxstyle='rarrow, pad=0.6', fc= sns.color_palette(colors[i])[-1] , ec='k', lw=2, alpha=0.5) )

        ax = fig.add_subplot(gs[i+1, 7])
        title_letter = f'({letters[i+0]}4)'
        title_letter = f'({letters[3]}{i+1})'
        m = 1
        xs_group = read_xs(network_type, N, space, d, seed, m, weight_list)
        group_index = group_index_from_feature_Kmeans(feature, m)
        A_reduced, _, _ = reducednet_effstate(A_unit, xs_multi[0], group_index)
        ax.annotate('$\\beta$', xy = (-0.7, 0.5),  xytext=(1, 1), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
        if i == 0:
            ax.annotate('Topology', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')
        plot_reduced_network_topology(ax, network_type, A_reduced, m, colors[i], color_bias, node_size_bias, title_letter)
        ax = fig.add_subplot(gs[i+1, 8])
        title_letter = f'({letters[i+0]}5)'
        title_letter = f'({letters[4]}{i+1})'
        if i == 0:
            ax.annotate('Dynamics', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')
        plot_xs_weight(ax, xlabel, ylabel, A_reduced, m, weight_list, xs_group, colors[i], linewidth, alpha, color_bias, title_letter, xs_multi, A_unit, group_index)

        if i == 2:
            groundtruth, = ax.plot([], [], color='tab:grey', alpha=0.8, label='ground truth', linewidth=2)
            reduced1, = ax.plot([], [], color=sns.color_palette(colors[0])[-1], linewidth=3)
            reduced2, = ax.plot([], [], color=sns.color_palette(colors[1])[-1],  linewidth=3)
            reduced3, = ax.plot([], [], color=sns.color_palette(colors[2])[-1],linewidth=3)
            ax.legend( [groundtruth, (reduced1, reduced2, reduced3)], ['ground truth', 'reduced'], fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.05,-1.8), handler_map={tuple: HandlerTuple(ndivide=None)} ) 



        for j, m in enumerate([3, 5, 10]):
            if i == 2:
                xlabel = '$w$'
            else:
                xlabel = ''
            if i == 1:
                ylabel = '$y^{(\\mathrm{gl})}_s$'
            else:
                ylabel = ''

            if j == 0:
                ax = fig.add_subplot(gs[4+i+1, 0])
            elif j == 1:
                ax = fig.add_subplot(gs[4+i+1, 3:5])
            elif j == 2:
                ax = fig.add_subplot(gs[4+i+1, 7])
            title_letter = f'({letters[i+0]}{6+j*2})'
            title_letter = f'({letters[5+j*2]}{i+1})'
            if i == 0:
                ax.annotate( f'$m={m}$', xy = (0.7, 0.5),  xytext=(0.95, 1.25), xycoords='axes fraction', fontsize=15, color='tab:grey', weight='bold')
                ax.annotate('$\\beta_{ab}$', xy = (-0.7, 0.5),  xytext=(0.85, 0.85), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
            xs_group = read_xs(network_type, N, space, d, seed, m, weight_list)
            group_index = group_index_from_feature_Kmeans(feature, m)
            A_reduced, _, _ = reducednet_effstate(A_unit, np.random.random(len(A_unit)), group_index)
            plot_reduced_network_topology(ax, network_type, A_reduced, m, colors[i], color_bias, node_size_bias, title_letter)

            if j == 0:
                ax = fig.add_subplot(gs[4+i+1, 1:3])
            elif j == 1:
                ax = fig.add_subplot(gs[4+i+1, 5:7])
            elif j == 2:
                ax = fig.add_subplot(gs[4+i+1, 8])
            title_letter = f'({letters[i+0]}{7+j*2})'
            title_letter = f'({letters[6+j*2]}{i+1})'
            #plot_xs_weight(ax, xlabel, ylabel, A_reduced, m, weight_list, xs_group, colors[i], linewidth, alpha, color_bias, title_letter, xs_multi, A_unit, group_index)
            plot_ygl_weight(ax, xlabel, ylabel, A_reduced, m, weight_list, xs_group, colors[i], linewidth, alpha, color_bias, title_letter, xs_multi, A_unit, group_index)


    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    ax.annotate( 'm-dimensional Reduction', xy = (0.7, 0.5),  xytext=(0.22, 0.5), xycoords='axes fraction', fontsize=14, color='tab:grey', weight='bold')
    dxdt = r'$\frac{dy^{(a)}}{dt} = F(x_i) + w \sum_{b=1}^{m} \beta_{ab} G(y^{(a)}, y^{(b)})$'
    t = ax.text(0.63, 0.58, dxdt, ha="center", va="center", rotation=0, size=15, bbox=dict(boxstyle="round,pad=0.3", fc="tab:grey", ec="k", lw=1, alpha=0.5))
    draw_brace(ax, (0.12 * ax.get_xlim()[1], 0.8 * ax.get_xlim()[1]), 'tab:grey', 3, '')

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.70, bottom=0.05, top=0.95)

def main_fig_onedimension(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics):
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
    fig = plt.figure(figsize=(13, 6))
    gs = mpl.gridspec.GridSpec(nrows=4, ncols=9, height_ratios=[0.9, 1, 1, 1], width_ratios=[1.05, 0.25, 0.75, 0.10, 0.75, 0.25, 0.75, 0.85, 1])
    #plt.rcParams.update({"text.usetex": True,})
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.annotate( 'One-dimensional Reduction', xy = (1.2, 0.8),  xytext=(0.55, 0.7), xycoords='axes fraction', fontsize=14, color='tab:grey', weight='bold')


    #plt.rcParams.update({"text.usetex": False,})
    for (i, network_type), N, d, seed, space in zip(enumerate(network_type_list), N_list, d_list, seed_list, space_list):
        if network_type == 'ER':
            color_bias = 0
            node_size_bias = 1
        else:
            color_bias = 0.1
            node_size_bias = 5


        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        if i == 2:
            xlabel = '$w$'
            xk_label = '$k$'
        else:
            xlabel = ''
            xk_label = ''
        if i == 1:
            ylabel = '$y^{(\\mathrm{gl})}_s$'
            ylabel_xs= '$x_s$'
            yk_label = '$P(k)$'
        else:
            ylabel_xs = ''
            ylabel = ''
            yk_label = ''

        ax = fig.add_subplot(gs[i+1, 0:2])
        ax.annotate('$A_{ij}$', xy = (-0.7, 0.5),  xytext=(0.9, 0.9), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
        title_letter = f'({letters[i+0]}1)'
        title_letter = f'({letters[0]}{i+1})'
        plot_network_topology(ax, network_type, N, A_unit, colors[i], color_bias, node_size_bias, title_letter)
        if network_type == 'SF':
            xy = (-0.4, 0.5)
        else:
            xy = (-0.7, 0.5)
        ax.annotate( network_type, xy=xy,  xytext=(-0.85 - 0.08* len(network_type) , 0.45), xycoords='axes fraction', fontsize=15, color=sns.color_palette(colors[i])[-1], alpha=.5, weight='bold')
        if i == 0:
            ax.annotate('Topology', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')

        xs_multi = read_xs(network_type, N, space, d, seed, N, weight_list)
        ax = fig.add_subplot(gs[i+1, 2])
        title_letter = f'({letters[i+0]}2)'
        title_letter = f'({letters[1]}{i+1})'
        ax.annotate(title_letter, xy=(-0.2, 1.03), xycoords="axes fraction", size=labelsize*0.8)
        simpleaxis(ax)
        k = np.sum(A_unit >0 , 0)
        if network_type == 'SF':
            sns.histplot(k, bins=20, stat='density', ax=ax, color = sns.color_palette(colors[i])[-1], alpha=0.5)
            ax.set_yscale('log')
        else:
            sns.histplot(k, bins=20, stat='density', ax=ax, color = sns.color_palette(colors[i])[-1], alpha=0.5)
        ax.set_xlabel(xk_label, fontsize=labelsize)
        ax.set_ylabel(yk_label, fontsize=labelsize)

        ax = fig.add_subplot(gs[i+1, 4:6])
        title_letter = f'({letters[i+0]}3)'
        title_letter = f'({letters[2]}{i+1})'
        linewidth = 2
        alpha = 0.8
        plot_xs_weight(ax, xlabel, ylabel_xs, A_unit, len(A_unit), weight_list, xs_multi, colors[i], linewidth, alpha, color_bias, title_letter)
        if i == 0:
            ax.annotate('Dynamics', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')

        ax = fig.add_subplot(gs[i+1, 6])
        ax.set_axis_off()
        ax.annotate(' ' * 10 , xy=(0.4, 0.50), xytext=(0.75, 0.5), xycoords='axes fraction', ha='center', va='bottom', bbox=dict(boxstyle='rarrow, pad=0.6', fc= sns.color_palette(colors[i])[-1] , ec='k', lw=2, alpha=0.5) )

        ax = fig.add_subplot(gs[i+1, 7])
        title_letter = f'({letters[i+0]}4)'
        title_letter = f'({letters[3]}{i+1})'
        m = 1
        xs_group = read_xs(network_type, N, space, d, seed, m, weight_list)
        group_index = group_index_from_feature_Kmeans(feature, m)
        A_reduced, _, _ = reducednet_effstate(A_unit, xs_multi[0], group_index)
        ax.annotate('$\\beta$', xy = (-0.2, 0.5),  xytext=(1, 1), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
        if i == 0:
            ax.annotate('Topology', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')
        plot_reduced_network_topology(ax, network_type, A_reduced, m, colors[i], color_bias, node_size_bias, title_letter)
        ax = fig.add_subplot(gs[i+1, 8])
        title_letter = f'({letters[i+0]}5)'
        title_letter = f'({letters[4]}{i+1})'
        if i == 0:
            ax.annotate('Dynamics', xy = (-0.7, 0.5),  xytext=( 0.25, 1.4), xycoords='axes fraction', fontsize=15, color='tab:grey', alpha=.5, weight='bold')
        plot_xs_weight(ax, xlabel, ylabel, A_reduced, m, weight_list, xs_group, colors[i], linewidth, alpha, color_bias, title_letter, xs_multi, A_unit, group_index)

        if i == 2:
            groundtruth, = ax.plot([], [], color='tab:grey', alpha=0.8, label='ground truth', linewidth=2)
            reduced1, = ax.plot([], [], color=sns.color_palette(colors[0])[-1], linewidth=3)
            reduced2, = ax.plot([], [], color=sns.color_palette(colors[1])[-1],  linewidth=3)
            reduced3, = ax.plot([], [], color=sns.color_palette(colors[2])[-1],linewidth=3)
            ax.legend( [groundtruth, (reduced1, reduced2, reduced3)], ['ground truth', 'reduced'], fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.05,4.8), handler_map={tuple: HandlerTuple(ndivide=None)} ) 



    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.70, bottom=0.11, top=0.95)

def main_fig_mdimension(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics):
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
    fig = plt.figure(figsize=(13, 5.5))
    gs = mpl.gridspec.GridSpec(nrows=4, ncols=9, height_ratios=[1.15, 1, 1, 1], width_ratios=[1.05, 0.25, 0.75, 0.10, 0.75, 0.25, 0.75, 0.85, 1])
    #plt.rcParams.update({"text.usetex": True,})


    #plt.rcParams.update({"text.usetex": False,})
    for (i, network_type), N, d, seed, space in zip(enumerate(network_type_list), N_list, d_list, seed_list, space_list):
        if network_type == 'ER':
            color_bias = 0
            node_size_bias = 1
        else:
            color_bias = 0.1
            node_size_bias = 5


        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        if i == 2:
            xlabel = '$w$'
            xk_label = '$k$'
        else:
            xlabel = ''
            xk_label = ''
        if i == 1:
            ylabel = '$y^{(\\mathrm{gl})}_s$'
            ylabel_xs= '$x_s$'
            yk_label = '$P(k)$'
        else:
            ylabel_xs = ''
            ylabel = ''
            yk_label = ''


        xs_multi = read_xs(network_type, N, space, d, seed, N, weight_list)
        linewidth = 2
        alpha = 0.8


        for j, m in enumerate([3, 5, 10]):
            if i == 2:
                xlabel = '$w$'
            else:
                xlabel = ''
            if i == 1:
                ylabel = '$y^{(\\mathrm{gl})}_s$'
            else:
                ylabel = ''

            if j == 0:
                ax = fig.add_subplot(gs[i+1, 0])
            elif j == 1:
                ax = fig.add_subplot(gs[i+1, 3:5])
            elif j == 2:
                ax = fig.add_subplot(gs[i+1, 7])
            title_letter = f'({letters[j*2]}{i+1})'

            if j == 0:
                if network_type == 'SF':
                    xytext = (-0.55 - 0.08* len(network_type) , 0.45)
                else:
                    xytext = (-0.85 - 0.08* len(network_type) , 0.45)
                ax.annotate( network_type, xy=(-0.7, 0.5), xytext=xytext, xycoords='axes fraction', fontsize=15, color=sns.color_palette(colors[i])[-1], alpha=.5, weight='bold')
            if i == 0:
                ax.annotate( f'$m={m}$', xy = (0.7, 0.5),  xytext=(0.95, 1.25), xycoords='axes fraction', fontsize=15, color='tab:grey', weight='bold')
                ax.annotate('$\\beta_{ab}$', xy = (-0.7, 0.5),  xytext=(0.85, 0.85), xycoords='axes fraction', fontsize=15, color='k', alpha=.8, weight='bold')
            xs_group = read_xs(network_type, N, space, d, seed, m, weight_list)
            group_index = group_index_from_feature_Kmeans(feature, m)
            A_reduced, _, _ = reducednet_effstate(A_unit, np.random.random(len(A_unit)), group_index)
            plot_reduced_network_topology(ax, network_type, A_reduced, m, colors[i], color_bias, node_size_bias, title_letter)

            if j == 0:
                ax = fig.add_subplot(gs[i+1, 1:3])
            elif j == 1:
                ax = fig.add_subplot(gs[i+1, 5:7])
            elif j == 2:
                ax = fig.add_subplot(gs[i+1, 8])
            title_letter = f'({letters[1+j*2]}{i+1})'
            plot_ygl_weight(ax, xlabel, ylabel, A_reduced, m, weight_list, xs_group, colors[i], linewidth, alpha, color_bias, title_letter, xs_multi, A_unit, group_index)

            if j == 2 and i == 2:
                groundtruth, = ax.plot([], [], color='tab:grey', alpha=0.8, label='ground truth', linewidth=2)
                reduced1, = ax.plot([], [], color=sns.color_palette(colors[0])[-1], linewidth=3)
                reduced2, = ax.plot([], [], color=sns.color_palette(colors[1])[-1],  linewidth=3)
                reduced3, = ax.plot([], [], color=sns.color_palette(colors[2])[-1],linewidth=3)
                ax.legend( [groundtruth, (reduced1, reduced2, reduced3)], ['ground truth', 'reduced'], fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.05,4.8), handler_map={tuple: HandlerTuple(ndivide=None)} ) 



    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.annotate( 'm-dimensional Reduction', xy = (0.7, 0.5),  xytext=(0.37, 0.5), xycoords='axes fraction', fontsize=14, color='tab:grey', weight='bold')
    draw_brace(ax, (0.12 * ax.get_xlim()[1], 0.8 * ax.get_xlim()[1]), 'tab:grey', 3, '')

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.70, bottom=0.1, top=0.95)


    

dynamics = 'CW_high'
beta_list = [5, 10, 15, 20, 25]
dynamics = 'CW'
beta_list = [20, 30, 40, 50, 60]
dynamics = 'genereg'
beta_list = [1.5, 2.0, 3.0, 4.0, 5.0]
dynamics = 'mutual'
beta_list = [1.0, 3.0, 5.0, 7.0, 9.0]

N = 100
m_list = [N]

arguments = (B, C, D, E, H, K_mutual)



network_type = 'ER'
N = 100
space = 'linear'
d = 1600
seed = 0

network_type = 'SBM_ER'
space = 'linear'
N = [33, 33, 34]
d = np.array([[0.9, 0.001, 0.001], [0.001, 0.5, 0.001], [0.001, 0.001, 0.05]]).tolist()
seed = 0

network_type = 'SF'
N = 100
space = 'log'
d = [2.1, N-1, 1]
seed = [0, 0]




network_type_list = ['ER', 'SF', 'SBM_ER']
space_list = ['linear', 'log', 'linear']
N_list = [100, 100, [33, 33, 34]]
d_list = [1600, [2.1, 99, 1], np.array([[0.9, 0.001, 0.001], [0.001, 0.5, 0.001], [0.001, 0.001, 0.05]]).tolist() ]
seed_list = [0, [15, 15], 0]
weight_list = np.round(np.arange(0.01, 0.6, 0.01), 5)

#main_fig(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics)

#main_fig_onedimension(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics)
main_fig_mdimension(network_type_list, space_list, N_list, d_list, seed_list, weight_list, dynamics)
"""
"degree distribution"
#sns.histplot(k, bins=20)

#fig, ax = plt.subplots()
attractor_value = 0.1
network_type = 'SF'
N = 100
space = 'log'
d = [2.1, N-1, 1]
seed = [1, 1]
seed_list = np.tile(np.arange(10, 20, 1), (2, 1)).transpose()
m = 20
seed_list = [[18, 18]]
weight_list = np.round(np.arange(0.2, 0.7, 0.01), 5)
for seed in seed_list:
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_list = []
    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_list.append(xs_multi)

    xs_list = np.vstack(( xs_list ))
    plt.plot(weight_list, xs_list,'tab:blue')

    xs_reduction_list = []
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    dynamics_multi = globals()[dynamics + '_multi']
    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        feature = feature_from_network_topology(A, G, space, 0.5, 'degree')
        group_index = group_index_from_feature_Kmeans(feature, m)
        A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(N_actual), group_index)
        initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
        xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
        xs_reduction_list.append(xs_reduction_deg_part)
    xs_reduction_list = np.vstack(( xs_reduction_list ))
    plt.plot(weight_list, xs_reduction_list, 'tab:red')
    plt.show()




m = 3
node_size_bias = 1
color_bias = 0
feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
group_index = group_index_from_feature_Kmeans(feature, m)
A_reduced, _, _ = reducednet_effstate(A_unit, np.random.random(len(A_unit)), group_index)
cmap = sns.color_palette('Reds', as_cmap=True)
k = np.sum(A_reduced, 1)
node_size = np.log(k+node_size_bias) 
node_size_dict = {u:v for u, v in zip(np.arange(len(k)), node_size)}
node_color_dict = {u: cmap(k_i / k.max()+color_bias ) for u, k_i in zip(np.arange(len(k)), k)}
edgelists = [ (ni, nj) for ni in range(m) for nj in range(m) if A_reduced[ni, nj]]


node_sizes = np.ones(len(k)) * 2
node_size_dict=  {u: node_sizes[u] for u in np.arange(len(k))}
radius = 0.05
widths = np.log(A_reduced / A_reduced.max() + 1.3) * 5 
edge_width = {(u, v) :  widths[v, u] * 0.3 if u == v else widths[v, u] for (u, v) in edgelists}
node_layout = 'spring'

node_layout = {0:(0.1, 0.2), 1:(0.9, 0.2), 2:(0.5, 0.8)}
node_edge_width = 1

#g = Graph(edgelists, edge_width=edge_width, arrows=True, node_layout = node_layout, node_size=node_size_dict, edge_layout='straight', node_color=node_color_dict, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.5, ax=ax, edge_layout_kwargs={'selfloop_radius': radius})

"""
