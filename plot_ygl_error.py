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

from collections import Counter
from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core
import scipy.stats as stats
import time
from netgraph import Graph
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib.colors import LogNorm

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
labelsize = 30
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

def df_select_d_sort_seed(d_df, seed_df, d_list, seed_list):
    index_select = []
    if type(d_df[0]) == str:
        d_df = [eval(i) for i in d_df]
        seed_df = [eval(i) for i in seed_df]
    for d_i in d_list:
        index_i = []
        seed_i = []
        for (i, d_df_i), seed_df_i in zip(enumerate(d_df), seed_df):
            if d_df_i == d_i:
                if type(seed_df_i) == list:
                    if seed_df_i[0] in seed_list:
                        seed_i.append(seed_df_i[0])
                        index_i.append(i) 
                else:
                    if seed_df_i in seed_list:
                        seed_i.append(seed_df_i)
                        index_i.append(i) 
        index_sort = np.argsort(seed_i)
        index_select.extend(np.array(index_i)[index_sort])
    return index_select

def y_bifurcation(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    cmap = sns.color_palette('flare', as_cmap=True)
    rows, cols = 2, int(len(m_list) / 2)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(3 * cols, 3 * rows))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for i, m in enumerate(m_list):
        des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
        yi_list = []
        for beta in beta_list:
            file_name = des + f'm={m}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_name, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            yi_list.append(np.array(yi[index_select],float))
            kmean_list = np.array(kmean[index_select], float)
            h1_list = np.array(h1[index_select], float)
            beta_unweighted_list = np.log10(h1_list+1)
            beta_unweighted_list = np.log10( h1_list + kmean_list)
        yi_list = np.vstack( (yi_list) ) 
        ax = axes[i//cols, i % cols]
        simpleaxis(ax)
        ax.annotate(f'({letters[i]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        #sns.lineplot(data=df, x='m', y=plot_type, ci=None, ax=ax)
        for yi, beta_i in zip(yi_list.transpose(), beta_unweighted_list):
            ax.plot(beta_list, yi, linestyle='-', linewidth=2, alpha=0.05, color=cmap((beta_i-beta_unweighted_list.min())/(beta_unweighted_list.max() - beta_unweighted_list.min())) )
        title_name = f'$m={m}$'
        ax.set_title(title_name, size=labelsize*0.6)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=16*0.8)
    vmin, vmax = beta_unweighted_list.min(), beta_unweighted_list.max()
    cax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks)-1, int), size=ticksize)
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize)
    #cax.get_yaxis().set_ticklabels(np.array((cbar_ticks), int), size=ticksize)
    cax.set_ylabel('$h(w=1)$', fontsize=labelsize*0.8)

    xlabel = '$ \\beta $'
    ylabel = '$y^{(\\mathrm{gl})}$'
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    fig.text(x=0.05, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()

def y_compare_beta_m(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    cmap = sns.color_palette('flare', as_cmap=True)
    rows, cols = len(beta_list), len(m_list) 
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(3 * cols, 3 * rows))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
    for i, beta in enumerate(beta_list):
        file_multi = des + f'm={N}_beta={beta}.csv'
        data = np.array(pd.read_csv(file_multi, header=None))
        d, seed, kmean, h1, h2, yi = data.transpose()
        index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
        y_multi = np.array(yi[index_select],float)
        kmean_list = np.array(kmean[index_select], float)
        h1_list = np.array(h1[index_select], float)
        beta_unweighted_list = np.log10(kmean_list+h1_list)
        beta_unweighted_list = h1_list
        for j, m in enumerate(m_list):
            ax = axes[i, j]
            simpleaxis(ax)
            file_group = des + f'm={m}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_group, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            y_group = np.array(yi[index_select],float)
            ax.annotate(f'({letters[i*cols+j]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
            for yi_multi, yi_group, beta_i in zip(y_multi, y_group, beta_unweighted_list):
                ax.plot(yi_multi, yi_group, 'o', markersize=5, alpha=0.1, color=cmap((beta_i-beta_unweighted_list.min())/(beta_unweighted_list.max() - beta_unweighted_list.min())) )
            title_name = f'$\\beta={int(beta)}$  |  $m={m}$'
            ax.set_title(title_name, size=labelsize*0.5)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=16*0.8)
    vmin, vmax = beta_unweighted_list.min(), beta_unweighted_list.max()
    cax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize)
    cax.get_yaxis().set_ticklabels(np.array((cbar_ticks), int), size=ticksize)
    cax.set_ylabel('$h(w=1)$', fontsize=labelsize*0.8)

    xlabel = '$ y^{(\\mathrm{gl})}_{\\mathrm{original}} $'
    ylabel = '$ y^{(\\mathrm{gl})}_{\\mathrm{reduction}} $'
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    fig.text(x=0.05, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()

def y_compare_beta(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    cmap = sns.color_palette('flare', as_cmap=True)
    colors = sns.color_palette("husl", 11) 
    markers = ['o', 's', 'P', 'X', 'D', '*'] 
    rows, cols = 1, len(beta_list)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(3 * cols, 3 * rows))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
    for i, beta in enumerate(beta_list):
        ax = axes[i]
        simpleaxis(ax)
        file_multi = des + f'm={N}_beta={beta}.csv'
        data = np.array(pd.read_csv(file_multi, header=None))
        d, seed, kmean, h1, h2, yi = data.transpose()
        index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
        y_multi = np.array(yi[index_select],float)
        kmean_list = np.array(kmean[index_select], float)
        h1_list = np.array(h1[index_select], float)
        beta_unweighted_list = np.log10( h1_list)
        beta_unweighted_list = h1_list
        for j, m in enumerate(m_list):
            file_group = des + f'm={m}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_group, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            y_group = np.array(yi[index_select],float)
            ax.annotate(f'({letters[i]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
            ax.plot(y_multi, y_group, markers[j], markersize=5, alpha=0.1, markeredgecolor=colors[j], markerfacecolor='none')
            title_name = f'$\\beta={int(beta)}$'
            ax.set_title(title_name, size=labelsize*0.5)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(labelsize=16*0.8)
    vmin, vmax = beta_unweighted_list.min(), beta_unweighted_list.max()
    xlabel = '$ y^{(\\mathrm{gl})}_{\\mathrm{original}} $'
    ylabel = '$ y^{(\\mathrm{gl})}_{\\mathrm{reduction}} $'
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    fig.text(x=0.05, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()

def yerror_m_beta(network_type, N, d_list, seed_list, dynamics_list, m_list, space, beta_list_lists):
    fig, axes = plt.subplots(2, int(len(dynamics_list)/2), sharex=True, sharey=True, figsize=(5*int(len(dynamics_list)/2), 2*3))
    cmap = sns.color_palette('flare', as_cmap=True)
    colors = sns.color_palette("husl", 11) 
    linestyles = [(i, j) for i in [1, 3, 5, 7, 9] for j in [1]]
    markers = ['o', 's', 'P', 'X', 'D', '*'] 
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for (k, dynamics), beta_list in zip(enumerate(dynamics_list), beta_list_lists):
        ax = axes[k//2, k%2]
        simpleaxis(ax)
        ax.annotate(f'({letters[k]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)

        des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
        for i, beta in enumerate(beta_list):
            file_multi = des + f'm={N}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_multi, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            y_multi = np.array(yi[index_select],float)
            kmean_list = np.array(kmean[index_select], float)
            h1_list = np.array(h1[index_select], float)
            beta_unweighted_list = np.log10( h1_list)
            beta_unweighted_list = h1_list
            error_mean = []
            error_std = []
            for j, m in enumerate(m_list):
                file_group = des + f'm={m}_beta={beta}.csv'
                data = np.array(pd.read_csv(file_group, header=None))
                d, seed, kmean, h1, h2, yi = data.transpose()
                index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
                y_group = np.array(yi[index_select],float)
                error = np.round(np.abs(y_multi - y_group), 10) / (y_multi + y_group)
                #ax.plot(m*np.ones(len(error)), error, 'o', markersize=5, alpha=0.5, color=colors[j])
                error_mean.append(np.mean(error) )
                error_std.append(np.std(error) )
            error_mean = np.array(error_mean)
            error_std = np.array(error_std)
            ax.plot(m_list, error_mean, linewidth=3, alpha=0.8, label=f'$\\beta={int(beta)}$', linestyle=(0, linestyles[i]) )
            #ax.fill_between(m_list, error_mean - error_std, error_mean + error_std, alpha=0.3)
            if 'high' in dynamics or dynamics == 'genereg':
                title_name = dynamics[:dynamics.find('_high')] + ' (H-->L)'
            else:
                title_name = dynamics + ' (L-->H)'
            ax.set_title(title_name, size=labelsize*0.5)

            ax.set_xscale('log')
            ax.set_yscale('symlog', linthreshy=1e-6)
            ax.tick_params(labelsize=ticksize*0.6)
            ax.legend(fontsize=legendsize*0.5, frameon=False, loc=4, bbox_to_anchor=(1.38,0) ) 

    xlabel = '$ m $'
    ylabel = '$ \\mathrm{Err} (y^{(\\mathrm{gl})})$'
    fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)

    #plt.tick_params(labelsize=16*0.9)
    plt.subplots_adjust(left=0.15, right=0.88, wspace=0.40, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()

def plot_xs_onenet(network_type, N, d, seed, dynamics, m_list, space, weight_list, weight_plot, threshold_value):
    """TODO: Docstring for plot_P_w.

    :weight_list: TODO
    :returns: TODO

    """    
    #fig, axes = plt.subplots(3, len(weight_plot), sharex=False, sharey=False, figsize=(4*len(weight_plot) , 3*3 ) )
    fig = plt.figure( figsize=(4*len(weight_plot) , 3*3 ))
    gs = mpl.gridspec.GridSpec(nrows=3, ncols=8, height_ratios=[1, 1, 1])
    letters = list('abcdefghijklmnopqrstuvwxyz')
    markers = ['o', '^', 's', 'p', 'P', 'h']
    linestyles = [(j, i) for i in [1, 3, 6, 9, 12] for j in [2, 5, 8, 11]]
    colors=sns.color_palette('hls', 40)
    alphas = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    sizes = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
    des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/'
    xs_multi_file = des + 'xs_multi/' + f'N={N}_d={d}_seed={seed}.csv'
    data_multi = np.array(pd.read_csv(xs_multi_file, header=None))
    weights_multi = data_multi[:, 0]
    index = [np.where(np.abs(weights_multi - weight) < 1e-02 )[0][0]  for weight in weight_list]
    xs_multi = data_multi[index, 1:]
    y_multi = betaspace(A_unit, xs_multi)[-1]
    y_group_list = dict()
    xs_group_list = dict()
    groups_node_nums = dict()
    y_group_list[N] = xs_multi
    xs_group_list[N] = xs_multi
    groups_node_nums[N] = np.array(np.ones(N), int)
    for m in m_list[:-1]:
        xs_group_file = des + f'degree_kmeans_space={space}/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data_group = np.array(pd.read_csv(xs_group_file, header=None))
        group_index = group_index_from_feature_Kmeans(feature, m)
        groups_node_nums[m] = np.array([len(i) for i in group_index])
        weights_group = data_group[:, :1]
        index = [np.where(np.abs(weights_group - weight) < 1e-02 )[0][0]  for weight in weight_list]
        y_group = data_group[index, 1:]
        xs_group = np.zeros( (len(y_group), N) )
        for i, group_i in enumerate(group_index):
            xs_group[:, group_i] = y_group[:, i:i+1]
        y_group_list[m] = y_group
        xs_group_list[m] = xs_group

    colormap = plt.get_cmap("RdBu")
    norm = mpl.colors.LogNorm(vmin=0.1, vmax=10)
    m_3D = [1, 4, 16, N]
    weight_3D = 0.15
    x_pos = np.random.random(N) * 5
    y_pos = np.random.random(N) * 5
    index_3D = np.where(np.abs(weights_multi - weight_3D)<1e-5)[0][0]
    for i, m in enumerate(m_3D):
        ax_nx = fig.add_subplot(gs[0, i*2])
        ax = fig.add_subplot(gs[0, i*2+1], projection='3d')
        ax_nx.annotate(f'({letters[i]})', xy=(-0.3, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        ax_nx.annotate(f'$m={m}$', xy=(1., 1.03), xycoords="axes fraction", size=labelsize*0.6)
        group_index = group_index_from_feature_Kmeans(feature, m)
        if m == N :
            A_reduced = A_unit
        else:
            A_reduced, _, _ = reducednet_effstate(A_unit, np.random.random(N), group_index)
        #axes[0, i].remove()
        #ax = fig.add_subplot(3, len(weight_plot), i+1, projection='3d')
        if m == N:
            edgelists = [ (ni, nj) for ni in range(m) for nj in range(ni) if A_reduced[ni, nj]]
        else:
            edgelists = [ (ni, nj) for ni in range(m) for nj in range(m) if A_reduced[ni, nj]]

        widths = np.log(A_reduced / A_reduced.max() + 1.3)  * 2
        edge_width = {(u, v) :  widths[v, u] for (u, v) in edgelists}
        node_list = np.arange(m)
        node_sizes = np.log(groups_node_nums[m] +1.0) / 1.8 
        node_size_dict=  {u: node_sizes[u] for u in node_list}
        radius = 0.03
        node_edge_width = 0.5
        scale = (1, 1)
        if m == 1:
            node_layout = {0:(0.7, 0.5)}
            node_size_dict = {0:15}
            radius = 0.2
            edge_width = 5
            node_edge_width=3
            ax_nx.set_xlim(0, 1)
            ax_nx.set_ylim(0, 1)
        else:
            node_layout = 'spring'
        color_i = colors[np.where(m==np.array(m_list))[0][0]]
        g = Graph(edgelists, edge_width=edge_width, arrows=True, node_layout=node_layout, node_edge_width=node_edge_width, node_size=node_size_dict, edge_layout='straight', node_color=color_i, node_alpha=0.8, edge_color=color_i, edge_alp=0.8, scale=scale, ax=ax_nx, edge_layout_kwargs={'selfloop_radius': radius})

        if m == N:
            xs_data = xs_multi[index_3D]
        else:
            xs_data = xs_group_list[m][index_3D]
        ax.scatter(x_pos, y_pos, xs_data, s=1, c=colormap(norm(xs_data)) )
        for ni, nj in list(G.edges):
            ax.plot([x_pos[ni], x_pos[nj]], [y_pos[ni], y_pos[nj]], [xs_data[ni], xs_data[nj]], linewidth=0.1, alpha=0.2, color='tab:grey')
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=20, azim=135)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0, 5, 10])
        ax.tick_params(axis='z', which='major', pad=-2)

    for j, weight in enumerate(weight_plot):
        index_plot = np.where(np.abs(weight_list - weight) < 1e-02 )[0][0] 
        ax = fig.add_subplot(gs[1, j*2:(j+1)*2])
        ax.set_ylim(0.09, 13)
        ax.annotate(f'({letters[j+4]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        simpleaxis(ax)
        for k, m in enumerate(m_list):
            y = y_group_list[m][index_plot]
            ax.scatter(x=np.ones(len(y)) * m, y=y, s= (groups_node_nums[m] / np.sum(groups_node_nums[m]) + 0.05) * 100, alpha=np.log(min(m_list)+0.5) / np.log(m+0.5), color=colors[k]) 
        ax.set(xscale='log', yscale='log')
           
        title_name = f'$w={weight}$'
        ax.set_title(title_name, size=labelsize*0.5)
        ax.set_xlabel('$m$', size=labelsize*0.5)
        ax.set_ylabel('$x_s$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[2, 0:2])
    simpleaxis(ax)
    ax.annotate(f'({letters[8]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    m_ygl = [1, 4, 16, 64, 256, N]
    for i_m, m in enumerate(m_ygl):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        xs_group_m = xs_group_list[m]
        y_gl = betaspace(A_unit, xs_group_m)[-1]
        ax.plot(weight_list, y_gl, linewidth=3, alpha=0.8, color=colors[m_index], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
    ax.set_title(title_name, size=labelsize*0.5)
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc=4, bbox_to_anchor=(1.01,0) ) 
    ax.set(yscale='log')
    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$y_{(\\mathrm{gl})}$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[2, 2:4])
    ax.annotate(f'({letters[9]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    simpleaxis(ax)
    for i_m, m in enumerate(m_ygl[:-1]):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        prediction_ratios = []
        for i_w, weight in enumerate(weight_list):
            high_nodes_multi = np.where(xs_group_list[N][i_w] > threshold_value)[0]
            low_nodes_multi = np.where(xs_group_list[N][i_w] < threshold_value)[0]
            high_nodes_group = np.where(xs_group_list[m][i_w] > threshold_value)[0]
            low_nodes_group = np.where(xs_group_list[m][i_w] < threshold_value)[0]
            prediction_ratio = (len(np.intersect1d(high_nodes_multi, high_nodes_group))  + len(np.intersect1d(low_nodes_multi, low_nodes_group)) )/ N
            prediction_ratios.append(prediction_ratio)
        ax.plot(weight_list, prediction_ratios, linewidth=3, alpha=0.8,  color=colors[m_index], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc=4, bbox_to_anchor=(1.01,0) ) 

    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('precision', size=labelsize*0.45)

    ax = fig.add_subplot(gs[2, 4:6])
    ax.annotate(f'({letters[10]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    simpleaxis(ax)
    xs_multi = xs_group_list[N]
    ygl_multi = betaspace(A_unit, xs_multi)[-1]
    error_list = []
    for m in m_list[:-1]:
        xs_group = xs_group_list[m]
        ygl_group = betaspace(A_unit, xs_group)[-1]
        error = np.round(np.abs(ygl_multi-ygl_group), 10) / (np.abs(ygl_multi + ygl_group) )
        error_list.append(error)
    error_list = np.vstack( (error_list) ) 
    m_ticks = m_list[:-1][::5]
    weight_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    sns.heatmap(error_list[::-1], vmin=0, vmax=np.max(1), linewidths=0, ax=ax)
    ax.set_yticks(np.arange(1, len(m_list)-1)[::5]+1.5)
    ax.set_yticklabels(m_ticks[::-1], rotation=-20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    ax.set_xticklabels(weight_ticks, rotation=20)

    ax.set_title('Error', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$m$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[2, 6:8])
    simpleaxis(ax)
    ax.annotate(f'({letters[11]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    ygl_group_list = []
    for m in m_list:
        xs_group = xs_group_list[m]
        ygl_group = betaspace(A_unit, xs_group)[-1]
        ygl_group_list.append(ygl_group)
    ygl_list = np.vstack( (ygl_group_list) ) 
    m_ticks = m_list[::5]
    weight_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    sns.heatmap(ygl_list[::-1], vmin=0, vmax=np.max(ygl_list.max()), cmap='YlGnBu', linewidths=0, ax=ax)
    ax.set_yticks(np.arange(0, len(m_list))[::5]-0.5)
    ax.set_yticklabels(m_ticks[::-1], rotation=-20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    ax.set_xticklabels(weight_ticks, rotation=20)

    ax.set_title('$y^{(\\mathrm{gl})}$', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$m$', size=labelsize*0.5)


    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.55, hspace=0.55, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + network_type + '_subplots_xs_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    return None

def y_bifurcation_3d(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    letters = list('abcdefghijklmnopqrstuvwxyz')
    cmap = sns.color_palette('flare', as_cmap=True)
    rows, cols = 2, int(len(m_list) / 2)
    fig = plt.figure( figsize=(9, 6 ))
    gs = mpl.gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[2, 1, 1])
    ax = fig.add_subplot(gs[0, :], projection='3d')
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=20, azim=135)
    ax.set_xticks(np.log(m_list))
    ax.set_xticklabels(m_list)
    ax.set_yticks([0, 5, 10])
    ax.set_yticklabels([10, 5, 0])
    ax.set_xlabel('$m$', fontsize=20, labelpad=12)
    ax.set_ylabel('$\\beta$', fontsize=20, labelpad=12)
    ax.set_zlabel('$y^{(\\mathrm{gl})}$', fontsize=20)
    ax.tick_params(labelsize=16*0.8)
    ax.annotate('(a)', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.9)

    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
    m_all = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]
    colors=sns.color_palette('hls', 40)
    yi_list_m = []
    for i, m in enumerate(m_list):
        if m == 1:
            linewidth = 3
        else:
            linewidth = 1
        yi_list = []
        for beta in beta_list:
            file_name = des + f'm={m}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_name, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            yi_list.append(np.array(yi[index_select],float))
            kmean_list = np.array(kmean[index_select], float)
            h1_list = np.array(h1[index_select], float)
            beta_unweighted_list = np.log10(h1_list+1)
            beta_unweighted_list = np.log10( h1_list + kmean_list)
        yi_list = np.vstack( (yi_list) ) 
        yi_list_m.append(yi_list)
        color_i = colors[np.where(m==np.array(m_all))[0][0]]
        for yi, beta_i in zip(yi_list.transpose(), beta_unweighted_list):
            ax.plot3D( np.log(m) * np.ones(len(beta_list)), beta_list[::-1], yi, linestyle='-', linewidth=linewidth, alpha=1/ np.log(m+1)/5, color=color_i , zorder=N / m)

    yi_list_m = np.array(yi_list_m)
    for i, m in enumerate([4, 64, N]):
        ax0 = fig.add_subplot(gs[1, i])
        title_letter = f'({letters[i+1]})'
        simpleaxis(ax0)
        print(yi_list)
        yi_list = yi_list_m[np.where(m == np.array(m_list))[0][0]]
        ax0.plot(beta_list, yi_list, linestyle='-', linewidth=linewidth, alpha=0.05, color=color_i )

        ax0.annotate(title_letter, xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        title_name = f'$m={m}$'
        ax0.set_title(title_name, size=labelsize*0.6)

        
    for i, m in enumerate([1, 4, 64]):
        ax0 = fig.add_subplot(gs[2, i])
        title_letter = f'({letters[i+4]})'
        simpleaxis(ax0)
        yi_list = yi_list_m[np.where(m == np.array(m_list))[0][0]]
        yi_N = yi_list_m[-1]
        error = np.abs(yi_list - yi_N) / np.abs(yi_list + yi_N)
        ax0.plot(beta_list, error, linestyle='-', linewidth=linewidth, alpha=0.05, color=color_i )

        ax0.annotate(title_letter, xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        title_name = f'$m={m}$'
        ax0.set_title(title_name, size=labelsize*0.6)


    xlabel = '$ \\beta $'
    ylabel = '$y^{(\\mathrm{gl})}$'
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    fig.text(x=0.05, y=0.2, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)

    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()

def m_opt_fun(dynamics):
    """TODO: Docstring for m_opt_fun.

    :dynamics: TODO
    :returns: TODO

    """
    pass

    """optimal m
    """
    m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist()
    error_threshold = 0.1
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    des_multi = des + 'y_multi_beta/'
    des_group = des + 'y_group_beta/'
    hk_file = '../data/' + 'mutual' + '/' + network_type + '/xs_bifurcation/' + 'ygl_beta/'
    hk_data = np.array(pd.read_csv(hk_file+'m=1_beta=1.0.csv', header=None))
    d_list, seed_list, kmean_list, h1_list, h2_list, yi_list = hk_data.transpose()
    kmean_dict = dict()
    h_dict = dict()
    hk_ratio_dict = dict()
    for d, seed, kmean, h1, h2 in zip(d_list, seed_list, kmean_list, h1_list, h2_list):
        kmean_dict[(d, seed)] = kmean
        h_dict[(d, seed)] = h1
        hk_ratio = h1 / kmean
        hk_ratio_dict[(d, seed)] = hk_ratio

    hk_ratio_list = np.array(list(hk_ratio_dict.values()))
    ratio_plot = []
    for ratio_i in np.arange(0.1, 4, 0.1):
        index_i = np.argmin(np.abs(hk_ratio_list - ratio_i) )
        ratio_plot.append(hk_ratio_list[index_i])
    h_plot = []
    kmean_plot = []
    m_opt_list = []
    for d, seed in zip(d_list[:], seed_list[:]):
        if hk_ratio_dict[(d, seed)] in ratio_plot:
            file_multi = des_multi + f'N={N}_d={eval(d)}_seed={eval(seed)}.csv'
            data = np.array(pd.read_csv(file_multi, header=None))
            w, beta, y_multi = data.transpose()
            y_group_list = []
            error_list = []
            for m in m_list:
                file_group = des_group + f'N={N}_d={eval(d)}_number_groups={m}_seed={eval(seed)}.csv'
                data = np.array(pd.read_csv(file_group, header=None))
                w, beta, y_group = data.transpose()
                error = np.round(np.abs(y_multi - y_group), 5) / np.abs(y_multi + y_group)
                error_list.append(error)
            error_list.append(np.zeros(len(error)))
            error_list = np.vstack((error_list))
            error_index = []
            for i, w_i in enumerate(w):
                error_i = np.where(error_list[:, i] < error_threshold )[0][0]
                error_index.append(error_i)

            m_opt = np.array(m_list + [N])[error_index]
            h1_w = h_dict[(d, seed)] * w
            kmean_w = kmean_dict[(d, seed)] * w
            h_plot.append(h1_w)
            kmean_plot.append(kmean_w)
            m_opt_list.append(m_opt)

    h_plot = np.hstack((h_plot))
    kmean_plot = np.hstack((kmean_plot))
    m_opt_list = np.hstack((m_opt_list))
    return h_plot, kmean_plot, m_opt_list

def m_opt_beta_w(dynamics):
    """TODO: Docstring for m_opt_fun.

    :dynamics: TODO
    :returns: TODO

    """
    pass

    """optimal m
    """
    cmap = sns.color_palette('flare', as_cmap=True)
    m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist()
    error_threshold = 0.2
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    des_multi = des + 'y_multi_beta/'
    des_group = des + 'y_group_beta/'
    hk_file = '../data/' + 'mutual' + '/' + network_type + '/xs_bifurcation/' + 'ygl_beta/'
    hk_data = np.array(pd.read_csv(hk_file+'m=1_beta=1.0.csv', header=None))
    d_list, seed_list, kmean_list, h1_list, h2_list, yi_list = hk_data.transpose()
    kmean_dict = dict()
    h_dict = dict()
    hk_ratio_dict = dict()
    for d, seed, kmean, h1, h2 in zip(d_list, seed_list, kmean_list, h1_list, h2_list):
        kmean_dict[(d, seed)] = kmean
        h_dict[(d, seed)] = h1
        hk_ratio = h1 / kmean
        hk_ratio_dict[(d, seed)] = hk_ratio

    hk_ratio_list = np.array(list(hk_ratio_dict.values()))
    ratio_plot = []
    for ratio_i in np.arange(0.1, 4, 0.1):
        index_i = np.argmin(np.abs(hk_ratio_list - ratio_i) )
        ratio_plot.append(hk_ratio_list[index_i])

    h_plot = []
    kmean_plot = []
    m_opt_list = []
    d_seed_plot = []
    for d, seed in zip(d_list[:], seed_list[:]):
        beta_unweighted =  kmean_dict[(d, seed)] + h_dict[(d, seed)] 
        h_unweighted = h_dict[(d, seed)]
        #if 2.4 < eval(d)[0] < 4:
        if 2.4 < eval(d)[0] < 4 and 0.5 < h_unweighted < 40:
            h_plot.append(h_dict[(d, seed)])
            kmean_plot.append(kmean_dict[(d, seed)] )
            file_multi = des_multi + f'N={N}_d={eval(d)}_seed={eval(seed)}.csv'
            data = np.array(pd.read_csv(file_multi, header=None))
            w, beta, y_multi = data.transpose()
            index_sort = np.argsort(w)
            w, beta, y_multi = w[index_sort], beta[index_sort], y_multi[index_sort]
            y_group_list = []
            error_list = []
            for m in m_list:
                file_group = des_group + f'N={N}_d={eval(d)}_number_groups={m}_seed={eval(seed)}.csv'
                data = np.array(pd.read_csv(file_group, header=None))
                w, beta, y_group = data.transpose()
                w, beta, y_group = w[index_sort], beta[index_sort], y_group[index_sort]
                error = np.round(np.abs(y_multi - y_group), 5) / np.abs(y_multi + y_group)
                error_list.append(error)
            error_list.append(np.zeros(len(error)))
            error_list = np.vstack((error_list))
            error_index = []
            for i, w_i in enumerate(w):
                error_i = np.where(error_list[:, i] < error_threshold )[0][0]
                error_index.append(error_i)
            m_opt = np.array(m_list + [N])[error_index]
            m_opt_list.append(m_opt)
            d_seed_plot.append([d, seed])
            #plt.semilogy(beta, m_opt, color=cmap(beta_unweighted/40), alpha=0.3 , label=f'{beta_unweighted}') 

    h_plot = np.array(h_plot)
    h_index = []
    h_separate = np.linspace(min(h_plot), max(h_plot) + 1, 10)
    for h_i, h_j  in zip(h_separate[:-1], h_separate[1:]):
        hi_index = np.where((h_plot <= h_j ) & (h_plot > h_i))[0]
        h_index.append(hi_index)
        mi_opt =  np.array(m_opt_list )[hi_index]
        plt.plot(beta, np.vstack((mi_opt)).mean(0) , label=f'h={h_i}-{h_j}' )

    plt.legend()



    return h_plot, kmean_plot, m_opt_list, d_seed_plot

def yerror_m_beta_illustration(network_type, N, d_list, seed_list, beta_1D_list, dynamics_list, arguments_list, initial_condition_list, m_list, space, beta_list_lists, beta_c_list, beta_heatmap_list):
    fig = plt.figure( figsize=(12, 9 ))
    gs = mpl.gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[1, 1, 1])

    cmap = sns.color_palette('flare', as_cmap=True)
    colors = sns.color_palette("husl", 11) 
    linestyles = [(i, j) for i in [1, 3, 5, 7, 9] for j in [1]]
    markers = ['o', 's', 'P', 'X', 'D', '*'] 
    letters = list('abcdefghijklmnopqrstuvwxyz')

    for (k, dynamics), arguments, initial_condition, beta_1D in zip(enumerate(dynamics_list), arguments_list, initial_condition_list, beta_1D_list):
        t = np.arange(0, 100, 0.01)
        
        if dynamics == 'CW_high':
            dynamics_multi = globals()['CW_multi']
        else:
            dynamics_multi = globals()[dynamics + '_multi']


        ax1 = fig.add_subplot(gs[0, k])
        simpleaxis(ax1)
        ax1.annotate(f'({letters[k]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        if 'high' in dynamics or dynamics == 'genereg':
            title_name = dynamics[:dynamics.find('_high')] + ' (H-->L)'
            color_1D='tab:red'
        else:
            title_name = dynamics + ' (L-->H)'
            color_1D='tab:blue'
        ax1.set_title(title_name, size=labelsize*0.4)

        des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
        y_1D = []
        for beta in beta_1D:
            net_arguments = ([0], [0], [beta], [0, 1])
            y_i = odeint(dynamics_multi, [initial_condition], t, args=(arguments, net_arguments))[-1]
            y_1D.append(y_i)
        ax1.plot(beta_1D, y_1D, color=color_1D, linewidth=3, alpha=0.8)
        ax1.set_xscale('log')

        start_i = [beta_1D[10], beta_1D[-10] ]
        #for start_i, end_i in zip(start_pos, end_pos):
        #ax1.annotate("", xy=end_i, xytext=start_i, arrowprops=dict(arrowstyle="->"))
        if dynamics == 'genereg':
            ax1.set_yscale('symlog', linthresh=1e-2)
            ax1.set_ylim(-1e-3)

        else:
            ax1.set_yscale('log')

    "need to be commented when generating real plots..."
    #m_list = m_list[::8]    
    for (k, dynamics), beta_list in zip(enumerate(dynamics_list), beta_list_lists):
        ax2 = fig.add_subplot(gs[1, k])
        simpleaxis(ax2)
        ax2.annotate(f'({letters[k+4]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)

        des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
        for i, beta in enumerate(beta_list):
            file_multi = des + f'm={N}_beta={beta}.csv'
            data = np.array(pd.read_csv(file_multi, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            y_multi = np.array(yi[index_select],float)
            kmean_list = np.array(kmean[index_select], float)
            h1_list = np.array(h1[index_select], float)
            beta_unweighted_list = np.log10( h1_list)
            beta_unweighted_list = h1_list
            error_mean = []
            error_std = []
            for j, m in enumerate(m_list):
                file_group = des + f'm={m}_beta={beta}.csv'
                data = np.array(pd.read_csv(file_group, header=None))
                d, seed, kmean, h1, h2, yi = data.transpose()
                index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
                y_group = np.array(yi[index_select],float)
                error = np.round(np.abs(y_multi - y_group), 10) / (y_multi + y_group)
                #ax.plot(m*np.ones(len(error)), error, 'o', markersize=5, alpha=0.5, color=colors[j])
                error_mean.append(np.mean(error) )
                error_std.append(np.std(error) )
            error_mean = np.array(error_mean)
            error_std = np.array(error_std)
            ax2.plot(m_list, error_mean, linewidth=3, alpha=0.8, label=f'$\\beta={int(beta)}$', linestyle=(0, linestyles[i]) )
            ax2.set_xscale('log')
            ax2.set_yscale('symlog', linthresh=1e-6)
            ax2.set_ylim(-2e-7, 1)
            if k > 0:
                ax2.set_yticks([])
            ax2.tick_params(labelsize=ticksize*0.6)
            ax2.legend(fontsize=legendsize*0.5, frameon=False, loc=4, bbox_to_anchor=(1.38,0) ) 
            
    "heatmap: the distance to beta_c"
    for (k, dynamics), beta_c, beta_heatmap in zip(enumerate(dynamics_list), beta_c_list, beta_heatmap_list):
        ax3 = fig.add_subplot(gs[2, k])
        simpleaxis(ax3)
        ax3.annotate(f'({letters[k+8]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)

        des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ygl_beta/'
        error_list = []
        for i, beta in enumerate(beta_heatmap):
            if type(beta) == float:
                beta_str = "{.1f}".format(beta)
            else:
                beta_str = str(beta)

            file_multi = des + f'm={N}_beta=' + beta_str + '.csv'
            data = np.array(pd.read_csv(file_multi, header=None))
            d, seed, kmean, h1, h2, yi = data.transpose()
            index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
            y_multi = np.array(yi[index_select],float)
            kmean_list = np.array(kmean[index_select], float)
            h1_list = np.array(h1[index_select], float)
            beta_unweighted_list = np.log10(h1_list)
            beta_unweighted_list = h1_list
            error_mean = []
            error_std = []
            for j, m in enumerate(m_list):
                file_group = des + f'm={m}_beta=' + beta_str + '.csv'
                data = np.array(pd.read_csv(file_group, header=None))
                d, seed, kmean, h1, h2, yi = data.transpose()
                index_select = df_select_d_sort_seed(d, seed, d_list, seed_list)
                y_group = np.array(yi[index_select],float)
                error = np.round(np.abs(y_multi - y_group), 10) / (y_multi + y_group)
                #ax.plot(m*np.ones(len(error)), error, 'o', markersize=5, alpha=0.5, color=colors[j])
                error_mean.append(np.mean(error) )
                error_std.append(np.std(error) )
            error_mean = np.array(error_mean)
            error_std = np.array(error_std)
            error_list.append(error_mean)

        error_list = np.vstack((error_list))
        error_list[error_list < 1e-3] = 1e-3

        m_ticks = m_list[::5]
        beta_ticks = beta_heatmap[::len(beta_heatmap) // 4] - beta_c
        sns.heatmap(error_list.transpose()[::-1], vmin=0, vmax=np.max(error_list), cmap='YlGnBu', linewidths=0, norm=LogNorm(1e-2, 1), ax=ax3, cbar = k == len(dynamics_list)-1)

        ax3.set_yticks(np.arange(0, len(m_list))[::5]+4.5)
        ax3.set_yticklabels(m_ticks[::-1], rotation=-20)
        ax3.set_xticks(np.arange(len(beta_heatmap))[::len(beta_heatmap) // 4] + 0.5)
        ax3.set_xticklabels(beta_ticks, rotation=20)
        if k > 0:
            ax3.set_yticks([])
        ax3.tick_params(labelsize=ticksize*0.6)

    ax3.annotate('$\\mathrm{Err} (y^{(\\mathrm{gl})})$', xy=(1.40, 0.38), xycoords="axes fraction", size=labelsize*0.5, rotation=90)


    """
    for (k, dynamics), beta_list in zip(enumerate(dynamics_list), beta_list_lists):
        ax3 = fig.add_subplot(gs[2, k])
        simpleaxis(ax3)
        ax3.annotate(f'({letters[8+k]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        h_plot, kmean_plot, m_opt_list = m_opt_fun(dynamics )
        ax3.scatter(x=h_plot/kmean_plot, y=h_plot+kmean_plot, s=10, c=np.log(1+m_opt_list), cmap='Reds')

    """

    xlabel = '$ m $'
    ylabel = '$ \\mathrm{Err} (y^{(\\mathrm{gl})})$'
    fig.text(x=0.01, y=0.55, verticalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
    fig.text(x=0.01, y=0.85, verticalalignment='center', s='$y^{(\\mathrm{gl})} (m=1)$', size=labelsize*0.6, rotation=90)
    fig.text(x=0.03, y=0.18, verticalalignment='center', s='$m$', size=labelsize*0.6, rotation=90)
    fig.text(x=0.5, y=0.36, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    fig.text(x=0.5, y=0.70, horizontalalignment='center', s='$\\beta$', size=labelsize*0.6)
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s='$\\beta^{\\mathrm{wt}} - \\beta_c^{\\mathrm{wt}} $', size=labelsize*0.6)

    #plt.tick_params(labelsize=16*0.9)
    plt.subplots_adjust(left=0.10, right=0.92, wspace=0.50, hspace=0.55, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type +  '_subplots_dis_err_std_noabs.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    
def plot_xs_onenet_update(network_type, N, d, seed, dynamics, m_list, space, weight_list, weight_plot, threshold_value, error_threshold_list):
    """TODO: Docstring for plot_P_w.

    :weight_list: TODO
    :returns: TODO

    """    
    #fig, axes = plt.subplots(3, len(weight_plot), sharex=False, sharey=False, figsize=(4*len(weight_plot) , 3*3 ) )
    fig = plt.figure( figsize=(4*len(weight_plot) , 13 ))
    gs = mpl.gridspec.GridSpec(nrows=5, ncols=6, height_ratios=[1.2, 0.7, 1, 1, 1])
    letters = list('abcdefghijklmnopqrstuvwxyz')
    markers = ['o', '^', 's', 'p', 'P', 'h']
    linestyles = [(j, i) for i in [1, 3, 6, 9, 12] for j in [2, 5, 8, 11]]
    colors=sns.color_palette('hls', 40)
    alphas = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    sizes = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
    des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/'
    xs_multi_file = des + 'xs_multi/' + f'N={N}_d={d}_seed={seed}.csv'
    data_multi = np.array(pd.read_csv(xs_multi_file, header=None))
    weights_multi = data_multi[:, 0]
    index = [np.where(np.abs(weights_multi - weight) < 1e-02 )[0][0]  for weight in weight_list]
    xs_multi = data_multi[index, 1:]
    y_multi = betaspace(A_unit, xs_multi)[-1]
    y_group_list = dict()
    xs_group_list = dict()
    groups_node_nums = dict()
    y_group_list[N] = xs_multi
    xs_group_list[N] = xs_multi
    groups_node_nums[N] = np.array(np.ones(N), int)
    for m in m_list[:-1]:
        xs_group_file = des + f'degree_kmeans_space={space}/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data_group = np.array(pd.read_csv(xs_group_file, header=None))
        group_index = group_index_from_feature_Kmeans(feature, m)
        groups_node_nums[m] = np.array([len(i) for i in group_index])
        weights_group = data_group[:, :1]
        index = [np.where(np.abs(weights_group - weight) < 1e-02 )[0][0]  for weight in weight_list]
        y_group = data_group[index, 1:]
        xs_group = np.zeros( (len(y_group), N) )
        for i, group_i in enumerate(group_index):
            xs_group[:, group_i] = y_group[:, i:i+1]
        y_group_list[m] = y_group
        xs_group_list[m] = xs_group

    colormap = plt.get_cmap("RdBu")
    norm = mpl.colors.LogNorm(vmin=0.1, vmax=10)
    m_3D = [N, 4, 1]
    weight_3D = 0.2
    x_pos = np.random.random(N) * 5
    y_pos = np.random.random(N) * 5
    index_3D = np.where(np.abs(weights_multi - weight_3D)<1e-5)[0][0]

    ax1, ax2 = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    flower_bee(ax1, ax2)
    ax1.annotate(f'({letters[0]})', xy=(-0.0, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    ax1.annotate('Bipartite Network', xy=(0.2, 1.03), xycoords="axes fraction", size=labelsize*0.5)
    ax1.annotate('$M_{ij}$', xy=(0.95, 0.5), xycoords="axes fraction", size=labelsize*0.5)
    ax2.annotate(f'({letters[1]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
    ax2.annotate('$A_{ij}$', xy=(-0.05, 0.5), xycoords="axes fraction", size=labelsize*0.5)
    ax2.annotate('$B_{ij}$', xy=(0.55, 0.5), xycoords="axes fraction", size=labelsize*0.5)
    ax2.annotate('Projection Network', xy=(0.2, 1.03), xycoords="axes fraction", size=labelsize*0.5)

    for i, m in enumerate(m_3D):
        ax_nx = fig.add_subplot(gs[1, i*2])
        ax = fig.add_subplot(gs[1, i*2+1], projection='3d')
        ax_nx.annotate(f'({letters[i+2]})', xy=(-0.58, 1.09), xycoords="axes fraction", size=labelsize*0.6)
        if m == N:
            ax_nx.annotate(f'$m=N$', xy=(1., 1.03), xycoords="axes fraction", size=labelsize*0.6)
        else:
            ax_nx.annotate(f'$m={m}$', xy=(1., 1.03), xycoords="axes fraction", size=labelsize*0.6)
        group_index = group_index_from_feature_Kmeans(feature, m)
        if m == N :
            A_reduced = A_unit
        else:
            A_reduced, _, _ = reducednet_effstate(A_unit, np.random.random(N), group_index)
        #axes[0, i].remove()
        #ax = fig.add_subplot(3, len(weight_plot), i+1, projection='3d')
        if m == N:
            edgelists = [ (ni, nj) for ni in range(m) for nj in range(ni) if A_reduced[ni, nj]]
            edge_width = 0.1
        else:
            edgelists = [ (ni, nj) for ni in range(m) for nj in range(m) if A_reduced[ni, nj]]

            widths = np.log(A_reduced / A_reduced.max() + 1.5) * 4
            edge_width = {(u, v) :  widths[v, u] * 0.5 if u == v else widths[v, u] for (u, v) in edgelists}

        node_list = np.arange(m)
        node_sizes = np.log(groups_node_nums[m] +1.0) / 1.8 
        scale = (1, 1)
        if m == N:
            node_sizes = np.log(1+np.sum(A_reduced, 1)) 
            node_edge_width = 0
            node_size_dict=  {u: node_sizes[u] for u in node_list}
            radius = 0.03
        elif m == 1:
            node_layout = {0:(0.7, 0.5)}
            node_size_dict = {0:15}
            radius = 0.2
            edge_width = 4
            node_edge_width=3
            ax_nx.set_xlim(0, 1)
            ax_nx.set_ylim(0, 1)
        else:
            node_sizes = np.log(6+np.sum(A_reduced, 1)) /1
            node_edge_width = 0.5
            node_size_dict=  {u: node_sizes[u] for u in node_list}
            node_layout = 'spring'
            radius = 0.04
        color_i = colors[np.where(m==np.array(m_list))[0][0]]
        if m == N:
            g = Graph(edgelists, edge_width=edge_width, arrows=True, node_layout=nx.spring_layout(G), node_edge_width=node_edge_width, node_size=node_size_dict, edge_layout='straight', node_color=color_i, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.5, scale=scale, ax=ax_nx, edge_layout_kwargs={'selfloop_radius': radius})
        else:
            g = Graph(edgelists, edge_width=edge_width, arrows=True, node_layout=node_layout, node_edge_width=node_edge_width, node_size=node_size_dict, edge_layout='straight', node_color=color_i, node_alpha=0.8, edge_color='tab:grey', edge_alp=0.5, scale=scale, ax=ax_nx, edge_layout_kwargs={'selfloop_radius': radius})

        if m == N:
            xs_data = xs_multi[index_3D]
        else:
            xs_data = xs_group_list[m][index_3D]
        ax.scatter(x_pos, y_pos, xs_data, s=1, c=colormap(norm(xs_data)) )
        for ni, nj in list(G.edges):
            ax.plot([x_pos[ni], x_pos[nj]], [y_pos[ni], y_pos[nj]], [xs_data[ni], xs_data[nj]], linewidth=0.1, alpha=0.2, color='tab:grey')
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=20, azim=135)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0, 5, 10])
        ax.tick_params(axis='z', which='major', pad=-2)

    for j, weight in enumerate(weight_plot):
        index_plot = np.where(np.abs(weight_list - weight) < 1e-02 )[0][0] 
        ax = fig.add_subplot(gs[2, j*2:(j+1)*2])
        ax.set_ylim(0.09, 13)
        ax.annotate(f'({letters[j+5]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
        simpleaxis(ax)
        for k, m in enumerate(m_list):
            y = y_group_list[m][index_plot]
            ax.scatter(x=np.ones(len(y)) * m, y=y, s= (groups_node_nums[m] / np.sum(groups_node_nums[m]) + 0.05) * 100, alpha=np.log(min(m_list)+0.5) / np.log(m+0.5), color=colors[k]) 
        ax.set(xscale='log', yscale='log')
           
        title_name = f'$w={weight}$'
        ax.set_title(title_name, size=labelsize*0.5)
        ax.set_xlabel('$m$', size=labelsize*0.5)
        ax.set_ylabel('$y(m)$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[3, 0:2])
    simpleaxis(ax)
    ax.annotate(f'({letters[8]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    m_ygl = [1, 4, 16, 64, 256, N]
    for i_m, m in enumerate(m_ygl):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        xs_group_m = xs_group_list[m]
        y_gl = betaspace(A_unit, xs_group_m)[-1]
        ax.plot(weight_list, y_gl, linewidth=3, alpha=0.8, color=colors[m_index], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
    ax.set_title(title_name, size=labelsize*0.5)
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc=4, bbox_to_anchor=(1.01,0) ) 
    ax.set(yscale='log')
    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$y_{(\\mathrm{gl})}$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[3, 2:4])
    ax.annotate(f'({letters[9]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    simpleaxis(ax)
    for i_m, m in enumerate(m_ygl[:-1]):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        prediction_ratios = []
        for i_w, weight in enumerate(weight_list):
            high_nodes_multi = np.where(xs_group_list[N][i_w] > threshold_value)[0]
            low_nodes_multi = np.where(xs_group_list[N][i_w] < threshold_value)[0]
            high_nodes_group = np.where(xs_group_list[m][i_w] > threshold_value)[0]
            low_nodes_group = np.where(xs_group_list[m][i_w] < threshold_value)[0]
            prediction_ratio = (len(np.intersect1d(high_nodes_multi, high_nodes_group))  + len(np.intersect1d(low_nodes_multi, low_nodes_group)) )/ N
            prediction_ratios.append(prediction_ratio)
        ax.plot(weight_list, prediction_ratios, linewidth=3, alpha=0.8,  color=colors[m_index], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc=4, bbox_to_anchor=(1.01,0) ) 

    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('precision', size=labelsize*0.45)

    ax = fig.add_subplot(gs[3, 4:6])
    simpleaxis(ax)
    ax.annotate(f'({letters[10]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    ygl_group_list = []
    for m in m_list:
        xs_group = xs_group_list[m]
        ygl_group = betaspace(A_unit, xs_group)[-1]
        ygl_group_list.append(ygl_group)
    ygl_list = np.vstack( (ygl_group_list) ) 
    m_ticks = m_list[::5]
    weight_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    sns.heatmap(ygl_list[::-1], vmin=0, vmax=np.max(ygl_list.max()), cmap='YlGnBu', linewidths=0, ax=ax)
    ax.set_yticks(np.arange(0, len(m_list))[::5]-0.5)
    ax.set_yticklabels(m_ticks[::-1], rotation=-20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    ax.set_xticklabels(weight_ticks, rotation=20)

    #ax.set_title('$y^{(\\mathrm{gl})}$', size=labelsize*0.5)
    ax.annotate('$y^{(\\mathrm{gl})}$', xy=(1.25, 0.4), xycoords="axes fraction", size=labelsize*0.5, rotation=90)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$m$', size=labelsize*0.5)


    ax = fig.add_subplot(gs[4, 0:2])
    simpleaxis(ax)
    ax.annotate(f'({letters[11]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    xs_multi = xs_group_list[N]
    y_multi = betaspace(A_unit, xs_multi)[-1]
    m_ygl = [1, 4, 16, 64, 256]
    for i_m, m in enumerate(m_ygl):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        xs_group_m = xs_group_list[m]
        y_gl = betaspace(A_unit, xs_group_m)[-1]
        error = np.abs(y_gl - y_multi) / np.abs(y_gl + y_multi)
        ax.plot(weight_list, error, linewidth=3, alpha=0.8, color=colors[m_index], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc='upper right' ) 
    #ax.set(yscale='log')
    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$\\mathrm{Err} (y_{(\\mathrm{gl})})$', size=labelsize*0.5)

    ax = fig.add_subplot(gs[4, 2:4])
    simpleaxis(ax)
    ax.annotate(f'({letters[12]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    xs_multi = xs_group_list[N]
    y_multi = betaspace(A_unit, xs_multi)[-1]
    error_list = []
    for i_m, m in enumerate(m_list):
        m_index = np.where(np.abs(m-np.array(m_list)) < 1e-3)[0][0]
        xs_group_m = xs_group_list[m]
        y_gl = betaspace(A_unit, xs_group_m)[-1]
        error = np.abs(y_gl - y_multi) / np.abs(y_gl + y_multi)
        error_list.append(error)
    error_list = np.vstack( (error_list) )
    colors_error = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f' ]
    for i_e, error_threshold in enumerate(error_threshold_list):
        m_opt_list = []
        for i_w, w in enumerate(range(error_list.shape[-1])):
            index_error = np.where(error_list[:, i_w] < error_threshold)[0]
            if len(index_error):
                m_opt = m_list[index_error[0]]
            else:
                m_opt = N
            m_opt_list.append(m_opt)
        ax.plot(weight_list, m_opt_list, linewidth=2, alpha=0.7, color=colors_error[i_e], label=f'$R_e={error_threshold}$', linestyle=(0, linestyles[i_e]) ) 
    ax.legend(fontsize=legendsize*0.4, frameon=False, loc='upper right' ) 
    #ax.legend(fontsize=legendsize*0.4, frameon=False, loc=4, bbox_to_anchor=(1.01,0) ) 
    #ax.set(yscale='log')
    ax.set_title('', size=labelsize*0.5)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$m_{\\mathrm{opt}}$', size=labelsize*0.5)
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[4, 4:6])
    ax.annotate(f'({letters[13]})', xy=(-0.1, 1.09), xycoords="axes fraction", size=labelsize*0.6)
    simpleaxis(ax)
    xs_multi = xs_group_list[N]
    ygl_multi = betaspace(A_unit, xs_multi)[-1]
    error_list = []
    for m in m_list[:-1]:
        xs_group = xs_group_list[m]
        ygl_group = betaspace(A_unit, xs_group)[-1]
        error = np.round(np.abs(ygl_multi-ygl_group), 10) / (np.abs(ygl_multi + ygl_group) )
        error_list.append(error)
    error_list = np.vstack( (error_list) ) 
    m_ticks = m_list[:-1][::5]
    weight_ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    sns.heatmap(error_list[::-1], vmin=0, vmax=np.max(1), linewidths=0, ax=ax)
    ax.set_yticks(np.arange(1, len(m_list)-1)[::5]+1.5)
    ax.set_yticklabels(m_ticks[::-1], rotation=-20)
    ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    ax.set_xticklabels(weight_ticks, rotation=20)

    ax.annotate('$\\mathrm{Err} (y^{(\\mathrm{gl})})$', xy=(1.29, 0.3), xycoords="axes fraction", size=labelsize*0.5, rotation=90)
    ax.set_xlabel('$w$', size=labelsize*0.5)
    ax.set_ylabel('$m$', size=labelsize*0.5)


    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.55, hspace=0.55, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + network_type + '_subplots_xs_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    return None

def flower_bee(ax1, ax2):
    """TODO: Docstring for flower_bee.

    :): TODO
    :returns: TODO

    """
    img_des = '../flower_bee/'
    bees = ['bee0.png', 'bee1.png', 'bee2.png', 'bee3.png', 'bee4.png', 'bee5.png']
    flowers = ['cherry-blossom.png', 'flower.png', 'jasmine.png', 'pink-cosmos.png']

    connections = [[0, 0], [0, 2], [1,0], [1, 1], [1, 3], [2, 3], [3, 2], [3, 3], [4, 0], [4, 2], [5, 1], [5, 3]]
    "bipartite network"
    bee_dict = defaultdict(set)
    flower_dict = defaultdict(set)
    for connection_i in connections:
        bee_dict[connection_i[0]].add(connection_i[1])
        flower_dict[connection_i[1]].add(connection_i[0])
    flower_net = []
    bee_net = []
    for bee_i in range(6):
        for bee_j in range(bee_i, 6):
            if len(bee_dict[bee_i] | bee_dict[bee_j]) < len(bee_dict[bee_i]) + len(bee_dict[bee_j]):
                bee_net.append([bee_i, bee_j])

    for flower_i in range(4):
        for flower_j in range(flower_i, 4):
            if len(flower_dict[flower_i] | flower_dict[flower_j]) < len(flower_dict[flower_i]) + len(flower_dict[flower_j]):
                flower_net.append([flower_i, flower_j])

    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    bee_coordinate = [[100, 20], [200, 20], [300, 20], [400, 20], [500, 20], [600, 20]]
    flower_coordinate = [[150, 150], [280, 150], [410, 150], [540, 150]]
    bee_connector = [[xi+23, yi +60] for xi, yi in bee_coordinate]
    flower_connector = [[xi+23, yi - 5] for xi, yi in flower_coordinate]
    ax1.set_xlim(50, 650)
    ax1.set_ylim(10, 210)
    for i, bee_i in enumerate(bees):
        img = mpimg.imread(img_des + bee_i)
        tx, ty = bee_coordinate[i]
        ax1.imshow(img, extent=(tx, tx+50, ty, ty+50))

    for i, flower_i in enumerate(flowers):
        img = mpimg.imread(img_des + flower_i)
        tx, ty = flower_coordinate[i]
        ax1.imshow(img, extent=(tx, tx+50, ty, ty+50))
    
    for connection_i in connections:
        i, j = connection_i 
        ax1.plot([bee_connector[i][0], flower_connector[j][0]], [bee_connector[i][1], flower_connector[j][1]], color='#66c2a5', linewidth=1.0, alpha=0.9)


    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlim(-320, 360)
    ax2.set_ylim(-10, 320)
    flower_bipar =[[-300, 50], [-60, 50], [-60, 250], [-300, 250] ]
    for i, flower_i in enumerate(flowers):
        img = mpimg.imread(img_des + flower_i)
        tx, ty = flower_bipar[i]
        ax2.imshow(img, extent=(tx, tx+50, ty, ty+50))
    
    flower_bi_connector = [[xi+70, yi+25] if xi == -300 else [xi-5, yi+25] for xi, yi in flower_bipar ]
    for i, j in flower_net:
        ax2.plot([flower_bi_connector[i][0], flower_bi_connector[j][0]], [flower_bi_connector[i][1], flower_bi_connector[j][1]], color='#66c2a5', linewidth=1.0, alpha=0.7)



    bee_bipar =[[100, 100], [200, 40], [300, 100], [300, 180], [200, 250], [100, 180] ]
    bee_bi_connector = [[100+60, 100 + 25], [200 + 25, 40 + 60], [300-10, 100 + 25], [300-10, 180 + 25], [200 + 25, 250-10], [100 + 60, 180 + 25] ]
    for i, bee_i in enumerate(bees):
        img = mpimg.imread(img_des + bee_i)
        tx, ty = bee_bipar[i]
        ax2.imshow(img, extent=(tx, tx+50, ty, ty+50))

    for i, j in bee_net:
        ax2.plot([bee_bi_connector[i][0], bee_bi_connector[j][0]], [bee_bi_connector[i][1], bee_bi_connector[j][1]], color='#66c2a5', linewidth=1.0, alpha=0.9)





N = 1000
dynamics = 'mutual'

network_type = 'SF'
gamma_list = [round(i, 1) if i%1 > 1e-5 else int(i) for i in np.hstack(( np.arange(2.1, 5.01, 0.2) , np.arange(6, 20.1, 1) )) ]
kmin_list = [3, 4, 5]
d_list = [[gamma, N-1, kmin] for gamma in gamma_list for kmin in kmin_list]
seed_list = np.arange(10).tolist()

space = 'log'

m_list = np.unique(np.array(np.round([(2**1) ** i for i in range(40)], 0), int) )
m_list = m_list[m_list < N/2].tolist() + [N]
beta_list = np.round(np.arange(0.1, 8.1, 0.1), 5)





dynamics = 'CW_high'
beta_list = [5, 10, 15, 20, 25]

dynamics = 'genereg'
beta_list = [1.5, 2.0, 3.0, 4.0, 5.0]

dynamics = 'CW'
beta_list = [20, 30, 40, 50, 60]

dynamics = 'mutual'
beta_list = [1.0, 3.0, 5.0, 7.0, 9.0]

m_list = [1, 4, 16, 64, 256, 512]
#y_compare_beta_m(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list)
#y_compare_beta(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list)

#y_bifurcation(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list)

dynamics_list = ['mutual', 'CW', 'genereg', 'CW_high']
arguments_list = [(B, C, D, E, H, K_mutual),(a, b),(B_gene, ), (a, b)]
initial_condition_list = [0.05, 0, 10, 10]
beta_list_lists = [[1.0, 3.0, 5.0, 7.0, 9.0], [20, 30, 40, 50, 60], [1.5, 2.0, 3.0, 4.0, 5.0], [5, 10, 15, 20, 25]]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )
#yerror_m_beta(network_type, N, d_list, seed_list, dynamics_list, m_list, space, beta_list_lists)
beta_1D_list = [np.round(np.arange(0.5, 20.1, 1), 5), np.arange(5, 100, 1), np.round(np.arange(1., 40.1, 0.2), 5), np.arange(1, 100, 1)]
beta_c_list = [7, 58, 2, 8]
beta_heatmap_list = [np.round(np.arange(0.5, 10.1, 0.5), 5), np.arange(10, 66, 5), np.round(np.arange(1.5, 5.1, 0.5), 5), np.arange(6, 31, 2)]

#yerror_m_beta_illustration(network_type, N, d_list, seed_list, beta_1D_list, dynamics_list, arguments_list, initial_condition_list, m_list, space, beta_list_lists, beta_c_list, beta_heatmap_list)


d = [2.5, 999, 3]
seed = [1, 1]
weight_list = np.arange(0.01, 0.6, 0.01)
weight_plot = [0.15, 0.2, 0.25, 0.3]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )
m_list = m_list[m_list<N].tolist() + [N]

threshold_value = 1
#plot_xs_onenet(network_type, N, d, seed, dynamics, m_list, space, weight_list, weight_plot, threshold_value)




beta_list = np.round(np.arange(0.1, 8.1, 0.1), 5)
m_list = [1, 4, 16, 64, 256, N]
#y_bifurcation_3d(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list)

m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]
weight_plot = [0.15, 0.2, 0.25]
error_threshold_list = [0.01, 0.05, 0.1, 0.5]

plot_xs_onenet_update(network_type, N, d, seed, dynamics, m_list, space, weight_list, weight_plot, threshold_value, error_threshold_list)



#sns.scatterplot(x=h_plot/kmean_plot, y=h_plot+kmean_plot, hue=np.log(1+m_opt_list), s=10)



"""
h_plot, kmean_plot, m_opt_list, d_seed_plot = m_opt_beta_w(dynamics)
simi_func = lambda x, y: np.sum(np.abs(x-y) / np.abs(x+y) )
simi_m = np.zeros( (len(m_opt_list), len(m_opt_list)) )
simi_k = np.zeros( (len(m_opt_list), len(m_opt_list)) )
network_type = 'SF'

k_list = []
for d, seed, in d_seed_plot:
    Ai, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, eval(seed), eval(d))
    ki = np.sum(Ai, 0)
    k_list.append(ki)

k_indicator_fun = lambda k: np.mean(k ** 2)  / np.mean(k)/ ((np.mean(k** 2) / np.mean(k) + np.mean(k)) )
k_indicator = np.zeros((len(m_opt_list)) )
for i in range(len(m_opt_list) ):
    k_indicator[i] = k_indicator_fun(k_list[i]) 
for i in range(len(m_opt_list)):
    for j in range(len(m_opt_list)):
        simi_m[i, j] = simi_func(m_opt_list[i], m_opt_list[j])
        #simi_m[i, j] = simi_func(y_multi_list[i], y_multi_list[j])
        simi_k[i, j] = simi_func(k_indicator[i], k_indicator[j])
masks = ~np.eye(len(m_opt_list) , dtype=bool)
plt.scatter(simi_m[masks], simi_k[masks])

y_multi_list = []
des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
des_multi = des + 'y_multi_beta/'
des_group = des + 'y_group_beta/'
for d, seed, in d_seed_plot:
    file_multi = des_multi + f'N={N}_d={eval(d)}_seed={eval(seed)}.csv'
    data = np.array(pd.read_csv(file_multi, header=None))
    w, beta, y_multi = data.transpose()
    index_sort = np.argsort(w)
    w, beta, y_multi = w[index_sort], beta[index_sort], y_multi[index_sort]
    y_multi_list.append(y_multi)


y_multi_list = np.array(y_multi_list)
index_plot = np.where(4< k_indicator)[0]
plt.plot(y_multi_list[index_plot].transpose())

"""
"""
rows, cols = 3, 4
fig = plt.figure( figsize=(12, 6 ))
group_1 = [0, 142, 158, 79, 174]
group_2 = [47, 50, 100, 108, 115]
group_3 = [20, 86, 145, 148, 171]
background = [i for i in range(len(m_opt_list)) if i not in group_1 + group_2 + group_3 ]
gs = mpl.gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1])
ax = fig.add_subplot(gs[0, 0])
simpleaxis(ax)
ax.plot(beta, np.vstack((np.array(m_opt_list)[background])).transpose(), color='grey', alpha=0.3, linewidth=0.5) 
ax.plot(beta, np.array(m_opt_list)[group_1[0]], color='tab:blue', alpha=0.8, linewidth=2, label='sample1')
ax.plot(beta, np.vstack((np.array(m_opt_list)[group_1[1:]])).transpose(), color='tab:blue', alpha=0.8, linewidth=2)
ax.plot(beta, np.array(m_opt_list)[group_2[0]], color='tab:red', alpha=0.8, linewidth=2, label='sample2')
ax.plot(beta, np.vstack((np.array(m_opt_list)[group_2[1:]])).transpose(), color='tab:red', alpha=0.8, linewidth=2)
ax.plot(beta, np.array(m_opt_list)[group_3[0]], color='tab:green', alpha=0.8, linewidth=2, label='sample3')
ax.plot(beta, np.vstack((np.array(m_opt_list)[group_3[1:]])).transpose(), color='tab:green', alpha=0.8, linewidth=2)
ax.set_yscale('log')
ax.legend(fontsize=12, frameon=False, loc=4, bbox_to_anchor=(1.05,0.4) ) 

ax = fig.add_subplot(gs[1, 0])
simpleaxis(ax)
for net_i in background + group_1 + group_2 + group_3:
    ki = k_list[net_i]
    bins = np.logspace(1, 30, 30, base=1.2)
    x, y = np.histogram(ki, bins)
    x = np.cumsum(x[::-1])[::-1]
    h = np.mean(ki ** 2) / np.mean(ki)
    counter = Counter(ki)
    if net_i in group_1:
        color = 'tab:blue'
        linewidth = 2
        alpha = 0.8
    elif net_i in group_2:
        color = 'tab:red'
        linewidth = 2
        alpha = 0.8
    elif net_i in group_3:
        color = 'tab:green'
        linewidth = 2
        alpha = 0.8
    else:
        color = 'tab:grey'
        linewidth=0.5
        alpha = 0.3
    ax.plot(y[np.where(x>800)[0][-1]: np.where(x>0)[0][-1]], x[np.where(x>800)[0][-1]: np.where(x>0)[0][-1]], color=color, linewidth=linewidth, alpha=alpha)
    ax.set_yscale('log')
    ax.set_xscale('log')


for i, group_i in enumerate([group_1, group_2, group_3]):
    ax = fig.add_subplot(gs[0, 1+i])
    simpleaxis(ax)
    ax.annotate('sample' + f'{i+1}', xy=(0.3, 1.05), xycoords="axes fraction", size=labelsize*0.5)
    ax.plot(beta, np.vstack((np.array(m_opt_list)[group_i])).transpose(), alpha=0.8, linewidth=2)
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[1, 1+i])
    simpleaxis(ax)
    for i, net_i in enumerate(group_i):
        ki = k_list[net_i]
        bins = np.logspace(1, 30, 30, base=1.2)
        x, y = np.histogram(ki, bins)
        x = np.cumsum(x[::-1])[::-1]
        h = np.mean(ki ** 2) / np.mean(ki)
        counter = Counter(ki)
        plt.scatter(y[np.where(x>800)[0][-1]:][1:], x[np.where(x>800)[0][-1]:], label=f'h={round(h,1)}', s=20)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(fontsize=10, frameon=False, loc=4, bbox_to_anchor=(1.08,0.48) ) 
        ax.set_yticklabels([])

fig.text(x=0.5, y=0.01, horizontalalignment='center', s='$k$', size=labelsize*0.6)
fig.text(x=0.5, y=0.5, horizontalalignment='center', s='$\\beta$', size=labelsize*0.6)
fig.text(x=0.04, y=0.3, horizontalalignment='center', s='$P(k)$', size=labelsize*0.6)
fig.text(x=0.04, y=0.75, horizontalalignment='center', s='$m_{\\mathrm{opt}}$', size=labelsize*0.6)
plt.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.45, bottom=0.1, top=0.95)
"""
