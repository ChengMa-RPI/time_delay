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

fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
colors = ['#fc8d62',  '#66c2a5', '#e78ac3']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_tau_c_m(ax, seed, weight_list):
    simpleaxis(ax)
    for weight in weight_list:
        group_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_group.csv'
        data = np.array(pd.read_csv(group_file, header=None))
        m_list, tau_group = data.transpose()
        ax.semilogx(m_list, tau_group, label = f'w={weight}', linewidth=2.5, alpha=0.8)
        multi_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_eigen.csv'
        data = np.array(pd.read_csv(multi_file, header=None))
        tau_multi = data[0, 0]
        #ax.semilogx(m_list, tau_multi * np.ones(len(m_list)), label = f'w={weight}')

        evo_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_evolution.csv'
        #data = np.array(pd.read_csv(evo_file, header=None))
        #tau_evo = data[0, 0]
        #ax.semilogx(m_list, tau_evo * np.ones(len(m_list)), label = f'w={weight}')
    ax.tick_params(axis='both', which='major', labelsize=ticksize*0.6)
    ax.set_xlabel('$m$', fontsize=labelsize*0.5)
    ax.set_ylabel('$\\tau_c$', fontsize=labelsize*0.5)
    ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.01,-0.02) ) 


    return ax

def plot_heatmap_tau_c_m_weight(ax, seed, weight_list):
    simpleaxis(ax)

    weight_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    tau_group = []
    m_opt = []
    for weight in weight_list:
        group_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_group.csv'
        data = np.array(pd.read_csv(group_file, header=None))
        m_list, tau_i = data.transpose()
        tau_group.append(tau_i)

        multi_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_eigen.csv'
        data = np.array(pd.read_csv(multi_file, header=None))
        tau_multi = data[0, 0]
        index = np.where(np.abs(tau_i - tau_multi) < 0.05)[0]
        if len(index):
            m_opt.append(m_list[index[0]])
        else:
            m_opt.append(0)

    tau_group = np.vstack((tau_group)).transpose()
    #sns.heatmap(tau_group[::-1, :], vmin=np.min(tau_group.min()), vmax=np.max(tau_group.max()), linewidths=0, ax=ax, cmap="YlGnBu" )
    sns.heatmap(tau_group[::-1, :], vmin=0, vmax=0.3, linewidths=0, ax=ax, cmap="YlGnBu" )
    m_ticks = np.array(m_list[::5], int)
    ax.set_yticks(np.arange(0, len(m_list))[::5]+0.5)
    ax.set_yticklabels(m_ticks[::-1], rotation=-20)
    #ax.set_xticks(np.array(weight_ticks) / np.unique(np.round(np.diff(weight_list) , 5))- 0.5)
    ax.set_xticks(np.arange(0, len(weight_list))[3:][::4]+0.5)
    ax.set_xticklabels(weight_ticks, rotation=20)

    ax.set_xlabel('$w$', fontsize=labelsize*0.5)
    ax.set_ylabel('$m$', fontsize=labelsize*0.5)

    for w_i, m_i in zip(weight_list, m_opt):
        if m_i:
            x0 = np.where(np.abs(w_i - weight_list) < 1e-5)[0][0]
            y0 = len(m_list) - np.where(np.abs(m_i - m_list) < 1e-5)[0][0] - 0.5
            ax.hlines( y0, x0, x0+1, linestyle='-', linewidth=2, color="red")
            ax.hlines( y0+1, x0, x0+1, linestyle='-', linewidth=2, color="red")
            ax.vlines( x0, y0, y0+1, linestyle='-', linewidth=2, color="red")
            ax.vlines( x0+1, y0, y0+1, linestyle='-', linewidth=2, color="red")
    ax.tick_params(axis='both', which='major', labelsize=ticksize*0.6)
    ax.annotate('$\\tau_c$', xy=(1.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
    return ax

dynamics = 'mutual'
network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
seed = [1, 1]
weight = 1.0
des = '../data/tau_compare/'

rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))
letters = list('abcdefghijklmnopqrstuvwxyz')


seed = [1, 1]

ax = axes[1, 0]
weight_list = [0.05, 0.1, 0.5, 1.0]
plot_tau_c_m(ax, seed, weight_list)
ax.annotate('(e)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax.annotate('net 1', xy=(1.0, 1.15), xycoords="axes fraction", size=labelsize*0.5)

ax = axes[1, 1]
weight_list = np.round(np.arange(0.05, 1.01, 0.05) , 5)
plot_heatmap_tau_c_m_weight(ax, seed, weight_list)
ax.annotate('(f)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)



seed = [0, 0]

ax = axes[1, 2]
weight_list = [0.05, 0.1, 0.5, 1.0]
plot_tau_c_m(ax, seed, weight_list)
ax.annotate('(g)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax.annotate('net 2', xy=(1.0, 1.15), xycoords="axes fraction", size=labelsize*0.5)

ax = axes[1, 3]
weight_list = np.round(np.arange(0.05, 1.01, 0.05) , 5)
plot_heatmap_tau_c_m_weight(ax, seed, weight_list)
ax.annotate('(h)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)


seed = [1, 1]
linestyles = [(i, j) for i in [2] for j in [1, 5, 7]]

weight == 1.0
ax1 = axes[0, 0]
ax2 = axes[0, 1]
simpleaxis(ax1)
simpleaxis(ax2)
delay_list = [0.2, 0.21, 0.22]
for i, delay in enumerate(delay_list):
    data = np.array(pd.read_csv( des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_delay={delay}_evolution.csv', header=None) ) 
    t, xs_delay = data[::10, 0], data[::10, 1:]
    xs_multi = np.array(pd.read_csv(des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_xs.csv', header=None))[:, 0]
    xs_diff = np.abs(xs_delay - xs_multi)
    xs_diff_ave = np.mean(xs_diff, 1)
    ax1.semilogy(t, xs_diff[:, ::100], linewidth=1.5, alpha=0.5, label = f'$\\tau={delay}$', color=colors[i])
    ax2.semilogy(t, xs_diff_ave, linewidth=1.5, alpha=0.8, label = f'$\\tau={delay}$', color=colors[i])

ax1.set_xlabel('$t$', fontsize=labelsize*0.5)
ax2.set_xlabel('$t$', fontsize=labelsize*0.5)
ax2.set_ylabel('$\\langle \Delta x \\rangle $', fontsize=labelsize*0.5)
ax1.set_ylabel('$ \Delta x_i $', fontsize=labelsize*0.5)
ax2.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(-0.78,0.0) ) 
ax1.annotate(f'$w={weight}$', xy=(0.9, 1.07), xycoords="axes fraction", size=labelsize*0.5)
ax1.annotate('(a)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax2.annotate('(b)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax1.tick_params(axis='both', which='major', labelsize=ticksize*0.6)
ax2.tick_params(axis='both', which='major', labelsize=ticksize*0.6)


weight = 0.1
ax1 = axes[0, 2]
ax2 = axes[0, 3]
simpleaxis(ax1)
simpleaxis(ax2)
delay_list = [0.25, 0.26, 0.27]
for i, delay in enumerate(delay_list):
    data = np.array(pd.read_csv( des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_delay={delay}_evolution.csv', header=None) ) 
    t, xs_delay = data[::10, 0], data[::10, 1:]
    xs_multi = np.array(pd.read_csv(des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_xs.csv', header=None))[:, 0]
    xs_diff = np.abs(xs_delay - xs_multi)
    xs_diff_ave = np.mean(xs_diff, 1)
    ax1.semilogy(t, xs_diff[:, ::100], linewidth=1.5, alpha=0.5, label = f'$\\tau={delay}$', color=colors[i])
    ax2.semilogy(t, xs_diff_ave, linewidth=1.5, alpha=0.8, label = f'$\\tau={delay}$', color=colors[i])

ax1.set_xlabel('$t$', fontsize=labelsize*0.5)
ax2.set_xlabel('$t$', fontsize=labelsize*0.5)
ax1.set_ylabel('$\Delta x_i $', fontsize=labelsize*0.5)
ax2.set_ylabel('$\\langle \Delta x \\rangle $', fontsize=labelsize*0.5)
ax2.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(-0.78,0.0) ) 
ax1.annotate(f'$w={weight}$', xy=(0.9, 1.07), xycoords="axes fraction", size=labelsize*0.5)
ax1.annotate('(c)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax2.annotate('(d)', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
ax1.tick_params(axis='both', which='major', labelsize=ticksize*0.6)
ax2.tick_params(axis='both', which='major', labelsize=ticksize*0.6)



"""

rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))
letters = list('abcdefghijklmnopqrstuvwxyz')


seed_list = np.arange(1, 9)
for i, seed in enumerate(seed_list):
    seed = [seed, seed]
    ax = axes[i // 4, i % 4]
    weight_list = [0.05, 0.1, 0.5, 1.0]
    weight_list = np.round(np.arange(0.05, 1.01, 0.05) , 5)
    plot_heatmap_tau_c_m_weight(ax, seed, weight_list)
    ax.annotate(f'({letters[i]})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)


"""

plt.subplots_adjust(left=0.1, right=0.95, wspace=0.35, hspace=0.45, bottom=0.1, top=0.95)
