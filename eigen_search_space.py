from dimension_reduction_delay_oop import Tau_Solution
import numpy as np 
from scipy.optimize import fsolve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
colors=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ]
sns_pos = sns.color_palette("rocket", as_cmap=True)
sns_neg = sns.color_palette("mako", as_cmap=True)
markersize = 2
markers = ['o', '*', '^', 'P', 'd', 's']
legendsize = 14
fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
alpha = 0.8
lw = 3

title_letters = list('abcdefghijklmnopqrstuvwxyz')

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def tau_eigen_data(network_type, N, d, seed, weight, m, dynamics, arguments, attractor_value, tau_list, nu_list):
    """TODO: Docstring for eigen_fg.

    :returns: TODO

    """
    ts = Tau_Solution(network_type, N, d, seed, [m], dynamics, [weight], arguments, attractor_value, tau_list, nu_list, delay1=None, delay2=None)
    tau_m, tau_sol = ts.tau_eigen(weight, m)
    return tau_m, tau_sol

def tau_eigen_initial(network_type, N, d, seed, weight, m, dynamics):
    """TODO: Docstring for eigen_fg.

    :returns: TODO

    """
    des = '../data/tau/' + dynamics + '/' + network_type + '/tau_initial_condition/' 
    des_file  = des + f'N={N}_d={d}_seed={seed}_weight={weight}_m={m}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    tau_sol = data[:, -1]
    initial_conditions = data[:, :2]

    """
    tau_select = np.arange(0.02, 0.9, 0.02)
    nu_select = np.arange(0.2, 20, 0.2)
    index_select = []
    for i, initial_condition in enumerate(initial_conditions):
        if sum(np.abs(initial_condition[0]  - tau_select) <1e-5) and sum(np.abs(initial_condition[1]  - nu_select) <1e-5)  :
            index_select.append(i)
    tau_sol = tau_sol[index_select]
    initial_conditions = initial_conditions[index_select]
    """
    tau_sol = np.round(tau_sol, 3)
    tau_unique = np.unique(tau_sol)
    #color_num = len(tau_unique[(tau_unique)>0&(tau_unique<1)]) //len(markers) + 1
    tau_upperbound = np.sort(tau_unique[tau_unique>0])[6]
    index = np.argsort(tau_sol)
    initial_conditions = initial_conditions[index]
    tau_sol = tau_sol[index]
    tau_map = defaultdict(list)
    for initial_condition, tau_i in zip(initial_conditions, tau_sol):
        if 0< tau_i < tau_upperbound:
            tau_key = tau_i
            tau_map[tau_key].append(initial_condition)
        """
        elif tau_i < 0:
            tau_key = 'neg'
        elif tau_i >= tau_upperbound:
            tau_key = 'large'
        """
    return tau_map

def plot_tau_all_initial(ax, tau_map):
    """TODO: Docstring for plot_tau_all_initial.

    :ax: TODO
    :: TODO
    :returns: TODO

    """
    simpleaxis(ax)
    color_num = 7
    sns_color = sns.color_palette("husl", color_num)
    tau_unique = list(tau_map.keys())
    color_map = {}
    marker_map = {}
    added_times = {}
    label_map = {}
    j = 0
    for _, tau_key in enumerate(tau_unique):
        if tau_key == 'neg':
            color = 'tab:grey'
            marker = 'o'
            label = '$\\tau<0$'

        else:
            marker = markers[j // color_num]
            color = sns_color[j % color_num]
            label = f'$\\tau={round(tau_key, 3)}$'
            j += 1
            
        color_map[tau_key] = color
        marker_map[tau_key] = marker
        label_map[tau_key] = label

    for tau_i in tau_map.keys():
        initial_condition_map = np.vstack(( tau_map[tau_i] ))
        ax.plot(initial_condition_map[:, 0], initial_condition_map[:,  1], color=color_map[tau_i], linestyle= 'None', marker=marker_map[tau_i], markersize=markersize, label=label_map[tau_i])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.375, -0.09), markerscale=3)

    return None

def plot_tau_eigen_nets(network_type, N, d, seed, weight_list, m_eigen, dynamics):
    """TODO: Docstring for plot_tau_eigen_nets.

    :arg1: TODO
    :returns: TODO

    """
    rows = len(weight_list)
    cols = len(m_eigen) 
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))

    for i, weight in enumerate(weight_list):
        tau_m = []
        for j, m in enumerate(m_eigen):
            tau_map = tau_eigen_initial(network_type, N, d, seed, weight, m, dynamics)
            ax = axes[i, j]
            ax.annotate(f'({title_letters[i *cols + j]})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
            if j == 0:
                ax.annotate(f'w={weight}', xy=(-0.42, 0.47), xycoords="axes fraction", size=labelsize*0.5)
            if i == 0:
                ax.set_title(f'm={m}', size=labelsize*0.5)
            plot_tau_all_initial(ax, tau_map)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$\\tau_0$", size=labelsize*0.7)
    fig.text(x=0.01, y=0.5, horizontalalignment='center', s="$\\nu_0$", size=labelsize*0.7)
    fig.subplots_adjust(left=0.11, right=0.92, wspace=0.40, hspace=0.25, bottom=0.15, top=0.95)
    plt.show()

def plot_tau_c_eigen_evolution(network_type, N, d, seed, weight_list, m_compare, dynamics):
    rows = 1
    cols = len(weight_list)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
    for j, weight in enumerate(weight_list):
        ax = axes[j]
        simpleaxis(ax)
        tau_eigen = []
        for m in m_compare:
            tau_map = tau_eigen_initial(network_type, N, d, seed, weight, m, dynamics)
            tau_c = np.min([i  for i in list(tau_map.keys()) if i != 'neg'])
            tau_eigen.append(tau_c)
        filename = '../data/tau/' + dynamics + '/' + network_type + '/' + 'evolution' + f'/N={N}_d={d}_seed={seed}_weight={weight}.csv'
        m_list, tau_evolution = np.array(pd.read_csv(filename, header=None)).transpose()
        tau_evolution_compare = [tau_i for tau_i, m_i in zip(tau_evolution, m_list) if m_i in m_compare]
        ax.set_title(f'w={weight}', size=labelsize*0.5)
        ax.semilogx(m_compare, tau_eigen, label = f'eigen', linewidth=2.5, alpha=0.8)
        ax.semilogx(m_compare, tau_evolution_compare, label = f'evolution', linewidth=2.5, alpha=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4) 
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$m$", size=labelsize*0.7)
    fig.text(x=0.05, y=0.5, horizontalalignment='center', s="$\\tau_c$", size=labelsize*0.7)
    fig.subplots_adjust(left=0.11, right=0.92, wspace=0.40, hspace=0.25, bottom=0.15, top=0.90)
    plt.show()
    return None




B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1
B_gene = 1 
a = 5
b = 1

N = 1000
network_type = 'SF'
d = [2.5, 999, 3]
seed = [0, 0]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
weight_list = np.round(np.arange(0.05, 1.01, 0.05), 5)
attractor_value = 5
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)

d = [2.5, 999, 3]
seed = [0, 0]
weight = 0.4
m = 64
tau_list = np.arange(0.01, 0.8, 0.01)
nu_list = np.arange(0, 20, 0.1)
#tau_c, tau_sol = tau_eigen_data(network_type, N, d, seed, weight, m, dynamics, arguments, attractor_value, tau_list, nu_list)
#tau_map = plot_tau_eigen(network_type, N, d, seed, weight, m, dynamics)
weight_list = [0.2, 0.4, 0.6, 0.8]
m_eigen = [2, 4, 64, 256]
m_compare = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, N]
plot_tau_eigen_nets(network_type, N, d, seed, weight_list, m_eigen, dynamics)
#plot_tau_c_eigen_evolution(network_type, N, d, seed, weight_list, m_compare, dynamics)
