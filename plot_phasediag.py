import sys
sys.path.insert(1, '/home/mac/RPI/research/')

import sympy as sp
import numpy as np 
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
import networkx as nx
import time
from ddeint import ddeint
from numpy import linalg as LA
import pandas as pd 
import scipy.io
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import itertools
from scipy import linalg as slin
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigs as sparse_eig
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, mutual_multi, network_generate, stable_state, ode_Cheng, ddeint_Cheng



mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:orange', 'tab:green', 'tab:red',  'tab:orange',  'tab:olive', 'grey', 'tab:brown', 'tab:cyan']) 



def phase_diagram(dynamics, tau, initial_condition):
    """TODO: Docstring for phase_diagram.

    :dynamics: TODO
    :beta_list: TODO
    :tau: TODO
    :returns: TODO

    """
    des_file = '../data/' + dynamics + f'/single/tau={tau}/x0={initial_condition}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    beta_list = data[:, 0]
    xs = data[:, 1]
    index = np.where(xs>1e-3)[0]
    beta_list = beta_list[index]
    xs = xs[index]
    plt.plot(beta_list, xs, '--', linewidth=lw, alpha=alpha, label=f'$\\tau=${tau}')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$\\beta$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)

    #plt.show()

def phase_diagram_3D(dynamics, tau_list, initial_condition):
    """TODO: Docstring for phase_diagram_3D.

    :arg1: TODO
    :returns: TODO

    """
    tau_plot = []
    beta_plot = []
    xs_plot = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for tau in tau_list:
        des_file = '../data/' + dynamics + f'/single/tau={tau}/x0={initial_condition}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        beta_list = data[:, 0]
        xs = data[:, 1]
        index = np.where(xs>1e-3)[0]
        beta_list = beta_list[index]
        xs = xs[index]
        tau_plot.extend([tau] * np.size(index))
        beta_plot.extend(beta_list.tolist())
        xs_plot.extend(xs.tolist())
        ax.scatter3D(beta_list, [tau] * len(index), xs)
    ax.set_xlabel('$\\beta$', fontsize=fs)
    ax.set_ylabel('$\\tau$', fontsize=fs)
    ax.set_zlabel('$x_s$', fontsize=fs)

    return tau_plot, beta_plot, xs_plot

def phase_multi_scatter(dynamics, network_type, N, d, beta_list, betaeffect, delay, initial_value):
    """TODO: Docstring for phase_multi.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :delay: TODO
    :initial_value: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs/' 
    if not os.path.exists(des):
        os.makedirs(des)
    for beta in beta_list:
        if betaeffect == 0:
            des_file = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
        else:
            des_file = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        xs = data[:, -1]
        plt.plot(beta * np.ones(len(xs)), xs, '.')
    plt.show()

def phase_multi_individual(dynamics, network_type, N, d, beta_list, betaeffect, delay, initial_value):
    """TODO: Docstring for phase_multi.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :delay: TODO
    :initial_value: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs/' 
    index_list = []
    if not os.path.exists(des):
        os.makedirs(des)
    xs_list = np.zeros((len(beta_list), 100))
    for beta, i in zip(beta_list, range(len(beta_list))):
        if betaeffect == 0:
            des_file = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
        else:
            des_file = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        seed = data[:, 0]
        xs = data[:, -1][np.argsort(seed)]
        xs_list[i] = xs
    for i in range(np.size(xs_list, 1)):
        if np.min(xs_list[:, i])< 0:
            if np.max(xs_list[:, i][np.where(xs_list[:, i] == -1)[0][0]:]) < 0:
                index_list.append(i)
        else:
            index_list.append(i)
    plt.plot(beta_list, xs_list[:, np.array(index_list)], 'o-', markerfacecolor="None", color='tab:blue', alpha=0.6)
    plt.xlabel('$\\beta$', fontsize =fs)
    plt.ylabel('$x_s$', fontsize =fs)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=6)
    #plt.legend(fontsize=legendsize, frameon=False)

    plt.show()

def phase_multi_bifurcation(dynamics, network_type, N, d_list, beta_list_d, betaeffect, delay, initial_value):
    """TODO: Docstring for phase_multi.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :delay: TODO
    :initial_value: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs/' 
    transition_beta = []
    for d, beta_list in zip(d_list, beta_list_d):
        xs_list = np.zeros((len(beta_list), 100))
        transition_index = []
        for beta, i in zip(beta_list, range(len(beta_list))):
            if betaeffect == 0:
                des_file = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
            else:
                des_file = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            seed = data[:, 0]
            xs = data[:, -1][np.argsort(seed)]
            xs_list[i] = xs
        for i in range(np.size(xs_list, 1)):
            if np.min(xs_list[:, i])< 0:
                index_instability = np.where(xs_list[:, i] == -1)[0][0]
                if np.max(xs_list[:, i][index_instability:]) < 0:
                    transition_index.append(index_instability)
            else:
                print('not enough')
        transition_beta.append([beta_list[np.min(transition_index)], beta_list[np.max(transition_index)]])
        
    if network_type == 'ER':
        x = np.array(d_list) * 2/N
        xlabel = '$\\langle k \\rangle $' 
    plt.plot(x, np.array(transition_beta), 'o-', markerfacecolor="None", color='tab:blue', alpha=alpha, linewidth = lw, label='$N=1000$')
    plt.xlabel(xlabel, fontsize =fs)
    plt.ylabel('$\\beta$', fontsize =fs)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=6)
    plt.legend(fontsize=legendsize, frameon=False)

    #plt.show()

def tau_distribution(dynamics, network_type, N, d, beta, label):
    """TODO: Docstring for tau_distribution.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :returns: TODO

    """
    color = next(colors)
    des_file = '../data/' + dynamics + '/' + network_type + f'/tau_multi/N={N}_d={d}_beta={beta}_logistic.csv'
    data = np.array(pd.read_csv(des_file, header=None))[:100]
    tau = data[:, -1]
    tau_ave = np.mean(tau)
    if label == 'd':
        if network_type == 'SF':
            n, bins, _ = plt.hist(tau, bins=np.arange(np.min(tau), np.max(tau) + 0.002, 0.002),  histtype='step', facecolor='None', edgecolor=color, linewidth=lw, alpha=0.6, label=f'$\\gamma=${d[0]}')
        elif network_type == 'ER':
            n, bins, _ = plt.hist(tau, bins=np.arange(np.min(tau), np.max(tau) + 0.002, 0.002),  histtype='step', facecolor='None', edgecolor=color, linewidth=lw, alpha=0.6, label=f'$\\langle k \\rangle$={int(2 * d/N)}')
    elif label == 'beta':
        n, bins, _ = plt.hist(tau, bins=np.arange(np.min(tau), np.max(tau) + 0.002, 0.002),  histtype='step', facecolor='None', edgecolor=color, linewidth=lw, alpha=0.6, label=f'$\\beta=${beta}')
    plt.plot(np.ones(2) * tau_ave, np.array([0, np.max(n)]), '--', color=color, linewidth=lw, alpha=0.6)
    plt.xlabel('$\\tau_c$', fontsize =fs)
    plt.ylabel('frequency', fontsize =fs)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=6)
    plt.legend( fontsize=legendsize, frameon=False)
    plt.xlim(0.17, 0.36)
    #bin_centers = 0.5*(bins[1:]+bins[:-1])
    #plt.plot(bin_centers, n, '--')
    #plt.show()
    
def tau_ave(dynamics, network_type, N, d_list, beta_list):
    """TODO: Docstring for tau_distribution.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :returns: TODO

    """
    for d in d_list:
        tau_ave = []
        for beta in beta_list:
            des_file = '../data/' + dynamics + '/' + network_type + f'/tau_multi/N={N}_d={d}_beta={beta}_logistic.csv'
            data = np.array(pd.read_csv(des_file, header=None))[:100]
            tau = data[:, -1]
            tau_ave.append(np.mean(tau))

        if network_type == 'ER':
            label = f'$\\langle k \\rangle=${int(2*d/N)}'
        elif network_type == 'SF':
            label = f'$\\gamma=${d[0]}'
        plt.plot(beta_list, np.array(tau_ave), 'o-', linewidth=lw, alpha=0.8, label=label)

    "1D"
    data = np.array(pd.read_csv('../data/' + dynamics + '/tau_1D.csv', header=None))
    beta = data[:, 0]
    index = np.where((beta-beta_list[0]>-0.1)&(beta-beta_list[-1]<0.1))[0]
    tau_1D = data[:, 1]
    plt.plot(beta[index], tau_1D[index], '--', color='tab:blue', linewidth=lw, alpha=0.8, label='single')
    plt.xlabel('$\\beta$', fontsize =fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=6)
    plt.legend( fontsize=legendsize, frameon=False)
    #bin_centers = 0.5*(bins[1:]+bins[:-1])
    #plt.plot(bin_centers, n, '--')
    #plt.show()
        
        



fs = 22
ticksize = 16
legendsize= 16

alpha = 0.8
lw = 3
marksize = 8

dynamics = 'harvest'
tau_list = [1.0, 1.5, 1.7, 1.9, 2.0]
initial_condition = 0.1

dynamics = 'mutual'
tau_list = [ 0.1, 0.2, 0.3]
initial_condition = 10.0

for tau in tau_list:
    #phase_diagram(dynamics, tau, initial_condition)
    pass

#result = phase_diagram_3D(dynamics, tau_list, initial_condition)

dynamics = 'mutual'
network_type = 'SF'
network_type = 'ER'
N = 100
beta_list_d = [np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(5.1, 5.5, 0.1), np.arange(6.6, 7, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(5.1, 5.5, 0.1), np.arange(6.1, 6.5, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(4.6, 4.9, 0.1), np.arange(5.2, 5.4, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 5.5, 0.5), np.arange(4.7, 5.0, 0.1))), 3))]
beta_list_d = [np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(5.4, 6.7, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(5.4, 6.6, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(5.2, 5.5, 0.1))), 3)), np.sort(np.round(np.hstack((np.arange(1, 5.5, 0.5), np.arange(4.8, 5.0, 0.1))), 3))]
beta_list = np.arange(1, 5.5, 0.5)
beta_list = np.sort(np.round(np.hstack((np.arange(1, 10, 0.5), np.arange(4.6, 4.9, 0.1), np.arange(5.2, 5.4, 0.1))), 3))
d = [4, 99, 3]
d = 800
d_list = [2000, 4000, 8000, 16000]
betaeffect = 1
delay = 0.2
initial_value = 5.0
#phase_multi_scatter(dynamics, network_type, N, d, beta_list, betaeffect, delay, initial_value)
phase_multi_individual(dynamics, network_type, N, d, beta_list, betaeffect, delay, initial_value)
#phase_multi_bifurcation(dynamics, network_type, N, d_list, beta_list_d, betaeffect, delay, initial_value)
beta = 7
d_list = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
network_type = 'ER'
d_list = [200, 400, 800, 1600]
network_type = 'SF'
N = 1000
d_list = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
colors=iter(['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink'])
for d in d_list:
    #tau_distribution(dynamics, network_type, N, d, beta, 'd')
    pass

colors=iter(['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive', 'grey', 'tab:brown', 'tab:cyan', 'tab:pink'])
beta_list = np.arange(1, 10, 2)
beta_list = [1, 3, 5, 7]
d = 2000
N = 1000
network_type = 'SF'
d = [2.5, 999, 3]
network_type = 'ER'
d = 8000
for beta in beta_list:
    #tau_distribution(dynamics, network_type, N, d, beta, 'beta')
    pass
network_type = 'SF'
d_list = [[2.5, 99, 3], [3, 99, 4], [3.8, 99, 5]]
network_type = 'ER'
d_list = [2000, 4000, 8000]
#tau_ave(dynamics, network_type, N, d_list, beta_list)
plt.show()
