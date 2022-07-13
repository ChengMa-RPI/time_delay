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
from scipy.sparse.linalg import eigs as sparse_eig
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, mutual_multi, network_generate, stable_state, ode_Cheng, ddeint_Cheng, generate_SF



mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#b3cde3', '#ccebc5', '#fbb4ae']) 

fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 10


def error_eigen_blocks(dynamics, network_type, N, d_list, beta_list, betaeffect, blocks_list):
    """TODO: Docstring for distribution_multi.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    for d in d_list:
        for beta in beta_list:
            error_list = []
            tau_eigen_list = []
            for group_num in blocks_list:
                des = '../data/' + dynamics + '/' + network_type 
                if betaeffect:
                    des_eigen = des + '/eigen_blocks/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}.csv'
                    des_evolution = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_Euler.csv'
                else:
                    des_eigen = des + '/eigen_blocks/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}.csv'
                    des_evolution = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_Euler.csv'
                    des_evolution = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_RK.csv'
                    #des_evolution = des + '/tau_multi/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
                data_eigen = np.array(pd.read_csv(des_eigen, header=None), float)
                data_evolution = np.array(pd.read_csv(des_evolution, header=None), float)
                tau_eigen = data_eigen[:, -1][np.argsort(data_eigen[:, 0])]
                tau_evolution = data_evolution[:, -1][np.argsort(data_evolution[:, 0])]
                error = np.mean(np.abs(tau_eigen - tau_evolution)/tau_evolution)
                tau_eigen_list.append(np.mean(tau_eigen))
                error_list.append(error)
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(blocks_list, error_list, '.', markersize=marksize, label=label)
            #plt.plot(blocks_list, tau_eigen_list, '.', alpha=alpha, label=label)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$n$', fontsize= fs)
    plt.ylabel('$error$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_d={d}_evolution_eigen'
    #plt.savefig('../report/report042921/' + figname + '.png' )
    #plt.close('all')
    return None


dynamics = 'PPI'
dynamics = 'BDP'
dynamics = 'mutual'
dynamics = 'CW'
N = 1000
betaeffect = 0

network_type = 'ER'
d_list = [2000]
beta_list = [0.01, 0.1, 1]

network_type = 'SF'
d_list = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_list = [[2.5, 999, 3]]
beta_list = [0.01, 0.1, 1]
beta_list = [1]


blocks_list = np.arange(1, 38, 1)
error_eigen_blocks(dynamics, network_type, N, d_list, beta_list, betaeffect, blocks_list)
