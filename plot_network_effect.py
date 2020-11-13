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
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, mutual_multi, network_generate, stable_state, ode_Cheng, ddeint_Cheng


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 


def x_feature(network_type, beta, d, N, seed_list):
    """TODO: Docstring for x_feature.

    :network_type: TODO
    :d_list: TODO
    :N_list: TODO
    :seed_list: TODO
    :feature: TODO
    :returns: TODO

    """
    for seed, i in zip(seed_list, range(len(seed_list))):
        print(i)
        A, _, _, _, _ = network_generate(network_type, N, beta, seed, d)
        L = np.copy(A)
        N_gcc = np.size(A, 0)
        degree = np.sum(np.heaviside(A, 0), 0)
        degree_sort = np.hstack((np.sort(degree)[::-1], np.zeros(N-N_gcc)))
        
        k_ave = np.mean(degree)
        degree_max = np.max(degree)
        degree_min = np.min(degree)
        heterogeneity_gao = np.mean((degree - k_ave)**2)/k_ave
        heterogeneity_barabasi = np.sum(np.abs(degree - degree.reshape(degree.size, 1)))/N_gcc**2/k_ave
        lambda_adj = np.max(np.real(LA.eig(A)[0]))
        np.fill_diagonal(L, degree)
        lambda_lap = np.max(np.real(LA.eig(L)[0]))
        y, x = np.histogram(degree, np.arange(degree_min, degree_max, 1))
        index = np.where(y>10)[0][-1]
        gamma = - np.polyfit(np.log(x[:index]), np.log(y[:index]), 1)[0] 

        data = np.hstack((seed, N_gcc, degree_max, degree_min, k_ave, heterogeneity_gao, heterogeneity_barabasi, lambda_adj, lambda_lap, gamma, degree_sort))
        des_file = '../data/' + network_type + f'_N={N}_d=' + str(d) + '_network.csv'
        if not os.path.exists(des_file):
            column_name = [f'seed{j}' for j in range(np.size(seed))]
            column_name.extend(['N_actual', 'degree_max', 'degree_min', 'degree_ave', 'heterogeneity_gao', 'heterogeneity_barabasi', 'lambda_adj', 'lambda_lap', 'gamma'])
            column_name.extend([i+j for i, j in zip(['k'] * N, np.arange(N).astype(str))] )

            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')

    return None

def plot_network_effect(network_set, d_set, N_set, beta_set):
    """TODO: Docstring for plot_network_effect.

    :network_set: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """
    for network_type, i in zip(network_set, range(len(network_set))):
        d = d_set[i]
        N = N_set[i]
        data = np.array(pd.read_csv('../report/report091620/'+ network_type + f'_N={N}' + '_logistic.csv', header=None).iloc[:, :])
        #plt.plot(beta_set, data[:, 3:].transpose(), '-o', linewidth=lw, alpha = alpha, label=network_type)
        plt.plot(beta_set, data[:, 3:].transpose(), '-o', linewidth=lw, alpha = alpha)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend(np.round(data[:, 2], 2), fontsize=legendsize)
    plt.show()

def plot_hetero_effect(network_type, d_list, N_list, beta, feature):
    """TODO: Docstring for plot_network_effect.

    :network_set: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """

    if feature == 'heterogeneity_gao':
        xlabels = '$h_1$'
    elif feature == 'heterogeneity_barabasi':
        xlabels = '$h_2$'
    elif feature == 'degree_max':
        xlabels = '$k_{max}$'
    elif feature == 'degree_ave':
        xlabels = '$\\langle k \\rangle$'
    elif feature == 'lambda_adj':
        xlabels = '$\\lambda_{(A)}$'
    elif feature == 'lambda_lap':
        xlabels = '$\\lambda_{(L)}$'
    elif feature == 'gamma':
        xlabels = '$\\gamma$'


    if np.size(d_list) == 1:
        d = d_list[0]
        loop_list = N_list
    elif np.size(N_list) == 1:
        N = N_list[0]
        loop_list = d_list
    for x in loop_list:
        if np.size(d_list) == 1:
            N = x
            labels = f'N={N}'
        elif np.size(N_list) == 1:
            d = x
            if network_type == 'SF':
                labels = f'$\\gamma={d[0]}$' + '_$k_{min} =$' + str(d[-1]) 
                '''
                if d[1] == N-1:
                    labels = 'no cut-offs'
                if d[1] == 0:
                    labels = '$k_{max}$ cut-offs'
                '''
            elif network_type == 'ER':
                labels = f'$\\langle k \\rangle$ ={round(d*2/N,2)}'

        
        tau_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_logistic.csv').iloc[:, :])
        #tau_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_evolution.csv').iloc[:, :])
        feature_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_network.csv')[feature])
        seed_list = np.array(tau_data[:, 0], dtype=int).tolist()
        x_feature = feature_data[seed_list]

        if network_type == 'SF':
            tau_all = tau_data[:, 2:].transpose()
        else:
            tau_all = tau_data[:, 1:].transpose()

        beta_set = np.arange(1, np.size(tau_all, 0)+1, 1)
        tau = tau_all[np.where(beta==beta_set)[0][0]]

        plt.plot(x_feature, tau, '.', linewidth=lw, alpha = 0.5*alpha, label=labels, color='tab:red')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', nbins=4)
    plt.xlabel(xlabels, fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False, loc='lower left')
    plt.legend(markerscale=2, handlelength=0.1, fontsize=legendsize, frameon=False)
    #plt.ylim([0.20, 0.31])
    #plt.xlim([-5,70])
    #plt.xlim([0.2, 1.1])
    #plt.xlim([-0.1, 2])
    #plt.xlim([2, 14])
    #plt.savefig('../report/report110420/' + network_type + f'_d={d}_' + feature + '.png')
    #plt.close('all')
    #plt.show()

def network_stat(network_type, d, N, beta):
    """TODO: Docstring for network_stat.

    :arg1: TODO
    :returns: TODO

    """
    feature_list = ['N_actual', 'degree_max', 'degree_min', 'degree_ave', 'gamma']
    data = pd.read_csv('../data/'+ network_type + f'_N={N}_d={d}' + '_network.csv')
    degree_min = np.array(data['degree_min'])
    index = np.where(degree_min>=d[-1])[0]
    feature_mean, feature_std = [], [] 
    for feature in feature_list:
        feature_data = np.array(data[feature])[index]
        feature_mean.append(np.mean(feature_data))
        feature_std.append(np.std(feature_data))
    return feature_mean, feature_std

def plot_SF(d_list, N, beta, feature, color):
    """TODO: Docstring for plot_network_effect.

    :network_set: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """

    network_type = 'SF'
    if feature == 'heterogeneity_gao':
        xlabels = '$h_1$'
    elif feature == 'heterogeneity_barabasi':
        xlabels = '$h_2$'
    elif feature == 'degree_max':
        xlabels = '$k_{max}$'
    elif feature == 'degree_ave':
        xlabels = '$\\langle k \\rangle$'
    elif feature == 'lambda_adj':
        xlabels = '$\\lambda_{(A)}$'
    elif feature == 'lambda_lap':
        xlabels = '$\\lambda_{(L)}$'
    elif feature == 'gamma':
        xlabels = '$\\gamma$'


    for d, i in zip(d_list, range(len(d_list))):
        labels = f'$kmin={d[-1]}$'

        
        tau_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_logistic.csv').iloc[:, :])
        feature_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_network.csv')[feature])
        seed_list = np.array(tau_data[:, 0], dtype=int).tolist()
        x_feature = feature_data[seed_list]

        tau_all = tau_data[:, 2:].transpose()

        beta_set = np.arange(1, np.size(tau_all, 0)+1, 1)
        tau = tau_all[np.where(beta==beta_set)[0][0]]
        if i<len(d_list)-1:
            plt.plot(x_feature, tau, '.', linewidth=lw, alpha = 0.5*alpha, color=color)
    plt.plot(x_feature, tau, '.', linewidth=lw, alpha = 0.5*alpha, label=labels, color=color)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel(xlabels, fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False, loc='lower left')
    plt.legend(markerscale=2, handlelength=0.1, fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report101920/' + network_type + f'_d={d}_' + feature)
    #plt.close('all')
    #plt.show()
        
def degree_dis(network_type, N, d, seed_list):
    """TODO: Docstring for degree_dis.

    :network_type: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """
    beta = 1
    for seed in seed_list:
        A, _, _, _, _ = network_generate(network_type, N, beta, seed, d)
        N_gcc = np.size(A, 0)
        degree = np.sum(np.heaviside(A, 0), 0)
        degree_max = np.max(degree)

        degree_sort = np.hstack((np.sort(degree)[::-1], np.zeros(N-N_gcc)))
        y, x = np.histogram(degree, np.arange(d[-1], degree_max, 1))
        index = np.where(y>10)[0][-1]
        gamma = - np.polyfit(np.log(x[:index]), np.log(y[:index]), 1)[0] 
        print(gamma)
        data = np.hstack((seed, gamma, degree_sort))

        des_file = '../data/' + network_type + f'_N={N}_d=' + str(d) + '_degree.csv'
        if not os.path.exists(des_file):
            column_name = [f'seed{j}' for j in range(np.size(seed))]
            column_name.extend(['gamma'])
            column_name.extend([i+j for i, j in zip(['k'] * N, np.arange(N).astype(str))] )

            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')

    return None

def plot_degree_dis(network_type, N, d, seed):
    """TODO: Docstring for plot_degree_dis.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """
    for d in d_list:
        degree = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_degree.csv'))[seed, 3:]
        degree_max = np.max(degree)
        y, x = np.histogram(degree, np.arange(d[-1], degree_max, 1))
        y = y/np.size(seed)
        index = np.where(y>10)[0][-1]
        plt.loglog(x[:-1], y, 'o', label='$k_{min}=$' + str(d[-1]))
        
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$k$', fontsize= fs)
    plt.ylabel('$N(k)$', fontsize =fs)
    plt.xlim(-1, 5*1e2)
    plt.ylim(0.9, 5e4)
    plt.legend(fontsize=legendsize, frameon=False)
    plt.savefig('../report/report111220/' + network_type + f'_N={N}_gamma={d[0]}_' + str(len(seed)) + '.png')
    plt.close('all')






fs = 22
ticksize = 20
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']
imag = 1j
d = 3

N_list = [1000]
d_set = [900]
d_set = [5, 10, 15, 20, 25, 50]
d_set = [2, 2.5, 3, 3.5, 4, 4.5, 5]
network_type = 'real'
network_type = 'RR'
network_type = '2D'
network_type = 'BA'
network_type ='star'
network_type ='star'
network_type = 'ER'
network_type = 'SF'
beta_set = np.arange(1, 2, 1)
beta = 1
feature = 'N_actual'
feature = 'degree_ave'
feature = 'heterogeneity_barabasi'
feature = 'lambda_adj'
feature = 'degree_max'
feature = 'heterogeneity_gao'

#plot_network_effect(network_set, d_set, N_set, beta_set)
seed_list = np.arange(500).tolist()
N_list = [5000]
d_list = [4]




kmin = [2, 4, 5]
kmin = [2, 3, 4, 5]
gamma = [2.5, 3, 3.5, 4]
gamma = [2.5]

ktry = 50
kmax = [50, 100, 200, 300, 500, 800]
d_list = np.hstack((np.meshgrid(kmin, gamma)[0], np.meshgrid(kmin, gamma)[1])).tolist()
d_list = [[3, 18, 20, 2], [3, 20, 20, 2], [3, 30, 20, 2], [3, 50, 20, 2], [3, 100, 20, 2], [3, 200, 20, 2], [3, 300, 20, 2], [3, 500, 20, 2], [3, 800, 20, 2]]

d_list = [2000, 3000, 4000]
d_list = [[[3.5, j, i, k] for j in kmax ]for i, k in zip([50, 50, 50], [5])]
#d_list.extend([[3, 0, 2], [3, 0, 5], [3.5, 0, 5]])
#d_list.extend([[3, 999, 5], [3.5, 999, 5]])
seed2 = [0]
seed1 = np.arange(20).tolist()
seed_list = np.vstack((np.ravel(np.meshgrid(seed1, seed2)[0]), np.ravel(np.meshgrid(seed1, seed2)[1]))).transpose().tolist()
seed_list = np.vstack((seed1, seed1)).transpose().tolist()



'''
for N in N_list:
    for d in d_list:
        x_feature(network_type, beta, d, N, seed_list)
'''
feature_list = ['heterogeneity_gao']
feature_list = ['heterogeneity_barabasi']
feature_list = ['lambda_adj']
feature_list = ['degree_ave']
feature_list = ['degree_max']

'''
for feature in feature_list: 
    for d, i in zip(d_list, range(len(d_list))):
        #plot_SF(d, N, beta, feature, colors[i])
        plot_hetero_effect(network_type, [d], N_list, beta, feature)
'''
#plot_hetero_effect(network_type, d_list, N_list, beta, feature)
N = 10000
d_list = [[i, j, k] for i in gamma for j in [N-1] for k in kmin]
'''
for d in d_list:
    degree_dis(network_type, N, d, seed_list)
'''
plot_degree_dis(network_type, N, d_list, [0])
