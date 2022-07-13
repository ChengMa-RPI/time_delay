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


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:red', 'tab:orange',  'tab:brown', 'tab:olive', 'grey', 'tab:cyan']) 


def x_feature(network_type, beta, betaeffect, d, N, seed_list):
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
        A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
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
                labels = f'$N={N}$'
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
        #plt.plot(x_feature/N, tau, '.', linewidth=lw, alpha = 0.5*alpha, label=labels, color='tab:red')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', nbins=4)
    plt.xlabel(xlabels, fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False, loc='lower left')
    plt.legend(markerscale=2, handlelength=0.1, fontsize=legendsize, frameon=False)
    plt.ylim([0.17, 0.29])
    #plt.xlim([-5,70])
    #plt.xlim([0.2, 1.1])
    #plt.xlim([-0.1, 2])
    #plt.xlim([2, 14])
    #plt.xlim([-1, 1900])
    #plt.xlim([-1, 120])
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
        
def degree_dis(network_type, N, d, original, seed_list):
    """TODO: Docstring for degree_dis.

    :network_type: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """
    # generate degree sequence following power law distribution
    gamma, kmax, kmin = d 
    p = lambda k: k ** (float(-gamma))
    k = np.arange(kmin, N, 1)
    pk = p(k) / np.sum(p(k))

    for seed in seed_list:
        if original:
            random_state = np.random.RandomState(seed[0])
            degree_seq = random_state.choice(k, size=N, p=pk)
            i = 0
            while np.sum(degree_seq)%2:
                i+=1
                degree_seq[-1] = np.random.RandomState(seed=seed[0]+N+i).choice(k, size=1, p=pk)


            degree = degree_seq
        else:
            beta = 1
            betaeffect = 0
            A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d=d)
            degree = np.sum(A>0, 0)
        degree_max = np.max(degree)

        #degree_sort = np.hstack((np.sort(degree)[::-1], np.zeros(N-N_gcc)))
        degree_sort = np.sort(degree)[::-1]
        y, x = np.histogram(degree, np.arange(d[-1], degree_max, 1))
        index = np.where(y>10)[0][-1]
        gamma = - np.polyfit(np.log(x[:index]), np.log(y[:index]), 1)[0] 
        print(gamma)
        data = np.hstack((seed, gamma, degree_sort))

        if original:
            des_file = '../data/' + network_type + f'_N={N}_d=' + str(d) + '_original_degree.csv'
        else:
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
        y = y/np.size(seed)/N
        plt.loglog(x[:-1], y, 'o', alpha=0.6, label='$k_{min}=$' + str(d[-1]))
        
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$k$', fontsize= fs)
    plt.ylabel('$p(k)$', fontsize =fs)
    plt.xlim(-1, 5*1e4)
    plt.ylim(7e-6, 1.1)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.savefig('../report/report111220/' + network_type + f'_N={N}_gamma={d[0]}_' + str(len(seed)) + '.png')
    #plt.close('all')
    plt.show()


def plot_max_degree(network_type, N, d, seed_list):
    """TODO: Docstring for plot_max_degree.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """
    degree = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_network.csv'))[:, 11:]
    k_max = np.max(degree, 1)
    y, x = np.histogram(k_max, np.arange(k_max.min(), k_max.max(), 20))
    y = y/np.size(seed_list)
    plt.loglog(x[:-1], y, 'o', alpha=0.6)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$k_{max}$', fontsize= fs)
    plt.ylabel('$p(k_{max})$', fontsize =fs)

    plt.show()
    return k_max


def plot_degree(network_type, N, d, original, seed_list):
    """TODO: Docstring for plot_max_degree.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :returns: TODO

    """
    if original:
        degree = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_original_degree.csv'))[seed_list, 3:]
    else:
        degree = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_degree.csv'))[seed_list, 3:]
    degree = np.ravel(degree)
    y, x = np.histogram(degree, np.arange(d[-1], degree.max()+1, 1))
    y = y/N/np.size(seed_list)
    y_cum = (y[::-1].cumsum())[::-1]
    y_scale = y_cum * N ** (d[0]-1) * d[-1] **(1-d[0]) + 1
    y_scale = y_cum 
    alpha_N = x[:-1]/N
    plt.loglog(alpha_N, y_scale, '.', alpha=0.6, label=f'N={N}')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\alpha$', fontsize= fs)
    plt.ylabel('y_scale', fontsize =fs)
    plt.ylabel('$P(\\alpha)$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)

    #plt.show()
    return degree

def compare_multi_single_delay(network_type, N, d, beta, betaeffect, seed_list):
    """TODO: Docstring for compare_multi_single_delay.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed_list: TODO
    :returns: TODO

    """
    if betaeffect:
        beta_wt = 'beta'
    else:
        beta_wt = 'wt'

    single_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_kmax_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    decouple_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    decouple_two_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_two_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    multi_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    degree = single_data[:, 2]
    tau_single = single_data[:, 3]
    tau_decouple = decouple_data[:, 3]
    tau_decouple_two = decouple_two_data[:, 3]
    tau_multi = multi_data[:, -1]
    plt.plot(degree, tau_single, '.', color='tab:blue', label='one-component')
    plt.plot(degree[:100], tau_decouple_two, '.', color='tab:green', label='decoupling')
    #plt.plot(degree/2, single_data[:, 4], '.', color='r')
    plt.plot(degree[np.array(multi_data[:100, 0], dtype=int).tolist()], tau_multi[:100], '.', color='tab:red', label='multi-delay')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$k_{max}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)

    plt.show()
    return degree, tau_single


def alignment(network_type, N, d, beta, betaeffect, seed_list):
    """TODO: Docstring for alignment.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :seed_list: TODO
    :returns: TODO

    """
    if betaeffect:
        beta_wt = 'beta'
    else:
        beta_wt = 'wt'

    #single_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_kmax_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    decouple_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    decouple_two_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_two_' + beta_wt + f'={beta}_logistic.csv').iloc[100:, :])
    decouple_two_single_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_two_single_' + beta_wt + f'={beta}_logistic.csv').iloc[100:, :])
    decouple_single_eff = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_decouple_eff_' + beta_wt + f'={beta}_logistic.csv').iloc[:100, :])
    multi_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_' + beta_wt + f'={beta}_logistic.csv').iloc[:, :])
    #tau_single = single_data[:, 3]
    tau_decouple = decouple_data[:, 3]
    tau_decouple_two = decouple_two_data[:, 3]
    tau_decouple_two_single = decouple_two_single_data[:, 3]
    tau_decouple_eff = decouple_single_eff[:, 3]
    tau_multi = multi_data[:, -1][np.argsort(multi_data[:np.size(seed_list), 0])]
    #plt.plot(tau_multi, tau_decouple, '.', color='tab:red', alpha=alpha*0.5)
    plt.plot(tau_multi, tau_decouple, '.', color='tab:red', alpha=alpha*0.5, label='single')
    plt.plot(tau_multi, tau_decouple_eff, '.', color='tab:orange', alpha=alpha*0.5, label='single_eff')
    #plt.plot(tau_multi, tau_decouple_two_single, '.', color='tab:blue', alpha=alpha*0.5)
    #plt.plot(tau_multi, tau_decouple_two_single, '.', color='tab:blue', alpha=alpha*0.5, label='two')
    #plt.plot(tau_multi, tau_single, '.', color='red')
    x = np.array([np.min(tau_decouple) -0.01, np.max(tau_decouple) + 0.01])
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$\\tau$ multi-node', fontsize= fs)
    plt.ylabel('$\\tau$ decoupling', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)
    plt.show()

def compare_multi_single_beta(beta, network_type, N, d_list):
    """TODO: Docstring for critical_beta_w.

    :arg1: TODO
    :returns: TODO

    """
    theory_data = np.array(pd.read_csv(f'../data/beta={beta}_logistic.csv', header=None).iloc[:, :])
    theory_wk = theory_data[:, 0]
    index = np.where((theory_wk>0.2) & (theory_wk<1.5))[0]
    theory_wk = theory_wk[index]
    theory_tau = theory_data[index, 1]
    for d in d_list:
        single_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + f'_decouple_eff_beta={beta}_logistic.csv').iloc[:, :])
        single_seed = single_data[:, 0]
        single_wk = single_data[:, 4]
        single_tau = single_data[:, 3]
        multi_data = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + f'_beta={beta}_logistic.csv').iloc[:, :])
        multi_seed = multi_data[:, 0]
        multi_wk = single_wk[[np.where(i == single_seed)[0][0] for i in multi_seed]]
        multi_tau = multi_data[:, 2]
        plt.plot(single_wk, single_tau, 'o', alpha=0.5)
        #plt.plot(single_wk, single_tau, 'o', alpha=0.5, label='single')
        plt.plot(multi_wk, multi_tau, '*', alpha=0.5)
        #plt.plot(multi_wk, multi_tau, '*', alpha=0.5, label='multi')
    plt.plot(theory_wk, theory_tau, color='k')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(axis='x', nbins=4)
    plt.xlabel('$w$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend(['single', 'multi'], fontsize=legendsize, frameon=False)

    #plt.show()

def critical_beta(beta_list):
    """TODO: Docstring for critical_beta_w.

    :arg1: TODO
    :returns: TODO

    """
    for beta in beta_list:

        theory_data = np.array(pd.read_csv(f'../data/beta={beta}_logistic.csv', header=None).iloc[:, :])
        theory_wk = theory_data[:, 0]
        theory_tau = theory_data[:, 1]
        plt.plot(theory_wk, theory_tau, label=f'$\\beta={beta}$',linewidth=lw, alpha=alpha)
        plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        #plt.locator_params(axis='x', nbins=4)
        plt.xlabel('$w$', fontsize= fs)
        plt.ylabel('$\\tau_c$', fontsize =fs)
        plt.legend( fontsize=legendsize, frameon=False)

    #plt.show()


def distribution_multi(dynamics, network_type, N, d, beta):
    """TODO: Docstring for distribution_multi.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/' 
    des_file = des + f'_N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[1:, -1], float)
    #bins = np.arange(1.7, 1.8, 0.002)
    bins = np.arange(0.9, 1.7, 0.02)
    plt.hist(data, bins=bins)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\tau$', fontsize= fs)
    plt.ylabel('$\mathcal{N}(\\tau)$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False)
    plt.show()
    return None

def compare_multi_decouple(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax):
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
        print(d)
        des = '../data/' + dynamics + '/' + network_type + '/tau_multi/' 
        if betaeffect:
            des_multi = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
            des_decouple = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_two_logistic.csv'
        else:
            des_multi = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
            des_decouple = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_two_logistic.csv'

        data_multi = np.array(pd.read_csv(des_multi, header=None), float)
        data_decouple = np.array(pd.read_csv(des_decouple, header=None), float)
        #data_multi = np.array(pd.read_csv(des_multi).loc[pd.read_csv(des_multi)['seed0'] != 'seed0'], float)
        #data_decouple = np.array(pd.read_csv(des_decouple).loc[pd.read_csv(des_decouple)['seed0'] != 'seed0'], float)
        tau_multi = data_multi[:, -1][np.argsort(data_multi[:, 0])][:100]
        tau_decouple = data_decouple[:, -2][np.argsort(data_decouple[:, 0])][:100]
        #plt.plot(tau_multi, tau_decouple, '.', color='tab:red', alpha=alpha)
        #xmin = np.min([tau_multi, tau_decouple])
        #xmax = np.max([tau_multi, tau_decouple])
        #margin = max((xmax - xmin)/10 , 0.02)
        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER':
            label = f'$\\langle k \\rangle = {int(2*d/N)}$'
        elif network_type == 'RR':
            label = f'$\\langle k \\rangle = {d}$'
        elif network_type =='2D':
            label='2D'
        #label = f'$\\beta=${beta}'
        plt.plot(tau_multi, tau_decouple, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{\\mathrm{multi}}$', fontsize= fs)
    plt.ylabel('$\\tau_{\\mathrm{decouple}}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}'
    #plt.savefig('../report/report021721/' + figname + '.png' )
    #plt.close('all')
    #plt.show()
    return None

def compare_multi_single(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax):
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
        print(d)
        des = '../data/' + dynamics + '/' + network_type + '/tau_multi/' 
        if betaeffect:
            des_multi = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
            des_single = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'
        else:
            des_multi = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
            des_single = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_eff_logistic.csv'

        data_multi = np.array(pd.read_csv(des_multi, header=None), float)
        #data_multi = np.array(pd.read_csv(des_multi).loc[pd.read_csv(des_multi)['seed0'] != 'seed0'], float)
        #data_single = np.array(pd.read_csv(des_single).loc[pd.read_csv(des_single)['seed0'] != 'seed0'], float)
        data_single = np.array(pd.read_csv(des_single, header=None), float)
        tau_multi = data_multi[:, -1][np.argsort(data_multi[:, 0])]
        tau_single = data_single[:, -3][np.argsort(data_single[:, 0])]
        #plt.plot(tau_multi, tau_decouple, '.', color='tab:red', alpha=alpha)
        #xmin = np.min([tau_multi, tau_decouple])
        #xmax = np.max([tau_multi, tau_decouple])
        #margin = max((xmax - xmin)/10 , 0.02)
        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER' or network_type == 'RR':
            label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
        elif network_type =='2D':
            label='2D'
        #label = f'$w={beta}$'
        plt.plot(tau_multi, tau_single, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{multi}$', fontsize= fs)
    plt.ylabel('$\\tau_{single}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def tau_beta_same_wt(dynamics, network_type, N, d_list, beta, xmin, xmax):
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
        print(d)
        des = '../data/' + dynamics + '/' + network_type  
        #des_multi = des + '/tau_multi/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
        des_multi = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_evolution.csv'
        des_beta = '../data/beta_wt/' + network_type + f'/d={d}_wt=1.csv'

        data_multi = np.array(pd.read_csv(des_multi, header=None), float)
        tau_multi = data_multi[:, -1][np.argsort(data_multi[:, 0])]
        data_beta = np.array(pd.read_csv(des_beta, header=None), float)
        beta_list = data_beta[:, -1][np.argsort(data_beta[:, 0])] * beta
        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER' or network_type == 'RR':
            label = f'$\\langle k \\rangle = {int(d * 2/N)}$'
        elif network_type =='2D':
            label='2D'
        label = f'$w={beta}$'
        plt.plot(beta_list, tau_multi, '.', alpha=alpha * 0.5, label=label)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\beta$', fontsize= fs)
    plt.ylabel('$\\tau_{c}$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def eigenvalue_tau(dynamics, network_type, N, d, beta, betaeffect):
    """TODO: Docstring for eigenvalue_tau.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    data = np.array(pd.read_csv('../data/'+ dynamics + '/' + network_type + f'/N={N}_d={d}_beta={beta}_eigenvalue_tau.csv', header=None).iloc[:, :])
    tau = data[:, 0]
    mu = data[:, 1]
    index = np.argsort(tau)[10:]
    tau = tau[index]
    mu = mu[index]
    plt.plot(tau, mu, linewidth = lw, alpha=alpha, label=f'$\\beta$={beta}')
    plt.plot(tau, np.zeros(len(tau)), '--', linewidth = lw, alpha=alpha, color='tab:grey')
    #plt.plot(tau, np.exp(mu), linewidth = lw, alpha=alpha, label=f'$\\beta$={beta}')
    #plt.plot(tau, np.ones(len(tau)), '--', linewidth = lw, alpha=alpha, color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau$', fontsize= fs)
    plt.ylabel('$Re(\\lambda)$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False)
    #plt.show()
    return None

def tau_diameter(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax):
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
        des = '../data/' + dynamics + '/' + network_type 
        if betaeffect:
            des_multi = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
        else:
            des_multi = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
        data_multi = np.array(pd.read_csv(des_multi, header=None), float)
        tau_multi = data_multi[:, -1][np.argsort(data_multi[:, 0])]
        des_diameter = des + '/diameter/'+ f'd={d}.csv'
        data_diameter = np.array(pd.read_csv(des_diameter, header=None)) 
        diameter_sort = data_diameter[:, -1][np.argsort(data_diameter[:, 0])]

        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER' or network_type == 'RR':
            label = f'$\\langle k \\rangle = {int(d * 2/ N)}$'
        elif network_type =='2D':
            label='2D'
        plt.plot(diameter_sort, tau_multi, '.', alpha=alpha * 0.5, label=label)
    #plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$D$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.ylim([xmin, xmax])
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    #plt.show()
    return None

def plot_tau_1D(beta_min, beta_max):
    """TODO: Docstring for plot_tau_1D.
    :returns: TODO

    """
    "1D"
    data = np.array(pd.read_csv('../data/' + dynamics + '/tau_1D.csv', header=None))
    beta_1D = data[:, 0]
    tau_1D = data[:, 1]
    index = np.where((beta_1D >beta_min) &(beta_1D<beta_max))[0]
    plt.plot(beta_1D[index], tau_1D[index], '--', color='tab:grey', linewidth=lw, alpha=0.8, label='single')
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    return None

def compare_multi_evolution(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax):
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
        print(d)
        des = '../data/' + dynamics + '/' + network_type 
        if betaeffect:
            des_multi = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
            des_evo = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_evolution.csv'
        else:
            des_multi = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'
            des_evo = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_evolution.csv'
        data_multi = np.array(pd.read_csv(des_multi, header=None), float)
        data_evo = np.array(pd.read_csv(des_evo, header=None), float)
        tau_multi = data_multi[:, -1][np.argsort(data_multi[:, 0])]
        tau_evo = data_evo[:, -1][np.argsort(data_evo[:, 0])]
        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER' or network_type == 'RR':
            label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
        elif network_type =='2D':
            label='2D'
        label = f'$w={beta}$'
        plt.plot(tau_multi, tau_evo, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{matrix}$', fontsize= fs)
    plt.ylabel('$\\tau_{evolution}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def compare_single_evolution(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax):
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
        print(d)
        des = '../data/' + dynamics + '/' + network_type 
        if betaeffect:
            des_single = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'
            des_evo = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_evolution.csv'
        else:
            des_single = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_eff_logistic.csv'
            des_evo = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_evolution.csv'
        data_single = np.array(pd.read_csv(des_single, header=None), float)
        data_evo = np.array(pd.read_csv(des_evo, header=None), float)
        tau_single = data_single[:, -3][np.argsort(data_single[:, 0])]
        tau_evo = data_evo[:, -1][np.argsort(data_evo[:, 0])]
        xmin = xmin
        xmax = xmax
        margin = 0
        x = np.array([ xmin-margin,  xmax+ margin])
        if network_type == 'SF':
            label = f'$\\gamma={d[0]}$'
        elif network_type == 'ER' or network_type == 'RR':
            label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
        elif network_type =='2D':
            label='2D'
        label = f'$w={beta}$'
        plt.plot(tau_evo, tau_single, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{evolution}$', fontsize= fs)
    plt.ylabel('$\\tau_{decouple}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def compare_single_RK(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax):
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
            print(d)
            des = '../data/' + dynamics + '/' + network_type 
            if betaeffect:
                des_single = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'
                des_evo = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_RK.csv'
            else:
                des_single = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_eff_logistic.csv'
                des_evo = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_RK.csv'
            data_single = np.array(pd.read_csv(des_single, header=None), float)
            data_evo = np.array(pd.read_csv(des_evo, header=None), float)
            tau_single = data_single[:, -3][np.argsort(data_single[:, 0])]
            tau_evo = data_evo[:, -1][np.argsort(data_evo[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_evo, tau_single, '.', alpha=alpha * 0.5, label=label)
        plt.plot(x, x, '--', color='tab:grey')
        plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.locator_params(axis='x', tight=True, nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.xlabel('$\\tau_{evolution}$', fontsize= fs)
        plt.ylabel('$\\tau_{decouple}$', fontsize =fs)
        plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
        plt.show()
        figname = dynamics + '_' + network_type  + f'_d={d}_compare_single_RK'
        #plt.savefig('../report/report041521/' + figname + '.png' )
        #plt.close('all')
    return None

def compare_evolution_RK(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax):
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
            print(d)
            des = '../data/' + dynamics + '/' + network_type + '/tau_evo/'
            if betaeffect:
                des_Euler = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_Euler.csv'
                des_RK = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_RK.csv'
            else:
                des_Euler = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_Euler.csv'
                des_RK = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_RK.csv'
            data_Euler = np.array(pd.read_csv(des_Euler, header=None), float)
            data_RK = np.array(pd.read_csv(des_RK, header=None), float)
            tau_Euler = data_Euler[:, -1][np.argsort(data_Euler[:, 0])]
            tau_RK = data_RK[:, -1][np.argsort(data_RK[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_Euler, tau_RK, '.', alpha=alpha * 0.5, label=label)
        plt.plot(x, x, '--', color='tab:grey')
        plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.locator_params(axis='x', tight=True, nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.xlabel('$\\tau_{Euler}$', fontsize= fs)
        plt.ylabel('$\\tau_{RK}$', fontsize =fs)
        #plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
        #plt.show()
        figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
        #plt.savefig('../report/report021121/' + figname + '.png' )
        #plt.close('all')
    return None

def compare_separate_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax):
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
        for beta  in beta_list:
            des = '../data/' + dynamics + '/' + network_type 
            if betaeffect:
                des_separate = des + '/tau_model/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_separate.csv'
                des_evo = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_Euler.csv'
            else:
                des_separate = des + '/tau_model/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_separate.csv'
                des_evo = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_Euler.csv'
            data_separate = np.array(pd.read_csv(des_separate, header=None), float)
            data_evo = np.array(pd.read_csv(des_evo, header=None), float)
            tau_separate = data_separate[:, -1][np.argsort(data_separate[:, 0])]
            tau_evo = data_evo[:, -1][np.argsort(data_evo[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_evo, tau_separate, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{multi}$', fontsize= fs)
    plt.ylabel('$\\tau_{2}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def compare_separate_single(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax):
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

            des = '../data/' + dynamics + '/' + network_type 
            if betaeffect:
                des_separate = des + '/tau_model/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_separate.csv'
                des_single = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'
            else:
                des_separate = des + '/tau_model/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_separate.csv'
                des_single = des + '/tau_multi/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_eff_logistic.csv'
            data_separate = np.array(pd.read_csv(des_separate, header=None), float)
            data_single = np.array(pd.read_csv(des_single, header=None), float)
            tau_separate = data_separate[:, -1][np.argsort(data_separate[:, 0])]
            tau_single = data_single[:, -3][np.argsort(data_single[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_single, tau_separate, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{1}$', fontsize= fs)
    plt.ylabel('$\\tau_{2}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_beta={beta}_single'
    #plt.savefig('../report/report021121/' + figname + '.png' )
    #plt.close('all')
    return None

def compare_eigen_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax):
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

            des = '../data/' + dynamics + '/' + network_type 
            if betaeffect:
                des_eigen = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_eigen_decouple.csv'
                des_evolution = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_Euler.csv'
            else:
                des_eigen = des + '/tau_multi/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_eigen_decouple.csv'
                des_evolution = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_Euler.csv'
            data_eigen = np.array(pd.read_csv(des_eigen, header=None), float)
            data_evolution = np.array(pd.read_csv(des_evolution, header=None), float)
            tau_eigen = data_eigen[:, -1][np.argsort(data_eigen[:, 0])]
            tau_evolution = data_evolution[:, -1][np.argsort(data_evolution[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_evolution, tau_eigen, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{evolution}$', fontsize= fs)
    plt.ylabel('$\\tau_{eigen}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_d={d}_evolution_eigen'
    #plt.savefig('../report/report042921/' + figname + '.png' )
    #plt.close('all')
    return None

def compare_eigen_blocks_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, group_num, xmin, xmax):
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

            des = '../data/' + dynamics + '/' + network_type 
            if betaeffect:
                des_eigen = des + '/eigen_blocks/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}.csv'
                des_evolution = des + '/tau_evo/' + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_Euler.csv'
            else:
                des_eigen = des + '/eigen_blocks/' + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}.csv'
                des_evolution = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_Euler.csv'
                des_evolution = des + '/tau_evo/'+ f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_RK.csv'
            data_eigen = np.array(pd.read_csv(des_eigen, header=None), float)
            data_evolution = np.array(pd.read_csv(des_evolution, header=None), float)
            tau_eigen = data_eigen[:, -1][np.argsort(data_eigen[:, 0])]
            tau_evolution = data_evolution[:, -1][np.argsort(data_evolution[:, 0])]
            xmin = xmin
            xmax = xmax
            margin = 0
            x = np.array([ xmin-margin,  xmax+ margin])
            if network_type == 'SF':
                label = f'$\\gamma={d[0]}$'
            elif network_type == 'ER' or network_type == 'RR':
                label = f'$\\langle k \\rangle = {int(d * 2 / N)}$'
            elif network_type =='2D':
                label='2D'
            label = f'$w={beta}$'
            plt.plot(tau_evolution, tau_eigen, '.', alpha=alpha * 0.5, label=label)
    plt.plot(x, x, '--', color='tab:grey')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$\\tau_{evolution}$', fontsize= fs)
    plt.ylabel('$\\tau_{eigen}$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    #plt.show()
    figname = dynamics + '_' + network_type  + f'_d={d}_evolution_eigen'
    #plt.savefig('../report/report042921/' + figname + '.png' )
    #plt.close('all')
    return None

fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']
imag = 1j
d = 3

dynamics = 'BDP'
dynamics = 'PPI'
dynamics = 'harvest'
N_list = [1000]
d_set = [900]
d_set = [5, 10, 15, 20, 25, 50]
d_set = [2, 2.5, 3, 3.5, 4, 4.5, 5]
network_type = 'real'
network_type = '2D'
network_type = 'BA'
network_type ='star'
network_type ='star'
network_type = 'SF'
network_type = 'RR'
network_type = 'ER'
beta_set = np.arange(1, 2, 1)
beta = 4
betaeffect = 1
feature = 'N_actual'
feature = 'degree_max'
feature = 'heterogeneity_gao'
feature = 'lambda_adj'
feature = 'degree_ave'
feature = 'heterogeneity_barabasi'

#plot_network_effect(network_set, d_set, N_set, beta_set)
seed_list = np.arange(500).tolist()
N_list = [5000]
d_list = [4]




kmin = [2, 4, 5]
kmin = [2, 3, 4, 5]
gamma = [2.5, 3, 3.5, 4]
gamma = [4]

ktry = 50
kmax = [50, 100, 200, 300, 500, 800]
d_list = np.hstack((np.meshgrid(kmin, gamma)[0], np.meshgrid(kmin, gamma)[1])).tolist()
d_list = [[3, 18, 20, 2], [3, 20, 20, 2], [3, 30, 20, 2], [3, 50, 20, 2], [3, 100, 20, 2], [3, 200, 20, 2], [3, 300, 20, 2], [3, 500, 20, 2], [3, 800, 20, 2]]

d_list = [2000, 3000, 4000]
d_list = [[[3.5, j, i, k] for j in kmax ]for i, k in zip([50, 50, 50], [5])]
#d_list.extend([[3, 0, 2], [3, 0, 5], [3.5, 0, 5]])
#d_list.extend([[3, 999, 5], [3.5, 999, 5]])
seed2 = [0]
seed1 = np.arange(0, 100, 1).tolist()
seed_list = np.vstack((np.ravel(np.meshgrid(seed1, seed2)[0]), np.ravel(np.meshgrid(seed1, seed2)[1]))).transpose().tolist()
seed_list = np.vstack((seed1, seed1)).transpose().tolist()


d_list = [[3, 4999, 3]]
'''
for N in N_list:
    for d in d_list:
        x_feature(network_type, beta, betaeffect, d, N, seed_list)
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
d_list = [[3, 9999, 3]]
original = 0

'''
for d in d_list:
    degree_dis(network_type, N, d, original, seed_list)
'''

#plot_degree_dis(network_type, N, d_list, [0])

betaeffect = 0
beta= 0.1
N = 1000
d_list = [[4, 999, 3]]
d_list = [[2.5, 999, 3], [3, 999, 3], [3.5, 999, 3], [4, 999, 3]]
#degree, tau_single = compare_multi_single_delay(network_type, N, d, beta, betaeffect, seed_list)

#alignment(network_type, N, d, beta, betaeffect, seed1)
beta = 0.1
#compare_multi_single_beta(beta, network_type, N, d_list)
beta_list = [0.1, 1, 2, 4, 6]
#critical_beta(beta_list)

dynamics = 'PPI'
N = 100
d = 200

d_RR = [4]
d_ER = [200]





dynamics = 'CW'
beta = 4
xmin =1.53
xmax = 1.56

beta = 1
xmin = 1.56
xmax = 1.57

dynamics = 'SIS'
beta = 2.2
xmin = 1.1
xmax = 1.9

beta = 2.9
xmin = 1
xmax = 7

dynamics = 'harvest'
beta = 1
xmin = 1.64
xmax = 1.65

beta = 1.8
xmin = 1.73
xmax = 1.74

beta = 1
xmin = 0.27
xmax = 0.36






dynamics = 'genereg'
beta = 3
xmin = 0.5
xmax = 2.5








dynamics = 'PPI'
beta = 1
xmin = 1.5
xmax = 1.8

beta = 4
xmin = 1.4
xmax = 2.2



dynamics = 'BDP'
beta = 1
xmin = 0.1
xmax = 1

beta = 4
xmin = 0.1
xmax= 16

dynamics = 'mutual'
beta = 1
xmin = 0.28
xmax = 0.36

beta = 1.7
xmin = 0.24
xmin = 0.0
xmax = 0.4




beta_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
beta_list = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
beta_list = [0.01, 0.1, 1]
beta = 1
betaeffect = 0

network_list = ['SF', 'ER', 'RR']
network_list = [ 'ER']
N = 1000
d_RR = [16]
d_SF =  [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_SF = [[3, 999, 4]]
d_ER = [2000, 4000, 8000]
d_ER = [8000]
for network_type in network_list:
    if network_type == 'SF':
        d_list = d_SF
    elif network_type == 'ER':
        d_list = d_ER
    elif network_type == 'RR':
        d_list = d_RR
    elif network_type == '2D':
        d_list = [4]

    #compare_multi_decouple(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax)
    #compare_multi_single(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax)
    #compare_multi_evolution(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax)
    #compare_single_evolution(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax)
    #compare_separate_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax)
    #compare_eigen_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax)
    compare_eigen_blocks_evolution(dynamics, network_type, N, d_list, beta_list, betaeffect, 10, xmin, xmax)
    #compare_separate_single(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax)
    #compare_evolution_RK(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax)
    #compare_single_RK(dynamics, network_type, N, d_list, beta_list, betaeffect, xmin, xmax)
    #tau_beta_same_wt(dynamics, network_type, N, d_list, beta, xmin, xmax)
    #tau_diameter(dynamics, network_type, N, d_list, beta, betaeffect, xmin, xmax)
    pass

#plot_tau_1D(0.05, 80)
plt.show()
dynamics = 'mutual'
N = 100
network_type = 'SF'
d = [3, 99, 3]
network_type = 'ER'
d = 200
beta = 4
beta_list = [0.1, 1, 4, 10]
betaeffect = 1
for beta in beta_list:
    #eigenvalue_tau(dynamics, network_type, N, d, beta, betaeffect)
    pass
