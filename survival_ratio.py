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
from mpl_toolkits.mplot3d import Axes3D

fs = 24
ticksize = 18
labelsize = 30
anno_size = 14
subtitlesize = 15
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8



def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


class StateDistribution():
    def __init__(self, network_type, N, d, seed, dynamics):
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.dynamics = dynamics
        self.des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/'

    def data_load(self, m, space):
        """TODO: Docstring for data_multi.

        :arg1: TODO
        :returns: TODO

        """
        N, d, seed = self.N, self.d, self.seed
        if m == N :
            xs_des = self.des + 'xs_multi/'
            xs_file = xs_des + f'N={N}_d={d}_seed={seed}.csv'
            self.data = np.array(pd.read_csv(xs_file, header=None))
            self.group_node_number = np.array( np.ones(self.data.shape[-1]-1), int)
            self.group_index = [[i] for i in range(N)]
        else:
            xs_des = self.des + f'degree_kmeans_space={space}/'
            xs_file = xs_des + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
            A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
            """
            file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
            A_unit = scipy.sparse.load_npz(file_A).toarray()
            """        
            G = nx.from_numpy_array(A_unit)
            feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
            group_index = group_index_from_feature_Kmeans(feature, m)
            self.group_index = group_index
            self.group_node_number = np.array([len(i) for i in group_index])
            self.data = np.array(pd.read_csv(xs_file, header=None))



    def plot_R_w(self, m, space):
        """TODO: Docstring for plot_P_w.

        :weight_list: TODO
        :returns: TODO

        """    
        N = self.N
        self.data_load(m, space)
        data = self.data 
        weight = data[:, 0]
        xs = data[:, 1:]
        R_list = []
        for xs_i in xs:
            R = sum(xs_i > survival_threshold)
            R_list.append(R)
        plt.plot(weight, np.array(R_list) / N, '.') 

    def plot_P_x_w(self, m, space):
        """TODO: Docstring for plot_P_w.

        :weight_list: TODO
        :returns: TODO

        """    
        N = self.N
        self.data_load(m, space)
        data = np.abs(self.data)
        data = data[1:]
        df = pd.DataFrame( data[::5, 1:].transpose()) 
        df.columns = data[::5, 0]
        df = pd.melt(df, var_name='weight', value_name='xs')
        g = sns.histplot(data=df, x='xs', hue='weight', log_scale=True, element='poly', fill=False) 
        #g = sns.kdeplot(data=df, x='xs', hue='weight', log_scale=True) 
        g.set(xscale='log', yscale='log')

    def plot_P_x_m(self, m_list, space, weight):
        """TODO: Docstring for plot_P_w.

        :weight_list: TODO
        :returns: TODO

        """    
        N = self.N
        df = pd.DataFrame() 
        for m in m_list:
            df_i = pd.DataFrame() 
            self.data_load(m, space)
            data = np.abs(self.data )
            weight_list = data[:, :1]
            index = np.where(abs(weight_list - weight) < 1e-2 ) [0][0]
            xs = data[index, 1:]
            df_i['xs'] = xs
            df_i['m'] = m
            df = pd.concat((df, df_i))
        df = df.reset_index()
        self.df = df

        g = sns.histplot(data=df, x='xs', hue='m', log_scale=True, element='poly', fill=False) 
        #g = sns.kdeplot(data=df, x='xs', hue='m', log_scale=True) 
        g.set(xscale='log', yscale='log')
        g.set_ylim(1e-3)

    def plot_x_w(self, m_list, space, weight_list):
        """TODO: Docstring for plot_P_w.

        :weight_list: TODO
        :returns: TODO

        """    
        N = self.N
        df = pd.DataFrame() 
        markers = ['o', '^', 's', 'p', 'P', 'h']
        colors=sns.color_palette()
        alphas = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
        sizes = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
        for i, m in enumerate(m_list):
            print(m)
            df_i = pd.DataFrame() 
            self.data_load(m, space)
            data = np.abs(self.data )
            weights = data[:, :1]
            index = [np.where(np.abs(weights - weight) < 1e-2 )[0][0] for weight in weight_list]
            xs = data[index, 1:]
            df_i['xs'] = np.ravel(xs)
            df_i['m'] = m
            df_i['weight'] = sum([[weight_i] * m  for weight_i in weight_list], [])

            df_i['group_node_number'] = np.ravel(np.tile( (self.group_node_number / np.sum(self.group_node_number) + 0.01) * 100, (1, len(weight_list)) ))
            self.df_i = df_i
            plt.scatter(x=df_i['weight'], y=df_i['xs'], s=df_i['group_node_number'], facecolors='none', marker=markers[i], edgecolors=colors[i], alpha=alphas[i]) 
            df = pd.concat((df, df_i))


def plot_x_m(network_type, N, d, seed_list, dynamics, m_list, space, weight_list):
    """TODO: Docstring for plot_P_w.

    :weight_list: TODO
    :returns: TODO

    """    
    fig, axes = plt.subplots(len(seed_list), len(weight_list) + 1, sharex=False, sharey=True, figsize=(3*(len(weight_list) +1), 3*len(seed_list)) ) 
    markers = ['o', '^', 's', 'p', 'P', 'h']
    linestyles = [(i, j) for i in [3, 6, 9] for j in [1, 5, 9, 13]]
    colors=sns.color_palette('hls', 11)
    alphas = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    sizes = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    for i, seed in enumerate(seed_list):
        s = StateDistribution(network_type, N, d, seed, dynamics)
        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)

        data_m = dict()
        data_all_m = dict()
        groups_node_nums = dict()
        for m in m_list:
            s.data_load(m, space)
            data = np.abs(s.data )
            data_all_m[m] = data
            weights = data[:, :1]
            index = [np.where(np.abs(weights - weight) < 1e-02 )[0][0]  for weight in weight_list]
            xs = data[index, 1:]
            data_m[m] = xs
            groups_node_nums[m] = s.group_node_number
        for j, weight in enumerate(weight_list):
            ax = axes[i][j]
            simpleaxis(ax)
            sizes = np.ravel(np.tile( (s.group_node_number / np.sum(s.group_node_number) + 0.01) * 100, (1, len(weight_list)) ))
            for k, m in enumerate(m_list):
                y = data_m[m][j]
                ax.scatter(x=np.ones(len(y)) * m, y=y, s= (groups_node_nums[m] / np.sum(groups_node_nums[m]) + 0.05) * 100, alpha=np.log(min(m_list)+0.5) / np.log(m+0.5), color=colors[k]) 
            ax.set(xscale='log', yscale='log')
               
            if i == 0:
                title_name = 'group  ' + f'$w={weight}$'
                ax.set_title(title_name, size=labelsize*0.5)

        ax = axes[i][j+1]
        simpleaxis(ax)
        for i_m, m in enumerate(m_list):
            data = data_all_m[m]
            weights = data[:, 0]
            weight_unique = np.arange(0.01, 0.6, 0.01)
            index_plot = [np.where(abs(weights - w_i) < 1e-5)[0][0] for w_i in weight_unique]
            y = data[index_plot, 1:]
            xs = np.repeat(y, groups_node_nums[m], axis=1)
            
            y_gl = betaspace(A_unit, xs)[-1]
            ax.plot(weight_unique, y_gl, linewidth=1, color=colors[i_m], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
        if i == 0:
            title_name = f'global'
            ax.set_title(title_name, size=labelsize*0.5)
            ax.legend(fontsize=legendsize*0.5, ncol=2, loc='lower right', frameon=False, ) 


    xlabel = '$ m $'
    ylabel = 'stable state'
    fig.text(x=0.03, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.7, rotation=90)
    fig.text(x=0.5, y=0.03, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.25, hspace=0.25, bottom=0.12, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + network_type + '_subplots_xs_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    return None

def plot_error_w(network_type, N, d, seed, dynamics, weight_list, m_list, space):
    """TODO: Docstring for plot_error_w.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :weight_list: TODO
    :m: TODO
    :returns: TODO

    """
    s = StateDistribution(network_type, N, d, seed, dynamics)
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    y_gl_list = []
    for i_m, m in enumerate(m_list):
        s.data_load(m, space)
        data = s.data
        group_node_number = s.group_node_number
        group_index = s.group_index
        
        weights = data[:, 0]
        weight_unique = np.arange(0.01, 0.6, 0.01)
        index_plot = [np.where(abs(weights - w_i) < 1e-5)[0][0] for w_i in weight_unique]
        y = data[index_plot, 1:]
        xs = np.zeros( (len(index_plot), N) )
        for i, group_i in enumerate(group_index):
            xs[:, group_i] = y[:, i:i+1]
        y_gl = betaspace(A_unit, xs)[-1]
        y_gl_list.append(y_gl)
    y_gl_list = np.vstack( (y_gl_list) ) 
    error_m = np.zeros(( len(m_list) -1))
    for i_m, m in enumerate(m_list[:-1]):
        error = np.abs(y_gl_list[-1] - y_gl_list[i_m] ) / (y_gl_list[-1] + y_gl_list[i_m])
        error_m[i_m] = error[-1]
        #plt.plot(weight_unique, error, label = f'm={m}')
    plt.plot(m_list[:-1], error_m ) 

    plt.yscale('log')
    plt.xscale('log')

    plt.legend()
    return error_m 


def plot_xs_onenet(network_type, N, d, seed, dynamics, m_list, space, weight_list):
    """TODO: Docstring for plot_P_w.

    :weight_list: TODO
    :returns: TODO

    """    
    fig, axes = plt.subplots(len(seed_list), len(weight_list) + 1, sharex=False, sharey=True, figsize=(3*(len(weight_list) +1), 3*len(seed_list)) ) 
    markers = ['o', '^', 's', 'p', 'P', 'h']
    linestyles = [(i, j) for i in [3, 6, 9] for j in [1, 5, 9, 13]]
    colors=sns.color_palette('hls', 11)
    alphas = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    sizes = [np.log(min(m_list)+1) / np.log(m+1) for m in m_list]
    s = StateDistribution(network_type, N, d, seed, dynamics)
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    data_m = dict()
    data_all_m = dict()
    groups_node_nums = dict()
    for m in m_list:
        s.data_load(m, space)
            data = np.abs(s.data )
            data_all_m[m] = data
            weights = data[:, :1]
            index = [np.where(np.abs(weights - weight) < 1e-02 )[0][0]  for weight in weight_list]
            xs = data[index, 1:]
            data_m[m] = xs
            groups_node_nums[m] = s.group_node_number
        for j, weight in enumerate(weight_list):
            ax = axes[i][j]
            simpleaxis(ax)
            sizes = np.ravel(np.tile( (s.group_node_number / np.sum(s.group_node_number) + 0.01) * 100, (1, len(weight_list)) ))
            for k, m in enumerate(m_list):
                y = data_m[m][j]
                ax.scatter(x=np.ones(len(y)) * m, y=y, s= (groups_node_nums[m] / np.sum(groups_node_nums[m]) + 0.05) * 100, alpha=np.log(min(m_list)+0.5) / np.log(m+0.5), color=colors[k]) 
            ax.set(xscale='log', yscale='log')
               
            if i == 0:
                title_name = 'group  ' + f'$w={weight}$'
                ax.set_title(title_name, size=labelsize*0.5)

        ax = axes[i][j+1]
        simpleaxis(ax)
        for i_m, m in enumerate(m_list):
            data = data_all_m[m]
            weights = data[:, 0]
            weight_unique = np.arange(0.01, 0.6, 0.01)
            index_plot = [np.where(abs(weights - w_i) < 1e-5)[0][0] for w_i in weight_unique]
            y = data[index_plot, 1:]
            xs = np.repeat(y, groups_node_nums[m], axis=1)
            
            y_gl = betaspace(A_unit, xs)[-1]
            ax.plot(weight_unique, y_gl, linewidth=1, color=colors[i_m], label=f'$m={m}$', linestyle=(0, linestyles[i_m]) ) 
        if i == 0:
            title_name = f'global'
            ax.set_title(title_name, size=labelsize*0.5)
            ax.legend(fontsize=legendsize*0.5, ncol=2, loc='lower right', frameon=False, ) 


    xlabel = '$ m $'
    ylabel = 'stable state'
    fig.text(x=0.03, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.7, rotation=90)
    fig.text(x=0.5, y=0.03, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    plt.subplots_adjust(left=0.12, right=0.95, wspace=0.25, hspace=0.25, bottom=0.12, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + network_type + '_subplots_xs_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    return None







dynamics = 'genereg'
dynamics = 'CW'
dynamics = 'mutual'
network_type = 'SF'
space = 'log'
N = 1000
d = [2.5, 999, 3]
seed = [0, 0]

m = N 
m = 40
survival_threshold = 1

state_dis = StateDistribution(network_type, N, d, seed, dynamics)

#state_dis.plot_P_x_w(m, space)
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )
m_list = np.unique(np.array(np.round([(2**1) ** i for i in range(10)], 0), int) ).tolist() + [N]

weight = 0.15
weight_list = [0.1, 0.2, 0.3, 0.4]
seed_list = [[i, i] for i in np.arange(0, 10)]
#plot_x_m(network_type, N, d, seed_list, dynamics, m_list, space, weight_list)
d = [2.5, 999, 3]
seed = [0, 0]
#m_list = [512, 1000]

error_list = []
for seed in seed_list:
    error_m = plot_error_w(network_type, N, d, seed, dynamics, weight_list, m_list, space)
    error_list.append(error_m)

plt.loglog(m_list[:-1], error_list) 

