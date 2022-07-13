import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import network_generate

import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import itertools
import time 
from scipy.integrate import odeint
from mutual_framework import betaspace, network_generate

B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1



fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

#mpl.rcParams['axes.prop_cycle'] = cycler(color=['#fc8d62', '#66c2a5', '#8da0cb', '#ffd92f', '#a6d854',  '#e78ac3', '#e5c494', '#b3b3b3']) 

def mutual_multi(x, t, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt



seed1 = [0, 0]
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()
seed_ER = seed1


N = 1000
beta = 0.15
betaeffect = 0

network_type = 'SF'
d = [2.5, 999, 3]
seed_list = seed_SF
seed = [0, 0]


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1



iteration_step = 10
group_num = 19
space = 'log'

A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)

des = '../data/' + dynamics + '/' + network_type + f'/xs_beta_two_cluster_iteraction_{iteration_step}_adaptive_' + space + '/'
delta_x_list = np.zeros((len(seed_list), iteration_step))

if betaeffect:
    des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'
else:
    des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + f'_group_num={group_num}_seed={seed}.csv'

data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
xs_multi = data[:, 0]
w = data[:, 1]
xs_beta = data[:, 2]
for k in range(iteration_step):
    node_group_number = data[:, 2+k * 5+ 1]
    node_index = np.array(data[:, 2+ k * 5+ 2], dtype=int)
    xs_nn = data[:, 2+ k*5+ 3]
    xs_groups = data[:, 2+ k*5+ 4]
    xs_group_decouple_iteration = data[:, 2 + k*5 + 5]
    N_actual = len(xs_multi)

    group_index = []
    for i in np.unique(node_group_number):
        group_index.append(node_index[np.where(node_group_number == i)[0]])
    length_groups = len(group_index)

    A_reduction = np.zeros((length_groups, length_groups))
    for  i in range(length_groups):
        for j in range(length_groups):
            k_i = w[group_index[i]] 
            A_reduction[i, j] = np.sum(k_i * np.sum(A[group_index[i]][:, group_index[j]], 1)) / k_i.sum()
            #A_reduction[i, j] = np.sum( np.sum(A[group_index[i]][:, group_index[j]], 1)) / len(k_i) 
    A_index = np.where(A_reduction>0)
    A_interaction = A_reduction[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree_reduction = np.sum(A_reduction>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree_reduction)))

initial_condition = np.ones(length_groups) * attractor_value
t = np.arange(0, 1000, 0.01)
net_arguments = (index_i, index_j, A_interaction, cum_index)
xs_group = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

