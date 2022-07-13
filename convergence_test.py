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
ticksize = 20
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))



def xs_group_partition_bifurcation(network_type, N, seed, d, weight, m, attractor_low, initial_high, space, tradeoff_para, method):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    dynamics_multi = globals()[dynamics + '_multi']

    A = A_unit * weight
    net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, m)
    A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(N_actual), group_index)
    initial_low = np.ones(len(A_reduction_deg_part)) * attractor_low
    initial_high = np.ones(len(A_reduction_deg_part)) * attractor_high
    xs_low = odeint(dynamics_multi, initial_low, t, args=(arguments, net_arguments_reduction_deg_part))
    xs_high = odeint(dynamics_multi, initial_high, t, args=(arguments, net_arguments_reduction_deg_part))
    return xs_low[::100], xs_high[::100]

def xs_multi_bifurcation(network_type, N, seed, d, weight, attractor_low, attractor_high):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    initial_low = np.ones(N_actual) * attractor_low
    initial_high = np.ones(N_actual) * attractor_high
    dynamics_multi = globals()[dynamics + '_multi']
    A = A_unit * weight
    net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
    xs_low = odeint(dynamics_multi, initial_low, t, args=(arguments, net_arguments))
    xs_high = odeint(dynamics_multi, initial_high, t, args=(arguments, net_arguments))
    return xs_low[::100], xs_high[::100]



 




dynamics = 'genereg'
arguments = (B_gene, )
attractor_value = 20

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1

dynamics = 'CW'
arguments = (a, b)



beta = 1
betaeffect = 0 


network_type = 'SF'
space = 'log'
N = 1000


method = 'degree'
tradeoff_para = 0.5

gamma = 2.4
kmin = 2
d = [gamma, N-1, kmin]
seed = [88, 88]

#xs_group_partition_bifurcation(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method)
attractor_low = 0
attractor_high = 1000
weight = 0.80
#xs_low, xs_high = xs_multi_bifurcation(network_type, N, seed, d, weight, attractor_low, attractor_high)
weight = 1
m = 5
xs_low, xs_high_reduction = xs_group_partition_bifurcation(network_type, N, seed, d, weight, m, attractor_low, attractor_high, space, tradeoff_para, method)

