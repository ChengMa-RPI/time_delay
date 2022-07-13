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

def generate_random_graph(N, seed, degree_seq, kmax=0):
    """generate scale-free network using configuration model with given gamma, kmin, kmax. 

    :N: TODO
    :seed: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :returns: TODO

    """
    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        degree_seq[-1] = np.random.RandomState(seed=seed[1]+N+i).choice(k, size=1, p=pk)

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
        if kmax == 1 or kmax == N-2:
            break
    A = nx.to_numpy_array(G)
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))

    return A, A_interaction, index_i, index_j, cum_index


def xs_multi(dynamics, A_unit, weight_list):
    """TODO: Docstring for xs_multi.

    :A: TODO
    :returns: TODO

    """
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi_list = []
    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        xs_multi_list.append(xs_multi)
    return np.vstack((xs_multi_list))

N = 1000

kmean = 5
hetero = 5
ksqmean = (hetero + kmean) * kmean
degrees = np.ones(N) * kmean
n_hubs = 20

while (ksqmean - np.mean(degrees ** 2) )/ksqmean > 1e-2:
    degrees[:n_hubs] += 1
    index = np.random.choice(np.where(degrees[n_hubs:] > 1)[0], n_hubs, replace=False)
    degrees[index+n_hubs] -= 1


degree_seq = np.array(degrees, int)
seed = [0, 2]
A, A_interaction, index_i, index_j, cum_index = generate_random_graph(N, seed, degree_seq)
k1 = np.sum(A, 0)
kcumean = np.mean(k1**3)


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)

weight_list = np.round(np.arange(0.60, 0.65, 0.01), 5)
attractor_value = 0.1
#xs_multi_list = xs_multi(dynamics, A, weight_list)
#y_reduction = betaspace(A, xs_multi_list)

# n_hubs = 1: wc = 0.28
# n_hubs = 2: wc = 0.24
# n_hubs = 3: wc = 0.22
# n_hubs = 4: wc = 0.22
# n_hubs = 5: wc = 0.22
# n_hubs = 6: wc = 0.20
# n_hubs = 7: wc = 0.21
# n_hubs = 8: wc = 0.21
# n_hubs = 9: wc = 0.22
# n_hubs = 10: wc = 0.24
# n_hubs = 11: wc = 0.24
# n_hubs = 12: wc = 0.25
# n_hubs = 13: wc = 0.26
# n_hubs = 14: wc = 0.25
# n_hubs = 15: wc = 0.27
# n_hubs = 16: wc = 0.27
# n_hubs = 17: wc = 0.28
# n_hubs = 18: wc = 0.28
# n_hubs = 19: wc = 0.28
# n_hubs = 20: wc = 0.28
# n_hubs = 21: wc = 0.29
# n_hubs = 22: wc = 0.30
# n_hubs = 25: wc = 0.31
# n_hubs = 30: wc = 0.33
# n_hubs = 40: wc = 0.36
# n_hubs = 50: wc = 0.38
# n_hubs = 60: wc = 0.39
# n_hubs = 70: wc = 0.42
# n_hubs = 80: wc = 0.43
# n_hubs = 90: wc = 0.44
# n_hubs = 100: wc = 0.45
# n_hubs = 150: wc = 0.50
# n_hubs = 200: wc = 0.54
# n_hubs = 250: wc = 0.56
# n_hubs = 300: wc = 0.58
# n_hubs = 330: wc = 0.63
# extreme case: 
