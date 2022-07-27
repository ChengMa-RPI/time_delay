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
from scipy.optimize import fsolve
import scipy
import time

B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1


method ='degree'
tradeoff_para = 0.5

network_type = 'SBM_ER'

N = [100, 100]
d = [[0.3, 0.05], [0.05, 0.6]]
seed = 0

N = [33, 33, 34]
d = np.array([[0.9, 0.001, 0.001], [0.001, 0.5, 0.001], [0.001, 0.001, 0.05]]).tolist()
seed = 0 
space = 'linear'
m=3

network_type = 'SF'
N = 1000
d = [2.5, 999, 3]
seed = [1, 1]
space = 'log'
m = 4

A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
N_actual = len(A_unit)
A = A_unit * 1
G = nx.from_numpy_array(A_unit)
feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
group_index = group_index_from_feature_Kmeans(feature, m)


A_submat = []
for i in range(len(group_index)):
    Ai_submat = []
    for j in range(len(group_index)):
        Ai_submat.append (A[group_index[i]][:, group_index[j]] )


    A_submat.append(Ai_submat)

# AA_T and the corresponding eigenvalue, eigenvector
AA_i = []
val_i = []
vec_i = []
for i in range(len(group_index)):
    AA_ij = []
    val_ij = []
    vec_ij = []
    for j in range(len(group_index)):
        if i == j:
            AA_T = A_submat[i][j]
        else:
            AA_T = A_submat[i][j].dot( A_submat[j][i] )
        eig_val, eig_vec = np.linalg.eig(AA_T)
        eig_val = np.round(eig_val, 10)
        domi_index = np.where(eig_val==np.max(eig_val))[0][0]
        domi_val = eig_val[domi_index]
        domi_vec = eig_vec[:, domi_index]
        domi_vec = np.array([i if np.abs(i) > 1e-10 else 0 for i in domi_vec] )

        AA_ij.append(AA_T) 
        val_ij.append(np.round(np.real(domi_val), 10))
        vec_ij.append(np.real(domi_vec))

    AA_i.append(AA_ij)
    val_i.append(val_ij)
    vec_i.append(vec_ij)


# calculate a_i
a_list = []
c_list = []
for i in range(len(group_index)):
    sub_dim = AA_i[i][0].shape[0]
    stand_unit = np.eye(sub_dim)
    c_st = np.zeros((sub_dim + 1, sub_dim + 1))
    b = np.zeros((sub_dim + 1) )
    b[-1] = 1
    for s in range(sub_dim):
        for t in range(sub_dim):
            for j in range(len(group_index)):
                #c_st[s, t] += (AA_i[i][j].dot(stand_unit[s]) - val_i[i][j] * stand_unit[s] ).dot(AA_i[i][j].dot(stand_unit[t]) - val_i[i][j] * stand_unit[t])
                c_st[s, t] += (AA_i[i][j][s] - val_i[i][j]  * stand_unit[s] ).dot( AA_i[i][j][t] - val_i[i][j]  * stand_unit[t] )
    c_st[:-1, -1] = -1
    c_st[-1, :-1] = 1
    c_st[-1, -1] = 0 
    c_list.append(c_st)
    a_i = np.linalg.solve(c_st, b)[:-1]
    a_list.append(a_i)



# reduced matrix

A_reduced = np.zeros((len(group_index), len(group_index)))
for i in range(len(group_index)):
    for  j in range(len(group_index)):
        if i == j:
            A_reduced[i][j] = val_i[i][j]
        else:
            Ax = np.real(A_submat[i][j].dot(vec_i[j][i] ))
            index = np.argmax(np.abs(Ax))
            lambd = Ax[index]/ vec_i[i][j][index]
            A_reduced[i][j] = lambd


# calculate mu
K_diag = []
for i in range(len(group_index)):
    K_i = []
    for j in range(len(group_index)):
        K_ij = np.diag(A_submat[i][j].sum(1) )
        K_i.append(K_ij)
    K_diag.append(K_i)
mu = np.zeros((len(group_index), len(group_index) ))
lambd = np.zeros((len(group_index), len(group_index) ))
A_W = np.zeros((len(group_index), len(group_index) ))
for i in range(len(group_index)):
    for j in range(len(group_index)):
        mu[i, j] = a_list[i].reshape(1, len(a_list[i])).dot(K_diag[i][j]).dot(a_list[i]) / np.sum(a_list[i]  ** 2)
        lambd[i, j] = a_list[j].reshape(1, len(a_list[j])).dot(A_submat[i][j].T).dot(a_list[i]) / np.sum(a_list[j] ** 2)
        A_W[i, j] = np.sum(K_diag[i][j] * a_list[i])



def f(x):
    return B + x * (1 - x/K) * ( x/C - 1)

def g(x_i, x_j):
    return x_i * x_j / (D + E * x_i + H * x_j)

def g1(x_i, x_j):
    return (D*x_j + H * x_j**2 ) / (D + E*x_i + H*x_j) **2

def g2(x_i, x_j):
    return (D*x_i + E * x_i**2 ) / (D + E*x_i + H*x_j) **2

def dynamics_reduced(x, t, f, g, g1, g2, A_reduced, mu_reduced, lambd_reduced):
    x_r = x.reshape(1, len(x))
    x_c = x.reshape(len(x), 1)
    #dxdt = f(x) + np.sum(A_reduced * g(x_c, x_r), 1) + np.sum( (mu_reduced - A_reduced) * g1(x_c, x_r), 1 )  * x
    dxdt = f(x) + np.sum(A_reduced * g(x_c, x_r), 1) + np.sum( (mu_reduced - A_reduced) * g1(x_c, x_r), 1 )  * x + ((lambd_reduced - A_reduced) *g2(x_c, x_r)).dot(x)
    
    return dxdt


initial_group =  np.ones(len(group_index)) * 0.1
t = np.arange(0, 1000, 0.01)
arguments = (B, C, D, E, H, K)
initial_multi = np.ones(N_actual) * 0.1
xs_group_list = []
xs_multi_list = []
weight_list = np.arange(0.1, 0.2, 0.01)
xs_multi_map = []
xs_beta_list = []
for w in weight_list:
    xs_group = odeint(dynamics_reduced, initial_group, t, args=(f, g, g1, g2, A_W * w, mu * w, lambd*w))
    xs_group_list.append(xs_group[-1])
    net_arguments = (index_i, index_j, w * A_interaction, cum_index)
    xs_multi = odeint(mutual_multi, initial_multi, t, args=(arguments, net_arguments))[-1]
    A_reduction, net_arguments_reduction, x_eff = reducednet_effstate(A_unit * w, initial_multi, group_index)
    xs_beta = odeint(mutual_multi, initial_group, t, args=(arguments, net_arguments_reduction))[-1]
    xs_multi_list.append(xs_multi)
    xs_beta_list.append(xs_beta)
    xs_map = [np.sum(a_list[i] * xs_multi[group_index[i]]) for i in range(len(group_index))]
    xs_multi_map.append(xs_map)


plt.plot(weight_list, np.vstack((xs_multi_map)) , 'r')
plt.plot(weight_list, np.vstack((xs_group_list)), 'b')


