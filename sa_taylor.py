import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import network_generate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import time
import networkx as nx
from sklearn.cluster import KMeans


def feature_from_network_topology(A, G, space, tradeoff_para, method):
    """TODO: Docstring for feature_from_network_topology.

    :off: TODO
    :returns: TODO

    """
    core_number = np.array(list(nx.core_number(G).values())) 
    k = np.sum(A>0, 0)
    N_actual = len(k)
    if method == 'kcore':
        feature = core_number/core_number.max()
    elif method == 'degree':
        if space == 'log':
            feature = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature = k/k.max()
    elif method == 'kcore_degree':
        feature1 = core_number/core_number.max()
        if space ==  'log':
            feature2 = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature2 = k/k.max()
        feature = feature1 * tradeoff_para + feature2 * (1-tradeoff_para)
    elif method == 'kcore_KNN_degree':
        core_group = np.where(core_number == core_number.max())[0]
        KNN = neighborshell_given_core(G, A, core_group)
        KNN_num = np.zeros((N_actual))
        for i, KNN_i in enumerate(KNN):
            for j in KNN_i:
                KNN_num[j] = len(KNN) - i
        feature1 = KNN_num / KNN_num.max()
        if space ==  'log':
            feature2 = np.log(k) / np.log(k).max()
        elif space == 'linear':
            feature2 = k/k.max()
        feature = feature1 * tradeoff_para + feature2 * (1-tradeoff_para)
    return feature

def group_index_from_feature_Kmeans(x, number_groups):
    """TODO: Docstring for group_index_from_feature.

    :arg1: TODO
    :returns: TODO

    """
    x_unique = np.unique(x)
    if number_groups >= np.size(x_unique):
        group_index = []
        for x_i in np.sort(x_unique):
            group_index.append(np.where(x == x_i)[0])
        print('number of groups is too large')

    else:
        kmeans = KMeans(n_clusters=number_groups, random_state=0).fit(x.reshape(-1, 1))
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        group_index = []
        for i in np.argsort(centers[:, 0]):
            group_index.append(np.where(labels == i)[0])
    return group_index[::-1]


def objective_function(A, w, group_index):
    """TODO: Docstring for objective_function.
    :returns: TODO

    """
    group_length = len(group_index)
    h1_list = np.zeros((group_length))
    h2_list = np.zeros((group_length, group_length))
    for kth, group_k in enumerate(group_index):
        s_k = w[group_k]
        a_k = s_k / np.sum(s_k)  # weighted vector for the group k
        t3 = a_k * s_k
        t4 = np.sum(t3) * a_k
        alpha_k = np.sum(t3 * t4) / np.sum(t4 ** 2)
        h1 = t3 - alpha_k * t4
        h1_list[kth] = np.sum(h1**2)
        for lth, group_l in enumerate(group_index):
            A_kl = A[group_k][:, group_l]
            s_kl = np.sum(A_kl, 1)
            s_l = w[group_l]
            a_l = s_l / np.sum(s_l)  
            t1 = A_kl.transpose().dot(a_k)
            t2 = a_l * np.sum(s_kl * a_k)
            if np.abs(np.sum(t2)) <1e-10:
                gamma_kl = 1
            else:
                gamma_kl = np.sum(t1 * t2) / np.sum(t2**2)
            #beta_kl = t1.sum() 
            h2 = t1 - gamma_kl *  t2
            h2_list[kth, lth] = np.sum(h2**2)
    return h1_list, h2_list

def objective_function_change(A, w, group_index, movein_group, moveout_group, h1_list, h2_list):
    """TODO: Docstring for objective_function.
    :returns: TODO

    """
    group_length = len(group_index)
    for kth in range(group_length):
        group_k = group_index[kth]
        s_k = w[group_k]
        a_k = s_k / np.sum(s_k)  # weighted vector for the group k
        if kth in [movein_group, moveout_group]:
            t3 = a_k * s_k
            t4 = np.sum(t3) * a_k
            alpha_k = np.sum(t3 * t4) / np.sum(t4 ** 2)
            h1 = np.sum((t3 - alpha_k * t4)**2)
            h1_list[kth] = h1
        for lth in range(group_length):
            group_l = group_index[lth]
            if (kth in [movein_group, moveout_group]) or (lth in [movein_group, moveout_group]):
                A_kl = A[group_k][:, group_l]
                s_kl = np.sum(A_kl, 1)
                s_l = w[group_l]
                a_l = s_l / np.sum(s_l)  
                t1 = A_kl.transpose().dot(a_k)
                t2 = a_l * np.sum(s_kl * a_k)
                if np.abs(np.sum(t2)) <1e-10:
                    gamma_kl = 1
                else:
                    gamma_kl = np.sum(t1 * t2) / np.sum(t2**2)
                h2_part = np.sum((t1 - gamma_kl *  t2) ** 2)
                h2_list[kth, lth] = h2_part
    return h1_list, h2_list

def random_move(group_index, node_move):
    """TODO: Docstring for random_move.

    :A: TODO
    :group_index: TODO
    :returns: TODO

    """
    group_length = len(group_index)
    for i, j in enumerate(group_index):
        if node_move in j:
            moveout_group = i 
            break
    group_list = np.arange(group_length)
    movein_group = np.random.choice(group_list[group_list != moveout_group])
    return moveout_group, movein_group

def one_trial(A, w, N, group_index, movein_group, moveout_group, node_move, h1_list, h2_list, T, R):
    """TODO: Docstring for objective_function.
    :returns: TODO

    """
    h_original = np.sqrt(h1_list + np.sum(h2_list, 1))
    group_each_length = np.array([len(i) for i in group_index])
    S1 = np.sum(h_original * group_each_length) / np.sum(group_each_length)  
    group_index_change = group_index.copy()
    group_in_change = group_index_change[movein_group]
    group_out_change = group_index_change[moveout_group]
    group_index_change[moveout_group] = group_out_change[group_out_change!=node_move]
    group_index_change[movein_group] = np.hstack((group_in_change, node_move))

    h1_list_change, h2_list_change = objective_function_change(A, w, group_index_change, movein_group, moveout_group, h1_list.copy(), h2_list.copy())
    h_change = np.sqrt(h1_list_change + np.sum(h2_list_change, 1))
    group_each_length = np.array([len(i) for i in group_index_change])
    S2 = np.sum(h_change * group_each_length) / np.sum(group_each_length)  
    S = S2 - S1

    if S <= 0 or (S>0 and np.exp(-S/T) > R):
        group_index = group_index_change
        h1_list = h1_list_change
        h2_list = h2_list_change
        change = S
    else:
        change = 0
    return group_index, h1_list, h2_list, change

def many_trial(network_type, N, beta, betaeffect, seed, d, number_groups, T0, T_num, T_decay, N_trial):
    """TODO: Docstring for network_info.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    T_list = np.array([T_decay**i for i in range(T_num)]) * T0
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    each_number_groups = int(np.sum([N])/number_groups)
    group_index = [np.arange(each_number_groups*i, each_number_groups * (i+1)) for i in range(number_groups)]
    """
    G = nx.from_numpy_array(A)
    method = 'degree'
    if network_type == 'SF':
        space = 'log'
    else:
        space = 'linear'
    feature = feature_from_network_topology(A, G, space, 0.5, method)
    group_index = group_index_from_feature_Kmeans(feature, number_groups)
    """
    h1_list, h2_list =  objective_function(A, w, group_index)

    change_list = []
    for T in T_list:
        print(T)
        node_move_list = np.random.choice(N_actual, N_trial)
        R_list = np.random.uniform(size=N_trial)
        for node_move, R in zip(node_move_list, R_list):
            group_length = len(group_index)
            moveout_group, movein_group = random_move(group_index, node_move)
            t1 = time.time()
            group_index, h1_list, h2_list, change = one_trial(A, w, N, group_index, movein_group, moveout_group, node_move, h1_list, h2_list, T, R)
            change_list.append(change)
            t2 = time.time()
    "write to .txt file"
    des_file = '../data/network_partition/' + network_type  + '/'
    if not os.path.exists(des_file):
        os.makedirs(des_file)
    file_name = des_file + f'N={N}_d={d}_seed={seed}_number_groups={number_groups}_trial={N_trial}_T=[{T0}, {T_num}, {T_decay}].txt'
    with open(file_name, 'w') as output:
        for i in range(group_length):
            output.write(','.join(map(str, group_index[i])) +'\n')
    return group_index, change_list






N = 1000
beta = 1
betaeffect = 0
network_type = 'ER'
seed = 0
d = 5
network_type = 'SF'
seed = [0, 0]
d = [2.5, 999, 3]
number_groups = 2
T = 1e-3
N_trial = 1000
#T_list = np.array([0.95**i for i in range(30)]) * 1e-5
T0 = 1e-5
T_num = 10
T_decay = 0.9

#a, H, Q, S = many_trial(network_type, N, beta, betaeffect, seed, d, number_groups, r, T0, T_num, T_decay, N_trial)

beta = 0.1
betaeffect = 0

N = [100, 100, 100]
network_type = 'SBM_ER'
seed = 0
d = np.array([[0.9, 0.0005, 0.0005], [0.0005, 0.5, 0.0005], [0.0005, 0.0005, 0.1]]).tolist()
number_groups = 5

N = 1000
network_type = 'SF'
seed = [0, 0]
d = [2.5, 999, 3]
number_groups = 5




N_trial = 100000
T0 = 1e-4
T_num = 40
T_decay = 0.9

t1 = time.time()
group_index, change_list = many_trial(network_type, N, beta, betaeffect, seed, d, number_groups, T0, T_num, T_decay, N_trial)
t2 = time.time()
print(t2 - t1)
