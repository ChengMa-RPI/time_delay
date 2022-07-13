import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import network_generate
from multi_dynamics import group_partition_degree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import time

def heterogeneity(degree_list):
    """TODO: the network heterogeneity using Barabasi definition in 'Controllability of complex networks'.

    :degree_list: a list of degrees for subgroups
    :returns: TODO

    """
    h_list = []
    m = len(degree_list)
    for degree in degree_list:
        n = len(degree)
        k_ave = sum(degree) / n
        h = np.sum(np.abs(degree - degree.reshape(degree.size, 1)))/n**2/k_ave
        h_list.append(h)
    h_whole = sum(h_list)/ m
    return h_whole

def modularity(A, group_index):
    """TODO: Docstring for modularity.
    :returns: TODO

    """
    t1 = time.time()
    N = np.size(A, 0)
    k = np.sum(A, 0)
    l = sum(k)/2
    k_matrix = k * k.reshape(N, 1)/2/l
    delta_matrix = np.zeros((N, N))
    t2 = time.time()
    for i in group_index:
        delta_matrix[np.ix_(i, i)] = 1

    t3 = time.time()
    Q = -np.sum((A - k_matrix) * delta_matrix) / 2 / l
    #Q = np.sum(A * (1 - delta_matrix))/ 2 / l
    t4 = time.time()
    #print(t2-t1, t3- t2, t4-t3)
    return Q

def modularity_change(A, l, degree, node_move, group_in, group_out):
    """TODO: Docstring for modularity.
    :returns: TODO

    """
    connection_i = A[node_move]
    k_i = degree[node_move]
    k_i_in = connection_i[group_in].sum()
    k_i_out = connection_i[group_out].sum()
    total_in = degree[group_in].sum()
    total_out = degree[group_out].sum()
    delta_Q = -1 / l * (k_i_in - k_i_out - total_in * k_i/2/l + (total_out-k_i) * k_i/2/l)
    return delta_Q

def interaction_change(A, l, degree, node_move, group_in, group_out):
    """TODO: Docstring for modularity.
    :returns: TODO

    """
    connection_i = A[node_move]
    k_i_in = connection_i[group_in].sum()
    k_i_out = connection_i[group_out].sum()
    delta_Q = -1 / l * (k_i_in - k_i_out)
    return delta_Q

def heterogeneity_barabasi_change(degree_in, degree_out):
    """TODO: the network heterogeneity using Barabasi definition in 'Controllability of complex networks'.

    :degree_list: a list of degrees for subgroups
    :returns: TODO

    """
    t1 = time.time()
    n_in = len(degree_in)
    n_out = len(degree_out)

    k_ave_in = degree_in.sum() / n_in
    k_ave_out = degree_out.sum() / n_out
    t2 = time.time()
    h_in = np.sum(np.abs(degree_in - degree_in.reshape(len(degree_in), 1)))/n_in**2 / k_ave_in
    h_out = np.sum(np.abs(degree_out - degree_out.reshape(len(degree_out), 1)))/n_out**2 / k_ave_out
    t3 = time.time()
    print(t2 - t1, t3 - t2)
    h_in_out = h_in + h_out
    return h_in_out

def heterogeneity_momentum_change(degree):
    """TODO: the network heterogeneity using Barabasi definition in 'Controllability of complex networks'.

    :degree_list: a list of degrees for subgroups
    :returns: TODO

    """
    n = len(degree)
    if n == 0:
        h = 0
    else:
        degree_ave = degree.sum()/n
        k_sq_ave = (degree ** 2).sum()/n
        h = (k_sq_ave - degree_ave**2)/k_sq_ave
    return h

def heterogeneity_range_change(degree):
    """TODO: the network heterogeneity using Barabasi definition in 'Controllability of complex networks'.

    :degree_list: a list of degrees for subgroups
    :returns: TODO

    """
    n = len(degree)
    if n == 0:
        h = 0
    else:
        h = 1 - degree.min()/degree.max()
    return h

def objective_function(A, group_index, r, T):
    """TODO: Docstring for objective_function.
    :returns: TODO

    """
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    l = np.sum(A)/2
    t0 = time.time()
    moveout_group, movein_group, node_move = random_move(group_index, N_actual)
    t1 = time.time()
    Q = modularity_change(A, l, w, node_move, group_index[movein_group], group_index[moveout_group])
    t2 = time.time()

    degree_in = w[group_index[movein_group]]
    degree_out = w[group_index[moveout_group]]
    h_before = heterogeneity_momentum_change(degree_in, degree_out)

    group_index_change = group_index.copy()
    group_index_change[moveout_group] = group_index_change[moveout_group][group_index_change[moveout_group]!=node_move]
    group_index_change[movein_group] = np.hstack((group_index_change[movein_group], node_move))

    t3 = time.time()
    degree_in = w[group_index_change[movein_group]]
    degree_out = w[group_index_change[moveout_group]]
    h_after = heterogeneity_momentum_change(degree_in, degree_out)

    H = (h_after - h_before)/len(group_index)
    t4 = time.time()
    print(t1 -t0, 'Q', t2 - t1, 'H:', t3 -t2, t4 - t3)
    S = r * H + (1-r) * Q
    if S <= 0:
        group_index = group_index_change
    else:
        p = np.exp(-S/T)
        R = np.random.uniform(1)
        if p > R:
            group_index = group_index_change
    return S

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

def one_trial(A, w, l, group_in, group_out, node_move, group_length, r, T, R):
    """TODO: Docstring for objective_function.
    :returns: TODO

    """
    Q = modularity_change(A, l, w, node_move, group_in, group_out)
    #Q = interaction_change(A, l, w, node_move, group_in, group_out)

    h_in = heterogeneity_momentum_change(w[group_in])
    h_out = heterogeneity_momentum_change(w[group_out])
    #h_in = heterogeneity_range_change(w[group_in])
    #h_out = heterogeneity_range_change(w[group_out])
    h_before = h_in + h_out

    group_in_change = group_in.copy()
    group_out_change = group_out.copy()
    group_out_change = group_out_change[group_out_change!=node_move]
    group_in_change = np.hstack((group_in_change, node_move))

    h_in = heterogeneity_momentum_change(w[group_in_change])
    h_out = heterogeneity_momentum_change(w[group_out_change])
    #h_in = heterogeneity_range_change(w[group_in_change])
    #h_out = heterogeneity_range_change(w[group_out_change])
    h_after = h_in + h_out

    H = (h_after - h_before)/group_length
    S = r * H + (1-r) * Q
    #print(H, Q)
    if S <= 0:
        group_in = group_in_change
        group_out = group_out_change
        delta_S = S
        delta_H = H
        delta_Q = Q
    else:
        p = np.exp(-S/T)
        if p > R:
            group_in = group_in_change
            group_out = group_out_change
            delta_S = S
            delta_H = H
            delta_Q = Q
        else:
            delta_S = 0
            delta_H = 0
            delta_Q = 0
    return group_in, group_out, delta_H, delta_Q, delta_S

def many_trial(network_type, N, beta, betaeffect, seed, d, group_num, r, T0, T_num, T_decay, N_trial):
    """TODO: Docstring for network_info.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    T_list = np.array([T_decay**i for i in range(T_num)]) * T0
    w = np.sum(A, 0)
    N_actual = np.size(A, 0)
    l = np.sum(A)/2
    if network_type == 'SF':
        space = 'log'
    elif network_type == 'ER':
        space = 'linear'
    group_index, rearange_index = group_partition_degree(w, group_num, N_actual, space)
    delta_S_list = []
    delta_Q_list = []
    delta_H_list = []

    for T in T_list:
        node_move_list = np.random.choice(N_actual, N_trial)
        R_list = np.random.uniform(size=N_trial)
        for node_move, R in zip(node_move_list, R_list):
            group_length = len(group_index)
            moveout_group, movein_group = random_move(group_index, node_move)
            group_in = group_index[movein_group]
            group_out = group_index[moveout_group]
            group_in, group_out, delta_H, delta_Q, delta_S = one_trial(A, w, l, group_in, group_out, node_move, group_length, r, T, R)
            group_index[movein_group] = group_in
            group_index[moveout_group] = group_out
            delta_S_list.append(delta_S)
            delta_H_list.append(delta_H)
            delta_Q_list.append(delta_Q)
    H = np.cumsum(delta_H_list)
    Q = np.cumsum(delta_Q_list)
    S = np.cumsum(delta_S_list)
    "write to .txt file"
    des_file = '../data/network_partition/' + network_type  + '/'
    if not os.path.exists(des_file):
        os.makedirs(des_file)
    file_name = des_file + f'N={N}_d={d}_seed={seed}_group_num={group_num}_trial={N_trial}_r={r}_T=[{T0}, {T_num}, {T_decay}].txt'
    with open(file_name, 'w') as output:
        for i in range(group_length):
            output.write(','.join(map(str, group_index[i])) +'\n')
    return group_index, H, Q, S






N = 1000
beta = 1
betaeffect = 0
network_type = 'ER'
seed = 0
d = 5
network_type = 'SF'
seed = [0, 0]
d = [2.5, 999, 3]
group_num = 2
r = 0.99
T = 1e-3
N_trial = 100000
#T_list = np.array([0.95**i for i in range(30)]) * 1e-5
T0 = 1e-5
T_num = 10
T_decay = 0.9

t1 = time.time()
#a, H, Q, S = many_trial(network_type, N, beta, betaeffect, seed, d, group_num, r, T0, T_num, T_decay, N_trial)
t2 = time.time()
print(t2 - t1)
