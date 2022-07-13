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

def save_ygl(network_type, N, d, seed, dynamics, m, space):
    """TODO: Docstring for error_ygl.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed_list: TODO
    :dynamics: TODO
    :returns: TODO

    """
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A_unit = scipy.sparse.load_npz(file_A).toarray()
    """    
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    beta_cal = betaspace(A_unit, [0])[0]
    if m == N:
        des_multi = des + 'xs_multi_beta/'
        file_multi = des_multi + f'N={N}_d={d}_seed={seed}.csv'
        data_multi = np.array(pd.read_csv(file_multi, header=None))
        weight = data_multi[:, 0]
        xs = data_multi[:, 1:]
        save_file = des + 'y_multi_beta/' + f'N={N}_d={d}_seed={seed}.csv'
    else:
        des_group = des + f'degree_kmeans_space={space}_beta/'
        file_group = des_group + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data_group = np.array(pd.read_csv(file_group, header=None))
        G = nx.from_numpy_array(A_unit)
        feature = feature_from_network_topology(A_unit, G, space, tradeoff_para=0.5, method='degree')
        group_index = group_index_from_feature_Kmeans(feature, m)
        y_group = data_group[:, 1:]
        xs_group = np.zeros( (len(data_group), N) )
        for i, group_i in enumerate(group_index):
            xs_group[:, group_i] = y_group[:, i:i+1]
        xs = xs_group
        weight = data_group[:, 0]
        save_file =  des + 'y_group_beta/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
    y_gl = betaspace(A_unit, xs)[-1]
    beta_list = beta_cal * weight
    data_save = np.vstack( (weight, beta_list, y_gl) ) 
    df = pd.DataFrame(data_save.transpose())
    df.to_csv(save_file, index=None, header=None, mode='w')
    return None
    
def save_ygl_parallel(network_type, N, d_list, seed_list, dynamics, m_list, space):
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    save_multi = des + 'y_multi_beta/'
    save_group = des + 'y_group_beta/'
    for des in [save_multi, save_group]:
        if not os.path.exists(des):
            os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(save_ygl, [(network_type, N, d, seed, dynamics, m, space) for seed, d in zip(seed_list, d_list) for m in m_list]).get()
    p.close()
    p.join()
    return None

def ygl_error(network_type, N, d, seed, dynamics, m, space, beta_list, save_des):
    """TODO: Docstring for ygl_error.

    :namics: TODO
    :m: TODO
    :space: TODO
    :returns: TODO

    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_A).toarray()
    degrees = np.sum(A, 0)
    kmean = np.mean(degrees)
    h1 = np.mean(degrees ** 2) / kmean - kmean
    h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / kmean

    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    file_multi = des + 'y_multi_beta/' + f'N={N}_d={d}_seed={seed}.csv'
    file_group =  des + 'y_group_beta/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
    data_multi = np.array(pd.read_csv(file_multi, header=None))
    data_group = np.array(pd.read_csv(file_group, header=None))
    weight_list, beta_multi, ygl_multi = data_multi.transpose()
    weight_list, beta_group, ygl_group = data_group.transpose()
    for beta in beta_list:
        index_multi = np.where(np.abs(beta_multi - beta) < 1e-5)[0]
        index_group = np.where(np.abs(beta_group - beta) < 1e-5)[0]
        if not len(index_multi) and len(index_group):
            print(index_multi, index_group, 'no data')
            break
        else:
            yi_multi = ygl_multi[index_multi[0]]
            yi_group = ygl_multi[index_group[0]]
            error = np.round( np.abs(yi_multi - yi_group), 10) / (yi_multi + yi_group)

        df = pd.DataFrame( np.array([d, seed, kmean, h1, h2, error], dtype='object').reshape(1, 6) ) 
        df.to_csv(save_des + f'm={m}_beta={beta}.csv', index=None, header=None, mode='a')
    return None

def ygl_error_parallel(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    save_des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/ygl_error/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    p = mp.Pool(cpu_number)
    p.starmap_async(ygl_error, [(network_type, N, d, seed, dynamics, m, space, beta_list, save_des) for seed, d in zip(seed_list, d_list) for  m in m_list]).get()
    p.close()
    p.join()
    return None

def read_ygl_together(network_type, N, d, seed, dynamics, m, space):
    """TODO: Docstring for read_ygl_together.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :seed: TODO
    :dynamics: TODO
    :m: TODO
    :space: TODO
    :returns: TODO

    """
    file_A = '../data/A_matrix/' + network_type + '/' + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(file_A).toarray()
    degrees = np.sum(A, 0)
    kmean = np.mean(degrees)
    h1 = np.mean(degrees ** 2) / kmean - kmean
    h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / kmean
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/'
    if m == N:
        file_multi = des + 'y_multi_beta/' + f'N={N}_d={d}_seed={seed}.csv'
        data = np.array(pd.read_csv(file_multi, header=None))
    else:
        file_group =  des + 'y_group_beta/' + f'N={N}_d={d}_number_groups={m}_seed={seed}.csv'
        data = np.array(pd.read_csv(file_group, header=None))
    weight_list, beta_list, ygl_list = data.transpose()
    for beta in beta_list:
        index = np.where(np.abs(beta_list - beta) < 1e-5)[0]
        if not len(index):
            print(index, 'no data')
            break
        else:
            yi = ygl_list[index[0]]

        df = pd.DataFrame( np.array([d, seed, kmean, h1, h2, yi], dtype='object').reshape(1, 6) ) 
        df.to_csv(save_des + f'm={m}_beta={beta}.csv', index=None, header=None, mode='a')
    return None

def read_ygl_parallel(network_type, N, d_list, seed_list, dynamics, m_list, space, beta_list):
    save_des = '../data/' + dynamics + '/' + network_type + '/' + 'xs_bifurcation/ygl/'
    if not os.path.exists(save_des):
        os.makedirs(save_des)
    p = mp.Pool(cpu_number)
    p.starmap_async(read_ygl_together, [(network_type, N, d, seed, dynamics, m, space, beta_list, save_des) for seed, d in zip(seed_list, d_list) for  m in m_list]).get()
    p.close()
    p.join()
    return None

dynamics = 'genereg'
dynamics = 'CW'
dynamics = 'mutual'
network_type = 'SF'
space = 'log'
N = 1000
d_list = [[2, 999, 3]]
seed_list = [[0, 0]]
m_list = [1, 5, 10, N]

cpu_number = 4
save_ygl_parallel(network_type, N, d_list, seed_list, dynamics, m_list, space)
