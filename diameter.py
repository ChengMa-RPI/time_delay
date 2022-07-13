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

def diameter_calculation(network_type, N, beta, betaeffect, seed_list, d):
    """TODO: Docstring for diameter.

    :network_type: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/diameter/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'd={d}.csv'
    for seed in seed_list:
        A, _, _, _, _ = network_generate(network_type, N, beta, betaeffect, seed, d)
        G = nx.convert_matrix.from_numpy_matrix(A)
        diameter = nx.algorithms.distance_measures.diameter(G)
        data = np.hstack((seed, diameter))
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')

    return diameter

def beta_calculation(network_type, N, wt, seed_list, d):
    """TODO: Docstring for diameter.

    :network_type: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/beta_wt/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'd={d}_wt={wt}.csv'
    for seed in seed_list:
        A, _, _, _, _ = network_generate(network_type, N, wt, 0, seed, d)
        beta, _ = betaspace(A, [0])
        data = np.hstack((seed, beta))
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')

    return None

network_type = 'SF'
dynamics = 'mutual'
N = 1000
beta = 0.1
betaeffect = 1
seed_list = np.arange(100).tolist()
seed_SF = np.vstack((seed_list, seed_list)).transpose().tolist()

d_list = [2000, 4000, 8000]
d_list = [[3, 999, 4], [3.8, 999, 5]]
wt_list = [1]
for d in d_list:
    #diameter = diameter_calculation(network_type, N, beta, betaeffect, seed_SF, d)
    for wt in wt_list:
        beta_calculation(network_type, N, wt, seed_list, d)
        pass
