import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, mutual_multi, network_generate, stable_state, ode_Cheng, ddeint_Cheng

import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
import networkx as nx
import multiprocessing as mp
import time
import os
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


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

fs = 22
ticksize = 14
alpha = 0.8


def mutual_multi_delay(f, x0, t, dt, d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd1 = np.where(t>d1, f[int((t-d1)/dt)], x0)
    xd2 = np.where(t>d2, f[int((t-d2)/dt)], x0)
    xd3 = np.where(t>d3, f[int((t-d3)/dt)], x0)

    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd1/K) * ( xd2/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    #dxdt = sum_f + x * np.array([sum_g[i:j].sum() for i, j in zip(cum_index[:-1], cum_index[1:])])

    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def evolution(network_type, N, beta, seed, arguments, d1, d2, d3, d=None):
    """TODO: Docstring for evolution.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    N = np.size(A, 0)
    initial_condition = np.ones(N) * 5 
    t = np.arange(0, 1000,0.01)
    #dyn_all = ddeint(mutual_multi_delay_original, g, t, fargs=(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
    t1 = time.time()
    dyn_all = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
    t2 = time.time()
    print(t2-t1)
    plt.plot(t, np.mean(dyn_all, 1), alpha = alpha)
    #plt.plot(t, dyn_all, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$\\langle x \\rangle$', fontsize =fs)
    plt.show()
    return dyn_all

def eigenvalue_zero(x, A, fx, fxt, Degree, gx_i, gx_j):
    """TODO: Docstring for matrix_variable.

    :x: TODO
    :fx: TODO
    :fxt: TODO
    :degree: TODO
    :gx_i: TODO
    :gx_j: TODO
    :returns: TODO

    """
    imag = 1j
    tau, nu = x
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - Degree * gx_i) - A * gx_j 
    eigenvalue, eigenvector = np.linalg.eig(M)
    zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
    return np.array([np.real(zeropoint), np.imag(zeropoint)])

def tau_eigenvalue(network_type, N, beta, nu_set, tau_set, arguments, seed, d=None):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    degree_weighted = np.sum(A, 0)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments, d=d)
    xs = xs_high

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2

    tau_sol = np.ones((np.size(tau_set), np.size(nu_set))) * 10
    for tau, i in zip(tau_set, range(np.size(tau_set))):
        for nu, j in zip(nu_set, range(np.size(nu_set))):
            t1=time.time()
            initial_condition = np.array([tau, nu])
            tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, degree_weighted, gx_i, gx_j))
            eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, degree_weighted, gx_i, gx_j)
            if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                tau_sol[i, j] = tau_solution
            t2=time.time()
            print(tau, nu, t2-t1, tau_solution)
    return tau_sol, A

def tau_multi_critical(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_critical = np.zeros(np.size(beta_set))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        tau_sol, A = np.ravel(tau_eigenvalue(network_type, N, beta, nu_set, tau_set, arguments, seed, d=d))
        tau_critical[i] = np.min(tau_sol[tau_sol>0])
        t2 = time.time()
        print(i, t2 - t1, tau_critical)

    degree = np.sum(np.heaviside(A, 0), 0)
    hetero = np.sum((degree - np.mean(degree))**2)/ N
    data = np.hstack((seed, np.mean(degree), hetero, tau_critical))
    data = pd.DataFrame(data.reshape(1, np.size(data)))
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    data.to_csv(des + network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None, index=None, mode='a')
    return tau_critical

imag = 1j
N = 2500

tau_set = np.array([0.316])
nu_set = np.array([5.2])
tau_set = np.arange(0.2, 0.4, 0.1)
nu_set = np.arange(1, 10, 1)
arguments = (B, C, D, E, H, K)
d = 0.8
d = 3
d_set = [900]
d_set = [200, 400, 600, 900, 1600, 2500]
d_set = [5, 10, 15, 20, 25, 50]
d_set = [2, 2.5, 3, 3.5, 4, 4.5, 5]
d_set = [3]
network_type = 'real'
network_type = 'RR'
network_type = '2D'
network_type = 'BA'
network_type = 'ER'
network_type = 'SF'
seed_set = np.arange(0, 100, 1).tolist()
beta_set = np.arange(1, 2, 1)

'''
for d in d_set:
    for seed in seed_set:
        tau_c = tau_multi_critical(network_type, N, arguments, beta_set, seed, nu_set = nu_set, tau_set = tau_set, d = d)
'''

beta = 1
d1 = 0.272
d2 = 0
d3 = 0
seed = 1
dyn_all = evolution(network_type, N, beta, seed, arguments, d1, d2, d3, d)

