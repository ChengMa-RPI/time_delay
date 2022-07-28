import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng, dde_RK45

import sympy as sp
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import odeint
import networkx as nx
import multiprocessing as mp
import time
from numpy import linalg as LA
import pandas as pd 
import scipy.io
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import itertools
from scipy import linalg as slin
from scipy.sparse.linalg import eigs as sparse_eig
from scipy.signal import find_peaks
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

cpu_number = 4
B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

def mutual_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt


def A_feature(network_type, N, seed, d, weight, space, tradeoff_para, method):
    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    A = A_unit * weight
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
    return A, feature, net_arguments

def xs_group_partition_bifurcation(A, feature, m, dynamics, arguments, attractor_value):
    group_index = group_index_from_feature_Kmeans(feature, m)
    t = np.arange(0, 1000, 0.01)
    A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(len(A)), group_index)
    dynamics_multi = globals()[dynamics + '_multi']
    initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
    xs_reduction = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
    return xs_reduction, group_index

def xs_multi_bifurcation(A, dynamics, arguments, attractor_value, net_arguments):
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(len(A)) * attractor_value
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    return xs_multi



def eigen_blocks(A, feature, m, dynamics, arguments, attractor_value):
    """TODO: Docstring for eigen_fg.

    :dynamics: TODO
    :xs: TODO
    :arguments: TODO
    :returns: TODO

    """
    xs_reduction, group_index = xs_group_partition_bifurcation(A, feature, m, dynamics, arguments, attractor_value)
    length_groups = len(group_index)
    N_actual = sum([len(i) for i in group_index])
    xs_groups = np.ones(N_actual)
    for i in range(length_groups):
        xs_groups[group_index[i]] = xs_reduction[i]
    xs_groups_T = xs_groups.reshape(len(xs_groups), 1)

    if dynamics == 'mutual':
        B, C, D, E, H, K = arguments
        fx_subgroup = (1-xs_reduction/K) * (2*xs_reduction/C-1)
        fxt_subgroup = -xs_reduction/K*(xs_reduction/C-1)

        denominator = D + E * xs_groups + H * xs_groups_T
        gx_i_groups = np.sum(A* (xs_groups_T/denominator - E * xs_groups * xs_groups_T/denominator ** 2 ), 0)
        gx_j_groups = A* (xs_groups/denominator - H * xs_groups * xs_groups_T/denominator ** 2 )

    "compute eigenvalues"
    tau_individual = []
    for i in range(length_groups):
        gx_i_subgroup = gx_i_groups[group_index[i]]
        gx_j_subgroup = gx_j_groups[group_index[i]][:, group_index[i]]
        L = gx_j_subgroup
        np.fill_diagonal(L, gx_i_subgroup)
        eigenvalue, eigenvector = np.linalg.eig(L)
        eigenvalue = np.real(eigenvalue)
        P = - (fx_subgroup[i] +eigenvalue)
        Q = - fxt_subgroup[i]
        PQ_index = np.where(np.abs(P/Q)<=1)[0]
        if len(PQ_index):
            P_index = P[PQ_index]
            Q_index = Q
            tau_list = np.arccos(-P_index/Q_index) /Q_index/np.sin(np.arccos(-P_index/Q_index))
            tau_individual.append(np.min(tau_list))

    tau_critical = np.min(tau_individual)

    return tau_critical

def tau_m(dynamics, arguments, network_type, N, seed, d, weight, m_list, attractor_value, space, tradeoff_para, method, tau_list, nu_list):
    """TODO: Docstring for eigen_fg.

    :dynamics: TODO
    :xs: TODO
    :arguments: TODO
    :returns: TODO

    """
    A, feature, net_arguments = A_feature(network_type, N, seed, d, weight, space, tradeoff_para, method)
    tau_group = []
    for m in m_list:
        tau_critical = eigen_blocks(A, feature, m, dynamics, arguments, attractor_value)
        tau_group.append(tau_critical)
    des = '../data/' + 'tau_compare/' 
    df_group = pd.DataFrame(np.vstack((m_list, tau_group)).transpose())
    group_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_group.csv'
    df_group.to_csv(group_file, index=None, header=None, mode='a')

    tau_multi = eigen_solution(A, dynamics, arguments, attractor_value, net_arguments, tau_list, nu_list)
    df_multi = pd.DataFrame([[tau_multi]])
    multi_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_eigen.csv'
    df_multi.to_csv(multi_file, index=None, header=None, mode='a')

    return tau_list, tau_multi

def eigenvalue_zero(x, A, fx, fxt, gx_i, gx_j):
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
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - gx_i) - gx_j 
    eigenvalue, eigenvector = np.linalg.eig(M)
    zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
    return np.array([np.real(zeropoint), np.imag(zeropoint)])

def eigen_solution(A, dynamics, arguments, attractor_value, net_arguments, tau_list, nu_list):
    """TODO: Docstring for eigen_fg.

    :dynamics: TODO
    :xs: TODO
    :arguments: TODO
    :returns: TODO

    """
    xs = xs_multi_bifurcation(A, dynamics, arguments, attractor_value, net_arguments)
    if dynamics == 'mutual':
        B, C, D, E, H, K = arguments
        fx = (1-xs/K) * (2*xs/C-1)
        fxt = -xs/K*(xs/C-1)
        xs_T = xs.reshape(len(xs), 1)
        denominator = D + E * xs + H * xs_T
        "A should be transposed to A_ji"
        gx_i = np.sum(A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
        gx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
    "compute eigenvalues"
    tau_sol = []
    for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
        print(initial_condition)
        tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, gx_i, gx_j))
        "check the solution given by fsolve built-in function."
        eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, gx_i, gx_j)
        if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
            #print(tau_solution, nu_solution)
            tau_sol.append(tau_solution)
    tau_sol = np.array(tau_sol)
    tau_critical = np.min(tau_sol[tau_sol>0])
    return tau_critical

def tau_evolution(network_type, N, seed, d, weight, dynamics, arguments, attractor_value, delay1, delay2, criteria_delay, criteria_dyn):
    """TODO: Docstring for tau_evolution.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, seed, d)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    N_actual = np.size(A, 0)
    initial_condition = np.ones(N_actual) * attractor_value 
    t = np.arange(0, 1000, 0.01)
    dynamics_multi = globals()[dynamics + '_multi']
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    initial_condition  = xs_multi - 0.01
    t = np.arange(0, 200, 0.001)
    dyn_dif = 1
    delta_delay = delay2 - delay1
    result = dict()
    while delta_delay > criteria_delay:
        if delay1 not in result:
            dyn_all1 = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay1, arguments, net_arguments))[-1000:]
            diff1 = np.max(np.max(dyn_all1, 0) - np.min(dyn_all1, 0))
            result[delay1] = diff1
        if delay2 not in result:
            dyn_all2 = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay2, arguments, net_arguments))[-1000:]
            diff2 = np.max(np.max(dyn_all2, 0) - np.min(dyn_all2, 0))
            result[delay2] = diff2
        if result[delay1] < criteria_dyn and (result[delay2] > criteria_dyn or np.isnan(result[delay2])):
            delay1 = np.round(delay1 + delta_delay/2, 10)
        elif result[delay1] > criteria_dyn or np.isnan(result[delay1]):
            delay2 = np.round(delay1, 10)
            delay1 = np.round(delay1 - delta_delay, 10)
        delta_delay = delay2 - delay1 
    df = pd.DataFrame([[delay1]])
    des = '../data/' + 'tau_compare/' 
    des_file = des + dynamics + '_' + network_type + f'_N={N}_d={d}_seed={seed}_weight={weight}_evolution.csv'
    df.to_csv(des_file, header=None, index=None, mode='a')
    return delay1

def tau_parallel(network_type, N, seed, d, weight_list, dynamics, arguments, attractor_value, delay1, delay2, criteria_delay, criteria_dyn, space, tradeoff_para, method, tau_list, nu_list):
    """TODO: Docstring for tau_evolution_parallel.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """

    des = '../data/' + 'tau_compare/' 
    if not os.path.exists(des):
        os.makedirs(des)

    p = mp.Pool(cpu_number)
    p.starmap_async(tau_m, [(dynamics, arguments, network_type, N, seed, d, weight, m_list, attractor_value, space, tradeoff_para, method, tau_list, nu_list) for weight in weight_list]).get()
    p.close()
    p.join()

    p = mp.Pool(cpu_number)
    p.starmap_async(tau_evolution, [(network_type, N, seed, d, weight, dynamics, arguments, attractor_value, delay1, delay2, criteria_delay, criteria_dyn) for weight in weight_list]).get()
    p.close()
    p.join()

    return None


dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1
N = 1000
network_type = 'SF'
d = [2.5, 999, 3]
seed = [1, 1]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]
m_list = [1]
 

m = 20
weight = 1
attractor_value = 5
space = 'log'
tradeoff_para = 0.5
method = 'degree'
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)

weight_list = np.round(np.hstack(([1], np.arange(0.05, 1, 0.05) )), 5)

delay1 = 0.1
delay2 = 1
criteria_delay = 0.01
criteria_dyn = 1e-3
tau_parallel(network_type, N, seed, d, weight_list, dynamics, arguments, attractor_value, delay1, delay2, criteria_delay, criteria_dyn, space, tradeoff_para, method, tau_list, nu_list)
