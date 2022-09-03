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



class Tau_Solution():
    def __init__(self, network_type, N, d, seed, m_list, dynamics, weight_list, arguments, attractor_value, tau_list, nu_list, delay1, delay2):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        if network_type == 'SF':
            self.space = 'log'
        else:
            self.space = 'linear'
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.m_list = m_list
        self.dynamics = dynamics
        self.weight_list = weight_list
        self.arguments = arguments
        self.attractor_value = attractor_value
        self.A_unit, self.A_interaction, self.index_i, self.index_j, self.cum_index = network_generate(network_type, N, 1, 0, seed, d)
        self.G = nx.from_numpy_array(self.A_unit)
        self.N_actual = len(self.A_unit)
        self.dynamics_multi = globals()[dynamics + '_multi']
        self.dynamics_multi_delay = globals()[dynamics + '_multi_delay']
        self.tau_list = tau_list
        self.nu_list = nu_list
        self.delay1 = delay1
        self.delay2 = delay2


    def get_feature(self, weight):
        """TODO: Docstring for feature.

        :arg1: TODO
        :returns: TODO

        """
        tradeoff_para = 0.5
        method = 'degree'
        A = self.A_unit * weight
        feature = feature_from_network_topology(A, self.G, self.space, tradeoff_para, method)
        return feature
    
    def simu_xs_group(self, weight, m):
        """TODO: Docstring for xs_group.

        :arg1: TODO
        :returns: TODO

        """
        feature = self.get_feature(weight)
        A = self.A_unit * weight
        group_index = group_index_from_feature_Kmeans(feature, m)
        t = np.arange(0, 1000, 0.01)
        A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(len(A)), group_index)
        initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * self.attractor_value
        xs_reduction = odeint(self.dynamics_multi, initial_condition_reduction_deg_part, t, args=(self.arguments, net_arguments_reduction_deg_part))[-1]
        return xs_reduction, A_reduction_deg_part, net_arguments_reduction_deg_part, group_index

    def simu_xs_multi(self, weight):
        """TODO: Docstring for xs_multi.

        :arg1: TODO
        :returns: TODO

        """
        t = np.arange(0, 1000, 0.01)
        initial_condition = np.ones(self.N_actual) * self.attractor_value
        net_arguments = (self.index_i, self.index_j, weight * self.A_interaction, self.cum_index)
        xs_multi = odeint(self.dynamics_multi, initial_condition, t, args=(self.arguments, net_arguments))[-1]
        return xs_multi

    def get_derivative(self, A, xs):
        """TODO: Docstring for function.

        :xs: TODO
        :returns: TODO

        """
        A_T = A.transpose()
        if self.dynamics == 'mutual':
            B, C, D, E, H, K = self.arguments
            fx = (1-xs/K) * (2*xs/C-1)
            fxt = -xs/K*(xs/C-1)
            xs_T = xs.reshape(len(xs), 1)

            "A_ij: interaction from j to i, should be transposed to A_ji for directed networks (dimension reduction)"
            denominator = D + E * xs + H * xs_T
            gx_i = np.sum(A_T * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
            gx_j = A_T * (xs/denominator - H * xs * xs_T/denominator ** 2 )

        return fx, fxt, gx_i, gx_j

    def eigenvalue_zero(self, x, fx, fxt, gx_i, gx_j):
        """TODO: Docstring for eigenvalue_zero.

        :arg1: TODO
        :returns: TODO

        """
        imag = 1j
        tau, nu = x
        M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - gx_i) - gx_j 
        eigenvalue, eigenvector = np.linalg.eig(M)
        zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
        return np.array([np.real(zeropoint), np.imag(zeropoint)])

    def tau_eigen(self, weight, m, save_file):
        """TODO: Docstring for eigen_fg.

        :returns: TODO

        """
        if m == self.N_actual:
            xs = self.simu_xs_multi(weight)
            A = self.A_unit * weight
        else:
            xs, A, net_arguments, group_index = self.simu_xs_group(weight, m)
        fx, fxt, gx_i, gx_j = self.get_derivative(A, xs)
        "compute eigenvalues"
        tau_sol = []
        tau_list = self.tau_list
        nu_list = self.nu_list
        for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
            tau_solution, nu_solution = fsolve(self.eigenvalue_zero, initial_condition, args=(fx, fxt, gx_i, gx_j))
            "check the solution given by fsolve built-in function."
            eigen_real, eigen_imag = self.eigenvalue_zero(np.array([tau_solution, nu_solution]), fx, fxt, gx_i, gx_j)
            if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                tau_sol.append(tau_solution)
        tau_sol = np.array(tau_sol)
        tau_critical = np.min(tau_sol[tau_sol>0])
        tau_m = pd.DataFrame(np.array([m, tau_critical]).reshape(1, 2) )
        print(tau_m)
        tau_m.to_csv(save_file, index=None, header=None, mode='a')
        return tau_critical

    def tau_evolution(self, weight, m, save_file, criteria_delay=1e-2, criteria_dyn=1e-3):
        """TODO: Docstring for tau_evolution.
         
        :returns: TODO
        
        """
        delay1 = self.delay1
        delay2 = self.delay2
        if m == self.N_actual:
            xs = self.simu_xs_multi(weight)
            A = self.A_unit * weight
            net_arguments = (self.index_i, self.index_j, weight * self.A_interaction, self.cum_index)
        else:
            xs, A, net_arguments, group_index = self.simu_xs_group(weight, m)
        initial_condition  = xs - 0.01
        t = np.arange(0, 200, 0.001)
        dyn_dif = 1
        delta_delay = delay2 - delay1
        result = dict()
        while delta_delay > criteria_delay:
            print(delay1, delay2)
            for delay_i in [delay1, delay2]:
                if delay_i not in result:
                    dyn_all_i = ddeint_Cheng(self.dynamics_multi_delay, initial_condition, t, *(delay_i, self.arguments, net_arguments))[-1000:]
                    diff_i = np.max(np.max(dyn_all_i, 0) - np.min(dyn_all_i, 0))
                    result[delay_i] = diff_i
            if result[delay1] < criteria_dyn and (result[delay2] > criteria_dyn or np.isnan(result[delay2])):
                delay1 = np.round(delay1 + delta_delay/2, 10)
            elif result[delay1] > criteria_dyn or np.isnan(result[delay1]):
                delay2 = np.round(delay1, 10)
                delay1 = np.round(delay1 - delta_delay, 10)
            delta_delay = delay2 - delay1 
        delay_m = pd.DataFrame(np.array([m, delay1]).reshape(1, 2) )
        delay_m.to_csv(save_file, index=None, header=None, mode='a')
        return delay_m

    def tau_mlist(self, weight, save_file, method_type):
        """TODO: Docstring for tau_eigen_mlist.

        :weight: TODO
        :returns: TODO

        """
        for m in self.m_list:
            if method_type == 'eigen':
                self.tau_eigen(weight, m, save_file)
            else:
                self.tau_evolution(weight, m, save_file)

    def tau_parallel(self, cpu_number, method_type):
        """TODO: Docstring for tau_evolution_parallel.

        :network_type: TODO
        :N: TODO
        :beta: TODO
        :: TODO
        :returns: TODO

        """

        if method_type == 'eigen':
            des = '../data/' + '/tau/' + self.dynamics + '/' + self.network_type + '/eigen/' 
        else:
            des = '../data/' + '/tau/' + self.dynamics + '/' + self.network_type + '/evolution/' 
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.tau_mlist, [(weight, des + f'N={self.N}_d={self.d}_seed={self.seed}_weight={weight}.csv', method_type) for weight in self.weight_list]).get()
        p.close()
        p.join()
        return None




B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

cpu_number = 4
N = 1000
network_type = 'SF'
d = [2.5, 999, 3]
seed = [0, 0]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
weight_list = np.round(np.arange(0.05, 1.01, 0.05), 5)
attractor_value = 5
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)
delay1 = 0.1
delay2 = 1

seed_list = [[i, i] for i in range(10)]
if __name__ == "__main__":
    for seed in seed_list:
        ts = Tau_Solution(network_type, N, d, seed, m_list, dynamics, weight_list, arguments, attractor_value, tau_list, nu_list, delay1, delay2)
    
        ts.tau_parallel(cpu_number, 'evolution')
        ts.tau_parallel(cpu_number, 'eigen')


