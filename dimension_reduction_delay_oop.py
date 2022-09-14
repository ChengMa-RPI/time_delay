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
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def genereg_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * xd
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt

def CW_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    a, b = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - xd
    sum_g = A_interaction / (1 + np.exp(a - b * x[index_j]))
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def test_odeint_ddeint(weight, m):
    """TODO: Docstring for test_odeint.

    :arg1: TODO
    :returns: TODO

    """

    dynamics = 'mutual'
    arguments = (B, C, D, E, H, K_mutual)
    ts = Tau_Solution(network_type, N, d, seed, m_list, dynamics, weight_list, arguments, attractor_value, tau_list, nu_list, delay1, delay2)
    if m == ts.N_actual:
        xs = ts.simu_xs_multi(weight)
        A = ts.A_unit * weight
        net_arguments = (ts.index_i, ts.index_j, weight * ts.A_interaction, ts.cum_index)
    else:
        xs, A, net_arguments, group_index = ts.simu_xs_group(weight, m)

    initial_condition  = xs - 0.01
    dt = 0.001
    t = np.arange(0, 100, dt)
    delay = 0.2
    t2 = time.time()
    xd_ddeint = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, ts.arguments, net_arguments))
    t3 = time.time()
    t_ddeint = t3 - t2
    return xd_ddeint, t_ddeint




class Tau_Solution():
    def __init__(self, network_type, N, d, seed, m_list, dynamics, weight_list, arguments, attractor_value, tau_list, nu_list, delay1, delay2, criteria_delay=1e-3, criteria_dyn=1e-3):
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
        self.criteria_delay = criteria_delay
        self.criteria_dyn = criteria_dyn


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
        xs_T = xs.reshape(len(xs), 1)
        if self.dynamics == 'mutual':
            B, C, D, E, H, K = self.arguments
            fx = (1-xs/K) * (2*xs/C-1)
            fxt = -xs/K*(xs/C-1)

            "A_ij: interaction from j to i, should be transposed to A_ji for directed networks (dimension reduction)"
            denominator = D + E * xs + H * xs_T
            gx_i = np.sum(A_T * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
            gx_j = A_T * (xs/denominator - H * xs * xs_T/denominator ** 2 )
        elif self.dynamics == 'genereg':
            B, = self.arguments
            fx = 0
            fxt = -B * np.ones(len(A_T))
            gx_i = 0
            gx_j = A_T * (2 * xs / (xs**2+1)**2) 

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

    def tau_eigen(self, weight, m):
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
            else:
                tau_sol.append(0)  # fake data no solution as 0
        tau_sol = np.array(tau_sol)
        tau_critical = np.min(tau_sol[tau_sol>0])
        tau_m = pd.DataFrame(np.array([m, tau_critical]).reshape(1, 2) )
        return tau_m, tau_sol

    def tau_evolution(self, weight, m):
        """TODO: Docstring for tau_evolution.
         
        :returns: TODO
        
        """
        criteria_dyn = self.criteria_dyn
        criteria_delay = self.criteria_delay
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
                delay1 = max(1e-3, np.round(delay1 - delta_delay, 10))
            elif result[delay2] < criteria_dyn:
                delay1 = delay2
                delay2 = delay2 * 2

            delta_delay = delay2 - delay1 
        delay_m = pd.DataFrame(np.array([m, delay1]).reshape(1, 2) )
        return delay_m

    def tau_mlist(self, weight, save_file, method_type):
        """TODO: Docstring for tau_eigen_mlist.

        :weight: TODO
        :returns: TODO

        """
        tau_list = self.tau_list
        nu_list = self.nu_list
        initial_conditions = np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose()
        filename = save_file
        for m in self.m_list:
            if method_type == 'eigen':
                tau_m, _ = self.tau_eigen(weight, m)
            elif method_type == 'evolution':
                self.tau_evolution_refine(weight, m)
                tau_m = self.tau_evolution(weight, m, save_file)
            else:
                _, tau_sol = self.tau_eigen(weight, m)
                tau_m = pd.DataFrame(np.hstack(( initial_conditions, tau_sol.reshape(len(tau_sol), 1) )) )
                filename = save_file + f'_m={m}.csv'
            tau_m.to_csv(filename, index=None, header=None, mode='a')

    def tau_evolution_refine(self, weight, m):
        """TODO: Docstring for tau_evolution_refine.

        :arg1: TODO
        :returns: TODO

        """
        filename = '../data/tau/' + self.dynamics + '/' + self.network_type + '/' + 'evolution_gross' + f'/N={self.N}_d={self.d}_seed={self.seed}_weight={weight}.csv'
        if os.path.exists(filename):
            data = np.array(pd.read_csv(filename, header=None))
            m_list, tau_list = data.transpose()
            tau_c = tau_list[np.argmin(np.abs(m_list - m))]
            self.delay1 = tau_c 
            self.delay2 = tau_c + 0.01

    def tau_evolution_save(self, weight, m, delay_list, t, interval, des):
        """TODO: Docstring for tau_evolution_refine.

        :arg1: TODO
        :returns: TODO

        """
        if m == self.N_actual:
            xs = self.simu_xs_multi(weight)
            A = self.A_unit * weight
            net_arguments = (self.index_i, self.index_j, weight * self.A_interaction, self.cum_index)
        else:
            xs, A, net_arguments, group_index = self.simu_xs_group(weight, m)
        initial_condition  = xs - 0.01
        for delay in delay_list:
            des_file = des + f'm={m}_d={self.d}_seed={self.seed}_weight={weight}_delay={delay}.csv'
            if not os.path.exists(des_file):
                dyn_all = ddeint_Cheng(self.dynamics_multi_delay, initial_condition, t, *(delay, self.arguments, net_arguments))[::interval]
                df = pd.DataFrame(np.hstack((t[::interval].reshape(len(t[::interval]), 1), dyn_all) ))
                df.to_csv(des_file, header=None, index=None)
        return None

    def tau_data_parallel(self, m_list, weight, interval=1, delay_list=None):
        """TODO: Docstring for tau_eigen_mlist.

        :weight: TODO
        :returns: TODO

        """
        if delay_list == None:
            filename = '../data/tau/' + self.dynamics + '/' + self.network_type + '/' + 'evolution' + f'/N={self.N}_d={self.d}_seed={self.seed}_weight={weight}.csv'
            if os.path.exists(filename):
                data = np.array(pd.read_csv(filename, header=None))
                m_all, tau_c = data.transpose()
                delay_lists = []
                for m in m_list:
                    tau_i = round(tau_c[np.abs(m_all - m) <1e-2][0], 2)
                    delay_list = np.round([tau_i + i for i in [-0.02, -0.01, 0, 0.01, 0.02]], 2)
                    delay_lists.append(delay_list)
            else:
                print("there is no critical delay calculated, please do this first!")
                return None
        else:
            delay_lists = [delay_list] * len(m_list)
        dt = 0.001
        if interval == 1000:
            t = np.arange(0, 200, dt)
            des = '../data/tau/' + self.dynamics + '/' + self.network_type + '/evolution_data/'
        elif interval == 1:
            t = np.arange(0, 50, dt)
            des = '../data/tau/' + self.dynamics + '/' + self.network_type + '/evolution_detail_data/'
        else:
            print('check the interval')
            return None
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.tau_evolution_save, [(weight, m, delay_list, t, interval, des) for m, delay_list in zip(m_list, delay_lists)]).get()
        p.close()
        p.join()
        return None

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
        elif method_type == 'evolution':
            des = '../data/' + '/tau/' + self.dynamics + '/' + self.network_type + '/evolution/' 
        else:
            des = '../data/' + '/tau/' + self.dynamics + '/' + self.network_type + '/tau_initial_condition/' 
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.tau_mlist, [(weight, des + f'N={self.N}_d={self.d}_seed={self.seed}_weight={weight}', method_type) for weight in self.weight_list]).get()
        p.close()
        p.join()
        return None



B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1
B_gene = 1 
a = 5
b = 1

cpu_number = 4
N = 1000
network_type = 'SF'
d = [2.5, 999, 3]
seed = [0, 0]
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) ).tolist() + [N]

dynamics = 'CW'
arguments = (a, b)

"genereg"
dynamics = 'genereg'
arguments = (B_gene, )
weight_list = np.round(np.arange(0.05, 1.01, 0.05), 5)
attractor_value = 10
tau_list = np.arange(1, 2, 0.5)
nu_list = np.arange(0.1, 1, 0.2)
delay1 = 1
delay2 = 2

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
weight_list = np.round(np.arange(0.1, 1.01, 0.1), 5)
attractor_value = 5
tau_list = np.arange(0.01, 0.9, 0.01)
nu_list = np.arange(0.1, 10, 0.1)
delay1 = 0.1
delay2 = 1



seed_list = [[i, i] for i in range(1, 10, 1)]
if __name__ == "__main__":
    for seed in seed_list:
        m_list = np.unique(np.array(np.round([(2**1) ** i for i in range(10)], 0), int) ).tolist() + [N]
        ts = Tau_Solution(network_type, N, d, seed, m_list, dynamics, weight_list, arguments, attractor_value, tau_list, nu_list, delay1, delay2)
    
        #ts.tau_parallel(cpu_number, 'evolution')
        #ts.tau_parallel(cpu_number, 'eigen')
        ts.tau_parallel(cpu_number, 'eigen_all')
        weight = 0.1
        m_evo = np.unique(np.array(np.round([(2**1) ** i for i in range(10)], 0), int) ).tolist() + [N]
        m_evo = [16]
        delay_list_list = [[0.19], [0.22], [0.25], [0.28]]
        for delay_list in delay_list_list:
            #ts.tau_data_parallel(m_evo, weight, interval=1, delay_list=delay_list)
            pass


