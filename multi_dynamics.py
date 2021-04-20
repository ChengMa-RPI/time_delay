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
from scipy.signal import find_peaks
from jitcdde import jitcdde
from jitcdde import y as jy
from jitcdde import t as jt

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

cpu_number = 4
B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

fs = 22
ticksize = 16
legendsize = 14
alpha = 0.8
lw = 2


class Net_Dyn:
    def __init__(self, network_type, N, beta, betaeffect, seed_list, d, dynamics, attractor_value, arguments, tau_list, nu_list):
        """TODO: Docstring for __init__.

        :network_type: TODO
        :N: TODO
        :d: TODO
        :dynamics: TODO
        :parameters: TODO
        :returns: TODO

        """
        self.network_type = network_type
        self.N = N 
        self.beta= beta
        self.betaeffect = betaeffect
        self.seed_list = seed_list
        self.d = d
        self.dynamics = dynamics
        self.attractor_value = attractor_value
        self.arguments = arguments
        self.tau_list = tau_list
        self.nu_list = nu_list

    def multi_stable(self, seed):
        """TODO: Docstring for stable_network.

        :seed: TODO
        :returns: TODO

        """
        network_type, N, beta, betaeffect, d, dynamics, attractor_value, arguments  = self.network_type, self.N, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments  

        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        N_actual = np.size(A, 0)
        net_arguments = (index_i, index_j, A_interaction, cum_index)
        self.A = A
        self.N_actual = N_actual
        self.net_arguments = net_arguments
        initial_condition = np.ones((N_actual)) * attractor_value
        t = np.arange(0, 1000, 0.01)
        if dynamics == 'mutual':
            xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'harvest':
            xs = odeint(harvest_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'genereg':
            xs = odeint(genereg_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'SIS':
            xs = odeint(SIS_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'BDP':
            xs = odeint(BDP_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'PPI':
            xs = odeint(PPI_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        elif dynamics == 'CW':
            xs = odeint(CW_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        return xs

    def single_stable(self, beta):
        """TODO: Docstring for stable_network.

        :seed: TODO
        :returns: TODO

        """
        initial_condition = self.attractor_value 
        arguments = self.arguments
        dynamics =self.dynamics
        t = np.arange(0, 1000, 0.01)
        if dynamics == 'mutual':
            xs = odeint(mutual_single, initial_condition, t, args=(beta, arguments))[-1]
        elif dynamics == 'harvest':
            xs = odeint(harvest_single, initial_condition, t, args=(arguments,))[-1]
        elif dynamics == 'genereg':
            xs = odeint(genereg_single, initial_condition, t, args=(beta, arguments))[-1]
        elif dynamics == 'SIS':
            # xs = odeint(SIS_single, initial_condition, t, args=(beta, arguments))[-1]
            B, = arguments
            xs = xs = 1 - B/beta
        elif dynamics == 'BDP':
            # xs = odeint(BDP_single, initial_condition, t, args=(beta, arguments))[-1]
            B, = arguments
            xs = beta / B
        elif dynamics == 'PPI':
            xs = odeint(PPI_single, initial_condition, t, args=(beta, arguments))[-1]
        elif dynamics == 'CW':
            xs = odeint(CW_single, initial_condition, t, args=(beta, arguments))[-1]
        return xs

    def save_data(self, des_file, data, column_name):
        
        """TODO: Docstring for save_data.
        :returns: TODO

        """

        "save data"
        if not os.path.exists(des_file):
            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')

        return None

    def eigen_solution(self, seed):
        """TODO: Docstring for eigen_fg.

        :dynamics: TODO
        :xs: TODO
        :arguments: TODO
        :returns: TODO

        """

        xs = self.multi_stable(seed)
        network_type, N, N_actual, beta, betaeffect, d, dynamics, attractor_value, arguments, A, tau_list, nu_list = self.network_type, self.N, self.N_actual, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments, self.A, self.tau_list, self.nu_list
        des = '../data/' + dynamics + '/' + network_type + '/tau_evo/'
        if betaeffect:
            des_evo = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_tau_evolution.csv'
        else:
            des_evo = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_tau_evolution.csv'
        data_evo = np.array(pd.read_csv(des_evo, header=None), float)
        tau_evo = data_evo[:, -1][data_evo[:, 0] == seed][0]
        tau_list = np.linspace(tau_evo* 0.8, tau_evo* 1.2, 3)
        
        if dynamics == 'mutual':
            B, C, D, E, H, K = arguments
            fx = (1-xs/K) * (2*xs/C-1)
            fxt = -xs/K*(xs/C-1)
            xs_T = xs.reshape(len(xs), 1)
            denominator = D + E * xs + H * xs_T
            "A should be transposed to A_ji"
            gx_i = np.sum(A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
            gx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
        elif dynamics == 'harvest':
            r, K, c = arguments
            fx = r * (1-xs/K) - 2 * c * xs / (xs**2+1)**2
            fxt = -r * xs  / K
            gx_i = -np.sum(A, 0)
            gx_j = A 
        elif dynamics == 'genereg':
            B, = arguments
            fx = 0
            fxt = -B * np.ones(N_actual)
            gx_i = 0
            gx_j = A * (2 * xs / (xs**2+1)**2) 
        elif dynamics == 'SIS':
            B, = arguments
            fx = 0
            fxt = -B 
            gx_i = - np.sum(A * xs, 1) 
            gx_j = A * (1-xs.reshape(len(xs), 1)) 
        elif dynamics == 'BDP':
            B, = arguments
            fx = 0
            fxt = -B * 2 * xs
            gx_i = 0
            gx_j = A 
        elif dynamics == 'PPI':
            B, F = arguments
            fx = 0
            fxt = -B 
            gx_i = - np.sum(A * xs, 1)
            gx_j = - A * xs.reshape(len(xs), 1) 
        elif dynamics == 'CW':
            a, b = arguments
            fx = 0
            fxt = -1 * np.ones(N_actual) 
            gx_i = 0
            gx_j = A * b * np.exp(a - b * xs) / (1 + np.exp(a - b * xs)) ** 2


        "compute eigenvalues"
        tau_sol = []
        for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
            tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, gx_i, gx_j))
            "check the solution given by fsolve built-in function."
            eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, gx_i, gx_j)
            if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                #print(tau_solution, nu_solution)
                tau_sol.append(tau_solution)
        tau_sol = np.array(tau_sol)
        tau_critical = np.min(tau_sol[tau_sol>0])
        "save data"
        data = np.hstack((seed, tau_critical))
        des = '../data/' + dynamics + '/' + network_type + '/tau_multi/'
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'

        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau_critical)
        return tau_critical

    def eigen_parallel(self, function, delay1=0, delay2=10):
        """TODO: Docstring for tau_critical.

        :arg1: TODO
        :returns: TODO

        """
        p = mp.Pool(cpu_number)
        if function == 'eigen':
            p.starmap_async(self.eigen_solution, [(seed, ) for seed in self.seed_list]).get()
        elif function == 'decouple_two':
            p.starmap_async(self.tau_decouple_two, [(seed, ) for seed in self.seed_list]).get()
        elif function == 'tau_evo':
            p.starmap_async(self.tau_evolution, [(seed, delay1, delay2) for seed in self.seed_list]).get()
        elif function == 'tau_RK':
            p.starmap_async(self.tau_RK, [(seed, delay1, delay2) for seed in self.seed_list]).get()
        elif function == 'tau_eff':
            p.starmap_async(self.tau_decouple_eff, [(seed) for seed in self.seed_list]).get()
        p.close()
        p.join()
        return None

    def tau_1D(self):
        """TODO: Docstring for tau_1D.

        :arg1: TODO
        :returns: TODO

        """
        beta = self.beta
        dynamics, arguments = self.dynamics, self.arguments
        xs = self.single_stable(beta)
        if dynamics == 'mutual':
            B, C, D, E, H, K = arguments
            P =  -(1 -xs / K) * (2 * xs / C-1)- (2 * beta * xs)/(D + E * xs + H * xs) + (beta * (E+H) * xs**2)/(D + (E+H) * xs)**2 
            Q = xs/K*(xs/C-1)
        
        elif dynamics == 'harvest':
            r, K, c = arguments
            P = -r * (1-xs/K) + 2 * c * xs / (xs**2+1)**2 
            Q = r * xs / K
        elif dynamics == 'genereg':
            B, = arguments
            P = - 2 * beta * xs / (xs**2 +1)**2
            Q = B
        elif dynamics == 'SIS':
            B, = arguments
            P = beta * (2 * xs - 1)
            Q = B
        elif dynamics == 'BDP':
            B, = arguments
            P = -beta 
            Q = B * 2 * xs
        elif dynamics == 'PPI':
            B, F = arguments
            P = beta * 2 * xs 
            Q = B 
        elif dynamics == 'CW':
            a, b = arguments
            P = - beta * b * np.exp(a - b * xs) / (1 + np.exp(a - b * xs)) ** 2
            Q = 1 



        if abs(P/Q)<=1:
            tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
            nu = np.arccos(-P/Q)/tau
        else:
            tau = -1
        des = '../data/' + dynamics + '/'
        if not os.path.exists(des):
            os.makedirs(des)

        des_file = des + 'tau_1D.csv'

        data = np.hstack((beta, tau))
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
        return None

    def tau_decouple_eff(self, seed):
        """TODO: Docstring for tau_kmax.

        :network_type: TODO
        :N: TODO
        :beta: TODO
        :betaeffect: TODO
        :returns: TODO

        """

        network_type, N, beta, betaeffect, d, dynamics, attractor_value, arguments, seed_list  = self.network_type, self.N, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments, self.seed_list

        des = '../data/' + dynamics + '/' + network_type + '/tau_multi/' 
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_eff_logistic.csv'


        t = np.arange(0, 1000, 0.01)
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        beta_eff, _ = betaspace(A, [0])
        wk = np.sum(A, 0)
        index_list = np.argsort(wk)
        tau_list = np.ones(len(index_list)) * (-1)
        for index, i in zip(index_list, range(len(index_list))):
            w = np.sum(A[index])
            if dynamics == 'mutual':
                B, C, D, E, H, K = arguments
                xs = odeint(mutual_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P =  - (w * x2)/(D + E * x1 + H * x2) + (w * E * x1 * x2)/(D + E * x1 + H * x2)**2 -(1-x1/K) * (2*x1/C-1)
                Q = x1/K*(x1/C-1)

            elif dynamics == 'harvest':
                r, K, c = arguments
                xs = odeint(harvest_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = -r * (1-x1/K) + 2 * c * x1 / (x1**2+1)**2 + w
                Q = r * x1 / K

            elif dynamics == 'genereg':
                B, = arguments
                xs = odeint(genereg_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = 0
                Q = B

            elif dynamics == 'SIS':
                B, = arguments
                xs = odeint(SIS_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = w * x2
                Q = B

            elif dynamics == 'BDP':
                B, = arguments
                xs = odeint(BDP_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = 0
                Q = B * 2 * x1

            elif dynamics == 'PPI':
                B, F = arguments
                xs = odeint(PPI_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = w * x2
                Q = B

            elif dynamics == 'CW':
                a, b = arguments
                xs = odeint(CW_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                x1, x2 = xs
                P = 0
                Q = 1

            if abs(P/Q)<=1:
                tau_list[i] = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) 
                #print(tau_list[i])
            else:
                print(xs, P, Q)


        tau = np.min(tau_list[tau_list>0])
        tau_index = index_list[np.where(tau == tau_list)[0][0]]
        data = np.hstack((seed, wk.max(), tau, wk[tau_index], np.where(np.sort(wk)[::-1]==wk[tau_index])[0][-1]))
     
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')

        print(seed, tau)
        return None

    def tau_decouple_two(self, seed):
        """TODO: Docstring for tau_kmax.

        :returns: TODO

        """
        network_type, N, beta, d, dynamics, attractor_value, arguments, tau_list, nu_list, betaeffect  = self.network_type, self.N, self.beta, self.d, self.dynamics, self.attractor_value, self.arguments, self.tau_list, self.nu_list, self.betaeffect
        des = '../data/' + dynamics + '/' + network_type + '/tau_multi/' 
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_two_logistic.csv'
        else:
            des_file = des + f'N={N}_d=' + str(d) + '_wt=' + str(beta) + '_decouple_two_logistic.csv'


        t = np.arange(0, 1000, 0.01)

        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        beta_eff, _ = betaspace(A, [0])
        wk = np.sum(A, 0)
        index_list = np.argsort(wk)[-10:]
        index_list = np.argsort(wk)
        tau_individual = []
        tau_index = []
        for index in index_list:
            w = np.sum(A[index])
            if dynamics == 'mutual':
                B, C, D, E, H, K = arguments
                xs = odeint(mutual_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = (1-xs/K) * (2*xs/C-1)
                fxt = -xs/K*(xs/C-1)
                g11 = w * (xs[1] /(D + E*xs[0] + H*xs[1]) - E*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
                g12 = w * (xs[0] /(D + E*xs[0] + H*xs[1]) - H*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
                g21 = 0
                g22 = beta_eff * (2*xs[1]/(D+E*xs[1]+H*xs[1]) - xs[1]**2 * (E+H) /(D+E*xs[1]+H*xs[1])**2)

            elif dynamics == 'harvest':
                r, K, c = arguments
                xs = odeint(harvest_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = r * (1-xs/K) - 2 * c * xs / (xs**2+1)**2
                fxt = -r * xs  / K
                g11 = -w
                g12 = w 
                g21 = 0
                g22 = 0

            elif dynamics == 'genereg':
                B, = arguments
                xs = odeint(genereg_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * np.ones(2)
                g11 = 0
                g12 = w * 2 * xs[1] / (xs[1]**2 + 1)**2
                g21 = 0
                g22 = beta * 2 * xs[1] / (xs[1]**2 + 1)**2

            elif dynamics == 'SIS':
                B, = arguments
                xs = odeint(SIS_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * np.ones(2)
                g11 = - w * xs[1]
                g12 = w  * (1 - xs[0])
                g21 = 0
                g22 = beta * (1 - 2 * xs[1])

            elif dynamics == 'BDP':
                B, = arguments
                xs = odeint(BDP_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * 2 * xs
                g11 = 0
                g12 = w 
                g21 = 0
                g22 = beta 

            elif dynamics == 'PPI':
                B, F = arguments
                xs = odeint(PPI_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B  * np.ones(2)
                g11 = - w * xs[1]
                g12 = - w * xs[0] 
                g21 = 0
                g22 = - beta * 2 * xs[1]  

            elif dynamics == 'CW':
                a, b = arguments
                xs = odeint(CW_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -1 * np.ones(2)
                g11 = 0
                g12 = w * b * np.exp(a - b * xs[1]) / (1 + np.exp(a - b * xs[1])) ** 2
                g21 = 0
                g22 = beta * b * np.exp(a - b * xs[1]) / (1 + np.exp(a - b * xs[1])) ** 2 

            g_matrix = np.array([[g11, g12], [g21, g22]])

            tau_sol = []
            for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
                tau_solution, nu_solution = fsolve(eigen_two_decouple, initial_condition, args=(fx, fxt, g_matrix))
                eigen_real, eigen_imag = eigen_two_decouple(np.array([tau_solution, nu_solution]), fx, fxt, g_matrix)
                #tau_solution, nu_solution = fsolve(determinant_two_decouple, initial_condition, args=(fx, fxt, g_matrix))
                #eigen_real, eigen_imag = determinant_two_decouple(np.array([tau_solution, nu_solution]), fx, fxt, g_matrix)
                if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                    tau_sol.append(tau_solution)
            tau_sol = np.array(tau_sol)
            if np.size(tau_sol[tau_sol>0]):
                tau = np.min(tau_sol[tau_sol>0])
                tau_individual.append(tau)

        tau = np.min(tau_individual)
        tau_index.append(np.argmin(tau_individual))
        data = np.hstack((seed, wk.max(), tau, tau_index))
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau)

        return None

    def tau_decouple_wk(self, wk_list):
        """TODO: Docstring for tau_kmax.

        :returns: TODO

        """
        attractor_value , beta_eff = self.attractor_value, self.beta
        des = '../data/' + dynamics + '/'  
        if not os.path.exists(des):
            os.makedirs(des)
        des_file = des + 'beta=' + str(beta) + '_wk_logistic.csv'

        t = np.arange(0, 1000, 0.01)
        tau_individual = np.ones(len(wk_list)) * (-1)
        for i, w in zip(range(len(wk_list)) ,wk_list):
            if dynamics == 'mutual':
                B, C, D, E, H, K = arguments
                xs = odeint(mutual_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = (1-xs/K) * (2*xs/C-1)
                fxt = -xs/K*(xs/C-1)
                g11 = w * (xs[1] /(D + E*xs[0] + H*xs[1]) - E*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
                g12 = w * (xs[0] /(D + E*xs[0] + H*xs[1]) - H*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
                g21 = 0
                g22 = beta_eff * (2*xs[1]/(D+E*xs[1]+H*xs[1]) - xs[1]**2 * (E+H) /(D+E*xs[1]+H*xs[1])**2)

            elif dynamics == 'harvest':
                r, K, c = arguments
                xs = odeint(harvest_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = r * (1-xs/K) - 2 * c * xs / (xs**2+1)**2
                fxt = -r * xs  / K
                g11 = -w
                g12 = w 
                g21 = 0
                g22 = 0

            elif dynamics == 'genereg':
                B, = arguments
                xs = odeint(genereg_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * np.ones(2)
                g11 = 0
                g12 = w * 2 * xs[1] / (xs[1]**2 + 1)**2
                g21 = 0
                g22 = beta * 2 * xs[1] / (xs[1]**2 + 1)**2

            elif dynamics == 'SIS':
                B, = arguments
                xs = odeint(SIS_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * np.ones(2)
                g11 = - w * xs[1]
                g12 = w  * (1 - xs[0])
                g21 = 0
                g22 = beta * (1 - 2 * xs[1])

            elif dynamics == 'BDP':
                B, = arguments
                xs = odeint(BDP_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B * 2 * xs
                g11 = 0
                g12 = w 
                g21 = 0
                g22 = beta 

            elif dynamics == 'PPI':
                B, F = arguments
                xs = odeint(PPI_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -B  * np.ones(2)
                g11 = - w * xs[1]
                g12 = - w * xs[0] 
                g21 = 0
                g22 = - beta * 2 * xs[1]  

            elif dynamics == 'CW':
                a, b = arguments
                xs = odeint(CW_decouple_two, np.ones(2) * attractor_value, t, args=(w, beta_eff, arguments))[-1]
                fx = 0
                fxt = -1 * np.ones(2)
                g11 = 0
                g12 = w * b * np.exp(a - b * xs[1]) / (1 + np.exp(a - b * xs[1])) ** 2
                g21 = 0
                g22 = beta * b * np.exp(a - b * xs[1]) / (1 + np.exp(a - b * xs[1])) ** 2 


            g_matrix = np.array([[g11, g12], [g21, g22]])

            tau_sol = []
            for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
                tau_solution, nu_solution = fsolve(eigen_two_decouple, initial_condition, args=(fx, fxt, g_matrix))
                eigen_real, eigen_imag = eigen_two_decouple(np.array([tau_solution, nu_solution]), fx, fxt, g_matrix)
                if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                    tau_sol.append(tau_solution)
            tau_sol = np.array(tau_sol)
            if np.size(tau_sol[tau_sol>0]):
                tau = np.min(tau_sol[tau_sol>0])
                tau_individual[i] = tau
            print(w, tau_individual[i])

        data = np.vstack((wk_list, tau_individual))
        df = pd.DataFrame(data.transpose())
        df.to_csv(des_file, index=None, header=None, mode='a')
     

        return None

    def tau_evolution(self, seed, delay1, delay2, criteria_dyn=1e-3, criteria_delay=5e-3):
        """TODO: Docstring for tau_evolution.

        :network_type: TODO
        :N: TODO
        :beta: TODO
        :seed: TODO
        :d: TODO
        :returns: TODO

        """
        xs = self.multi_stable(seed)
        network_type, N, N_actual, beta, betaeffect, d, dynamics, attractor_value, arguments, A, tau_list, nu_list = self.network_type, self.N, self.N_actual, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments, self.A, self.tau_list, self.nu_list

        if dynamics == 'mutual':
            dynamics_function = mutual_multi_delay
        elif dynamics == 'harvest':
            dynamics_function = harvest_multi_delay
        elif dynamics == 'genereg':
            dynamics_function = genereg_multi_delay
        elif dynamics == 'SIS':
            dynamics_function = SIS_multi_delay
        elif dynamics == 'BDP':
            dynamics_function = BDP_multi_delay
        elif dynamics == 'PPI':
            dynamics_function = PPI_multi_delay
        elif dynamics == 'CW':
            dynamics_function = CW_multi_delay

        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        net_arguments = (index_i, index_j, A_interaction, cum_index)

        initial_condition = xs - 1e-3
        dt = 0.001
        t = np.arange(0, 200, dt)
        index = int(10/dt)
        dyn_dif = 1
        delta_delay = delay2 - delay1
        result = dict()
        while delta_delay > criteria_delay:
            if delay1 not in result:
                dyn_all1 = ddeint_Cheng(dynamics_function, initial_condition, t, *(delay1, arguments, net_arguments))[-index:]
                diff1 = np.max(np.max(dyn_all1, 0) - np.min(dyn_all1, 0))
                result[delay1] = diff1
            if delay2 not in result:
                dyn_all2 = ddeint_Cheng(dynamics_function, initial_condition, t, *(delay2, arguments, net_arguments))[-index:]
                diff2 = np.max(np.max(dyn_all2, 0) - np.min(dyn_all2, 0))
                result[delay2] = diff2
            if result[delay1] < criteria_dyn and (result[delay2] > criteria_dyn or np.isnan(result[delay2])):
                delay1 = np.round(delay1 + delta_delay/2, 10)
            elif result[delay1] > criteria_dyn or np.isnan(result[delay1]):
                delay2 = np.round(delay1, 10)
                delay1 = np.round(delay1 - delta_delay, 10)
            elif result[delay2] < criteria_dyn:
                delay1 = delay2
                delay2 = delay2 * 2
            delta_delay = delay2 - delay1 
        print(seed, delay1, delay2)
        data = np.hstack((seed, delay1))
        des = '../data/' + dynamics + '/' + network_type + '/tau_evo/'
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'N={N}_d={d}_beta={beta}_tau_Euler.csv'
        else:
            des_file = des + f'N={N}_d={d}_wt={beta}_tau_Euler.csv'

        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
        return None

    def tau_RK(self, seed, delay1, delay2, criteria_dyn=1e-3, criteria_delay=5e-3):
        """TODO: Docstring for tau_evolution.

        :network_type: TODO
        :N: TODO
        :beta: TODO
        :seed: TODO
        :d: TODO
        :returns: TODO

        """
        xs = self.multi_stable(seed)
        network_type, N, N_actual, beta, betaeffect, d, dynamics, attractor_value, arguments, A, tau_list, nu_list = self.network_type, self.N, self.N_actual, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments, self.A, self.tau_list, self.nu_list

        if dynamics == 'mutual':
            dynamics_function = mutual_multi_delay
        elif dynamics == 'harvest':
            dynamics_function = harvest_multi_delay
        elif dynamics == 'genereg':
            dynamics_function = genereg_multi_delay
        elif dynamics == 'SIS':
            dynamics_function = SIS_multi_delay
        elif dynamics == 'BDP':
            dynamics_function = BDP_multi_delay
        elif dynamics == 'PPI':
            dynamics_function = PPI_multi_delay
        elif dynamics == 'CW':
            dynamics_function = CW_multi_delay

        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        net_arguments = (index_i, index_j, A_interaction, cum_index)

        initial_condition = xs - 1e-3
        dt = 0.01
        t = np.arange(0, 200, dt)
        index = int(10/dt)
        dyn_dif = 1
        delta_delay = delay2 - delay1
        result = dict()
        while delta_delay > criteria_delay:
            if delay1 < 0:
                print(seed, delay1, delay2)
            if delay1 not in result:
                dyn_all1 = dde_RK45(dynamics_function, initial_condition, t, *(delay1, arguments, net_arguments))[-index:]
                diff1 = max(np.max(np.max(dyn_all1, 0) - np.min(dyn_all1, 0)), np.max(np.abs(dyn_all1[-1]-xs)))
                result[delay1] = diff1
            if delay2 not in result:
                dyn_all2 = dde_RK45(dynamics_function, initial_condition, t, *(delay2, arguments, net_arguments))[-index:]
                diff2 = np.max(np.max(dyn_all2, 0) - np.min(dyn_all2, 0), np.max(np.abs(dyn_all2[-1]-xs)))
                result[delay2] = diff2
            if result[delay1] < criteria_dyn and (result[delay2] > criteria_dyn or np.isnan(result[delay2])):
                delay1 = np.round(delay1 + delta_delay/2, 10)
            elif result[delay1] > criteria_dyn or np.isnan(result[delay1]):
                delay2 = np.round(delay1, 10)
                delay1 = np.round(delay1 - delta_delay, 10)
            elif result[delay2] < criteria_dyn:
                delay1 = delay2
                delay2 = delay2 * 2
            delta_delay = delay2 - delay1 
        print(seed, delay1, delay2)
        data = np.hstack((seed, delay1))
        des = '../data/' + dynamics + '/' + network_type + '/tau_evo/'
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'N={N}_d={d}_beta={beta}_tau_RK.csv'
        else:
            des_file = des + f'N={N}_d={d}_wt={beta}_tau_RK.csv'

        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
        return None




def harvest_single(x, t, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :c: bifurcation parameter 
    :returns: derivative of x 

    """
    r, K, c = arguments
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    return dxdt

def harvest_multi(x, t, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    r, K, c = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt

def harvest_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, (f[int(round((t-d)/dt))] + f[int(round((t-d)/dt)) +1])/2, x0)
    index_i, index_j, A_interaction, cum_index = net_arguments
    r, K, c = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def harvest_single_delay(f, x0, t, dt, d, arguments):
    """TODO: Docstone_delayring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    r, K, c = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1)
    return dxdt

def harvest_decouple_delay(f, x0, t, dt, d, w, x_fix, arguments):
    """TODO: Docstone_delayring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    r, K, c = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1) + w * (x_fix - x)
    return dxdt

def harvest_decouple_two_delay(f, x0, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    r, K, c = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = r * x * (1 - xd/K) - c * x**2 / (x**2 + 1)
    sum_g = np.array([w * (x[1] - x[0]), 0])
    dxdt = sum_f + sum_g
    return dxdt

def harvest_decouple_eff_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    r, K, c = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([r * x[0] * (1 - xd[0]/K) - c * x[0]**2 / (x[0]**2 + 1), r * x[1] * (1 - x[1]/K) - c * x[1]**2 / (x[1]**2 + 1)])
    sum_g = np.array([w * (x[1] - x[0]), 0])
    dxdt = sum_f + sum_g
    return dxdt

def harvest_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    r, K, c = arguments
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = np.array([w * (x[1] - x[0]), 0])
    dxdt = sum_f + sum_g
    return dxdt


def genereg_single(x, t, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    dxdt = -B * x + beta * x**2/(x**2+1)
    return dxdt

def genereg_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden

    dxdt = -B * xd + beta * x**2/(x**2+1)
    return dxdt

def genereg_multi(x, t, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * x 
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

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
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * xd
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])

    return dxdt

def genereg_decouple_two_delay(f, x0, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = -B * xd
    sum_g = np.array([w * (x[1]**2/(x[1]**2+1)), beta * (x[1]**2/(x[1]**2+1))])
    dxdt = sum_f + sum_g
    return dxdt

def genereg_decouple_eff_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([-B * xd[0], -B * x[1]])
    sum_g = np.array([w * (x[1]**2/(x[1]**2+1)), beta * (x[1]**2/(x[1]**2+1))])
    dxdt = sum_f + sum_g
    return dxdt

def genereg_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, = arguments
    sum_f = -B * x
    sum_g = np.array([w * (x[1]**2/(x[1]**2+1)), beta * (x[1]**2/(x[1]**2+1))])
    dxdt = sum_f + sum_g
    return dxdt


def SIS_single(x, t, beta, arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    dxdt = -B * x + beta * x * (1-x)
    return dxdt

def SIS_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = -B * xd + beta * x * (1-x)
    return dxdt

def SIS_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x 
    sum_g = A_interaction * (1-x[index_i]) * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def SIS_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * xd
    sum_g = A_interaction * (1-x[index_i]) * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def SIS_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, = arguments
    sum_f = -B * x
    sum_g = np.array([w * x[1] * (1-x[0]), beta * x[1] * (1-x[1])])
    dxdt = sum_f + sum_g
    return dxdt


def BDP_single(x, t, beta, arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    dxdt = -B * x ** 2 + beta * x 
    return dxdt

def BDP_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = -B * xd ** 2 + beta * x 
    return dxdt

def BDP_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x ** 2 
    sum_g = A_interaction * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def BDP_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - B * xd ** 2
    sum_g = A_interaction * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def BDP_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, = arguments
    sum_f = -B * x ** 2
    sum_g = np.array([w * x[1], beta * x[1]])
    dxdt = sum_f + sum_g
    return dxdt

def BDP_decouple_eff_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([-B * xd[0] ** 2, -B * x[1] ** 2])
    sum_g = np.array([w * x[1], beta * x[1]])
    dxdt = sum_f + sum_g
    return dxdt

def BDP_decouple_eff(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([-B * x[0] ** 2, -B * x[1] ** 2])
    sum_g = np.array([w * x[1], beta * x[1]])
    dxdt = sum_f + sum_g
    return dxdt


def PPI_single(x, t, beta, arguments):
    """Protein protein interaction model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, F = arguments
    dxdt = F - B * x - beta * x ** 2 
    return dxdt

def PPI_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, F = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = F - B * xd - beta * x ** 2
    return dxdt

def PPI_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    B, F = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = F - B * x
    sum_g = - A_interaction * x[index_i] * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def PPI_multi_delay(f, x0, x, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    index = int(round((t-d)/dt))
    xd = np.where(t>d, f[index], x0)
    B, F = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = F - B * xd
    sum_g = - A_interaction * x[index_i] * x[index_j]
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def PPI_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, F = arguments
    sum_f = F - B * x
    sum_g = np.array([- w * x[0] * x[1], - beta * x[1] * x[1]])
    dxdt = sum_f + sum_g
    return dxdt

def PPI_decouple_two_delay(f, x0, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    x = f[int(t/dt)]
    B, F = arguments
    sum_f = F - B * xd
    sum_g = np.array([- w * x[0] * x[1], - beta * x[1] * x[1]])
    dxdt = sum_f + sum_g
    return dxdt

def PPI_decouple_eff_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, F = arguments
    sum_f = np.array([F - B * xd[0], F - B * x[1]])
    sum_g = np.array([- w * x[0] * x[1], - beta * x[1] * x[1]])
    dxdt = sum_f + sum_g
    return dxdt

def PPI_decouple_eff(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, F = arguments
    sum_f = np.array([F - B * x[0], F - B * x[1]])
    sum_g = np.array([- w * x[0] * x[1], - beta * x[1] * x[1]])
    dxdt = sum_f + sum_g
    return dxdt


def CW_single(x, t, beta, arguments):
    """Protein protein interaction model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :returns: TODO

    """
    a, b = arguments
    dxdt = - x + beta / (1 + np.exp(a - b * x))
    return dxdt

def CW_single_delay(f, x0, t, dt, d, beta, arguments):
    """TODO: Docstring for genereg_single.

    :x: TODO
    :t: TODO
    :beta: TODO
    :arguments: TODO
    :returns: TODO

    """
    a, b = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = - xd + beta / (1 + np.exp(a - b * x))
    return dxdt

def CW_multi(x, t, arguments, net_arguments):
    """SIS model

    :x: TODO
    :t: TODO
    :arguments: TODO
    :net_arguments: TODO
    :returns: TODO

    """
    a, b = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - x
    sum_g = A_interaction / (1 + np.exp(a - b * x[index_j]))
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
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = - xd
    sum_g = A_interaction / (1 + np.exp(a - b * x[index_j]))
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def CW_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, F = arguments
    sum_f = - x
    sum_g = np.array([w / (1 + np.exp(a - b * x[1])), beta / (1 + np.exp(a - b * x[1]))])
    dxdt = sum_f + sum_g
    return dxdt





def mutual_single(x, t, beta, arguments):
    """original dynamics N species interaction.

    :x: 1 dynamic variable
    :t: time series
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + beta * x**2 / (D + (E+H) * x)
    return dxdt

def mutual_single_delay(f, x0, t, dt, d, beta, arguments):
    """original dynamics N species interaction.

    :x: 1 dynamic variable
    :t: time series
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    dxdt = B + x * (1 - xd/K) * ( x/C - 1) + beta * x**2 / (D + (E+H) * x)
    return dxdt

def mutual_multi(x, t, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

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

def mutual_decouple_two(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = np.array([w * x[0] * x[1] / (D + E * x[0] + H * x[1]), beta * x[1] * x[1] / (D + E * x[1] + H * x[1])])
    dxdt = sum_f + sum_g
    return dxdt

def mutual_decouple_two_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = np.array([w * x[0] * x[1] / (D + E * x[0] + H * x[1]), beta * x[1] * x[1] / (D + E * x[1] + H * x[1])])
    dxdt = sum_f + sum_g
    return dxdt

def mutual_decouple_eff_delay(f, x0, x, t, dt, d, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([B + x[0] * (1 - xd[0]/K) * (x[0]/C - 1), B + x[1] * (1-x[1]/K) * (x[1]/C - 1)])
    sum_g = np.array([w * x[0]*x[1] / (D + E * x[0] +H * x[1]), beta * x[1]**2 / (D + (E+H) * x[1])])
    dxdt = sum_f + sum_g
    return dxdt

def mutual_decouple_eff(x, t, w, beta, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    #x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = np.array([B + x[0] * (1 - x[0]/K) * (x[0]/C - 1), B + x[1] * (1-x[1]/K) * (x[1]/C - 1)])
    sum_g = np.array([w * x[0]*x[1] / (D + E * x[0] +H * x[1]), beta * x[1]**2 / (D + (E+H) * x[1])])
    dxdt = sum_f + sum_g
    return dxdt



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

def eigen_two_decouple(x, fx, fxt, g_matrix):
    """TODO: Docstring for eigen_two_decouple.

    :arg1: TODO
    :returns: TODO

    """
    imag = 1j
    tau, nu = x
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag)) - g_matrix 
    eigenvalue, eigenvector = np.linalg.eig(M)
    zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
    return np.array([np.real(zeropoint), np.imag(zeropoint)])

def determinant_two_decouple(x, fx, fxt, g_matrix):
    """TODO: Docstring for eigen_two_decouple.

    :arg1: TODO
    :returns: TODO

    """
    imag = 1j
    tau, nu = x
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag)) - g_matrix 
    determinant = np.linalg.det(M)
    return np.array([np.real(determinant), np.imag(determinant)])


def mutual_1D_delay(f, x0, t, dt, d1, d2, d3, beta, arguments):
    """TODO: Docstring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd1 = np.where(t>d1, f[int((t-d1)/dt)], x0)
    xd2 = np.where(t>d2, f[int((t-d2)/dt)], x0)
    xd3 = np.where(t>d3, f[int((t-d3)/dt)], x0)

    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden

    dxdt = B + x * (1 - xd1/K) * ( x/C - 1) + beta * x**2/(D+E*x+H*x)
    return dxdt

def mutual_single_one_delay(f, x0, t, dt, d, w, x_fix, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = w * x * x_fix / (D + E * x + H * x_fix)
    dxdt = sum_f + sum_g
    return dxdt

def mutual_single_one(x, t, w, x_fix, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + w * x * x_fix / (D + E * x + H * x_fix)
    return dxdt


def evolution(network_type, N, beta, betaeffect, seed, arguments, d1, d2, d3, d=None):
    """TODO: not useful.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N = np.size(A, 0)
    initial_condition = np.ones(N) * 5 
    t = np.arange(0, 50, 0.001)
    t1 = time.time()
    dyn_all = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
    t2 = time.time()
    print(t2-t1)
    plt.plot(t[5000:], dyn_all[5000:, np.argmax(np.sum(A>0, 0))], alpha = alpha)
    #plt.plot(t, dyn_all, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$ x $', fontsize =fs)
    #plt.show()
    return dyn_all

def tau_evolution(network_type, N, beta, betaeffect, seed, d, arguments, delay1, delay2, criteria_delay, criteria_dyn):
    """TODO: Docstring for tau_evolution.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N = np.size(A, 0)
    initial_condition = np.ones(N) * 5 
    t = np.arange(0, 200, 0.001)
    dyn_dif = 1
    delta_delay = delay2 - delay1
    result = dict()
    while delta_delay > criteria_delay:
        if delay1 not in result:
            dyn_all1 = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay1, 0, 0, N, index_i, index_j, A_interaction, cum_index, arguments))[-10000:]
            diff1 = np.max(np.max(dyn_all1, 0) - np.min(dyn_all1, 0))
            result[delay1] = diff1
        if delay2 not in result:
            dyn_all2 = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay2, 0, 0, N, index_i, index_j, A_interaction, cum_index, arguments))[-10000:]
            diff2 = np.max(np.max(dyn_all2, 0) - np.min(dyn_all2, 0))
            result[delay2] = diff2
        if result[delay1] < criteria_dyn and (result[delay2] > criteria_dyn or np.isnan(result[delay2])):
            delay1 = np.round(delay1 + delta_delay/2, 10)
        elif result[delay1] > criteria_dyn or np.isnan(result[delay1]):
            delay2 = np.round(delay1, 10)
            delay1 = np.round(delay1 - delta_delay, 10)
        delta_delay = delay2 - delay1 
    print(seed, delay1, delay2)
    data = np.hstack((seed, delay1))
 
    column_name = [f'seed{i}' for i in range(np.size(seed))]
    column_name.extend([ str(beta) for beta in beta_set])

    des = '../data/' + dynamics + '/' + network_type + '/tau_evo/'
    if not os.path.exists(des):
        os.makedirs(des)

    if betaeffect:
        des_file = des + network_type + f'N={N}_d={d}_beta={beta}_evolution.csv'
    else:
        des_file = des + network_type + f'N={N}_d={d}_wt={beta}_evolution.csv'

    if not os.path.exists(des_file):
        df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
        df.to_csv(des_file, index=None, mode='a')
    else:
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    print('good')

    return delay2

def tau_evolution_parallel(network_type, N, beta, betaeffect, seed_list, d, arguments, delay1, delay2, criteria_delay, criteria_dyn):
    """TODO: Docstring for tau_evolution_parallel.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """

    p = mp.Pool(cpu_number)
    p.starmap_async(tau_evolution, [(network_type, N, beta, betaeffect, seed, d, arguments, delay1, delay2, criteria_delay, criteria_dyn) for seed in seed_list]).get()
    p.close()
    p.join()

    return None

def compare_evo_eigenvalue(network_type, N, beta, seed,d ):
    """TODO: Docstring for compare_evo_eigenvalue.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    data_eigen = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_logistic.csv').iloc[:, :])
    seed_eigen = np.array(data_eigen[:, 0], int)
    tau_eigen = data_eigen[:, -1][np.argsort(seed_eigen)]
    data_evo = np.array(pd.read_csv('../data/'+ network_type + f'_N={N}_d=' + str(d) + '_evolution.csv').iloc[:, :])
    seed_evo = np.array(data_evo[:, 0], int)
    tau_evo = data_evo[:, -1]
    tau_eigen = tau_eigen[seed_evo]
    plt.plot(tau_eigen, tau_evo, 'o')
    plt.plot(np.arange(0.2, 0.4, 0.1), np.arange(0.2, 0.4, 0.1))

def evolution_analysis(network_type, dynamics_multi, dynamics_multi_delay, dynamics_decouple_two_delay, arguments, N, beta, betaeffect, seed, d, delay):
    """TODO: Docstring for evolution_oscillation.

    :arg1: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    beta_eff, _ = betaspace(A, [0])
    degree= np.sum(A>0, 0)
    N = np.size(A, 0)
    dt = 0.01
    t = np.arange(0, 200, dt)
    initial_condition = np.ones(N) * 5
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    xs_high = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    t1 = time.time()
    deviation = 0.0001
    dyn_multi = ddeint_Cheng(dynamics_multi_delay, xs_high - deviation, t, *(delay, arguments, net_arguments))
    t2 = time.time()
    plot_diff(dyn_multi, xs_high, dt, 'tab:red', 'multi-delay')
    diff = np.sum(np.abs(dyn_multi - xs_high), 0)
    diffmax_index = np.argmax(diff) 
    t3 = time.time()
    w = np.sum(A[diffmax_index])
    xs_decouple = ddeint_Cheng(dynamics_decouple_two_delay, initial_condition[0:2], t, *(0, beta_eff, w, arguments))[-1]
    dyn_decouple = ddeint_Cheng(dynamics_decouple_two_delay, xs_decouple - deviation, t, *(delay, beta_eff, w, arguments))[:, 0]

    plot_diff(dyn_decouple, xs_decouple[0], dt, 'tab:blue', 'decouple-delay')
    #xs_eff = fsolve(mutual_1D, np.mean(initial_condition), args=(0, beta_eff, arguments))
    #xs_eff = np.mean(xs_high)
    #xs_eff = np.mean(np.setdiff1d(xs_high, xs_high[index]))
    #xs_high_max = ddeint_Cheng(one_single_delay, np.array([xs_high[index]]), t, *(0, 0, 0, w, xs_eff, arguments))[-1]
    initial_condition = np.ones(1) * 5
    #initial_condition = xs_high_max - 0.0001

    t4 = time.time()
    #dyn_one = ddeint_Cheng(one_single_delay, initial_condition, t, *(delay, 0, 0, w, xs_eff, arguments))[:, 0]
    t5 = time.time()
    print(t2-t1, t3-t2, t5-t4)
    #plot_diff(dyn_one, xs_high_max, dt, 'tab:green', 'one-component')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$A_{x}$', fontsize =fs)
    plt.legend(frameon=False, fontsize=legendsize, loc='lower left') 
    plt.show()

    return dyn_multi, xs_high

def plot_diff(dyn, xs, dt, color, label):
    """TODO: Docstring for plot_diff.

    :dyn: TODO
    :xs: TODO
    :returns: TODO

    """

    diff = (dyn - xs).transpose()
    size = 1 if np.ndim(diff) == 1 else np.shape(diff)[0]
    peak_index_list = []
    for i in range(size):
        if size>1:
            diff_i = diff[i]
        else:
            diff_i = diff

        peak_index, _ = list(find_peaks(diff_i))
        peak = diff_i[peak_index]
        positive_index = np.where(peak>0)[0]
        peak_positive = peak[positive_index]
        peak_index_positive = peak_index[positive_index]
        peak_index_list.append(peak_index_positive)
        if i +1 < size:
            plt.semilogy(peak_index_positive*dt, peak_positive, '.-', alpha = alpha, color=color)
        else:
            plt.semilogy(peak_index_positive*dt, peak_positive, '.-', label=label, alpha = alpha, color=color)

    return peak_index_list

def evolution_mutual_single(delay, beta, arguments):
    """TODO: Docstring for evolution.

    :arg1: TODO
    :returns: TODO

    """
    initial_condition = np.ones(1) * 5 
    dt = 0.001
    t = np.arange(0, 100, dt)
    t1 = time.time()
    dyn_all = ddeint_Cheng(mutual_single_delay, initial_condition, t, *(delay, beta, arguments))[:, 0]
    dyn_diff = np.diff(dyn_all) 
    
    t2 = time.time()
    print(t2-t1)
    #plt.plot(t, dyn_all, alpha = alpha, linewidth=2)
    plt.plot(dyn_all[1:][::10], dyn_diff[::10], '.', alpha = alpha, linewidth=2)
    #plt.plot(t, dyn_all, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$x$', fontsize= fs)
    plt.ylabel('$dx/dt$', fontsize =fs)
    #plt.show()
    return dyn_all

def evolution_mutual_multi(network_type, N, seed, d, delay, beta, betaeffect, arguments):
    """TODO: Docstring for evolution.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    initial_condition = np.ones(N) * 5.0
    dt = 0.001
    t = np.arange(0, 100, dt)
    t1 = time.time()
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    dyn_all = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
    
    t2 = time.time()
    print(t2-t1)
    y = np.mean(dyn_all, 1)
    dydt = np.diff(y) /dt
    

    plt.plot(t[::10], dyn_all[::10], alpha = alpha, linewidth=1, color='tab:red')
    #plt.plot(y[1:][::10], dydt[0:][::10], '.', alpha = alpha, linewidth=2, color='tab:red')
    #plt.plot(t, dyn_all, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)

    plt.xlabel('$\\langle x\\rangle$', fontsize =fs)
    plt.ylabel('$\\langle x\\rangle/dt$', fontsize =fs)
    #plt.show()
    return dyn_all


def decouple_chain_rule(A, arguments, xs):
    """TODO: wrong

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    dFx = (1 -xs / K) * (2 * xs / C-1)
    dFx_tau = -xs/K*(xs/C-1)
    xs_T = xs.reshape(len(xs), 1)
    denominator = D + E * xs + H * xs_T
    "A should be transposed to A_ji"
    dgx_i = A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 )
    dgx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
    print(np.sum(dgx_i>0), np.sum(dgx_j>0), np.sum(dgx_i<0), np.sum(dgx_j<0), np.sum(dFx<0))
    tau_list = []
    N = np.size(A, 0)
    for i in range(N):
        P = - (dFx[i] + np.sum(dgx_i[:, i]))
        for j in range(N):
            dxjxi = (-dgx_j[i, j]/( dFx[j] + np.sum(dgx_i[:, j]) + np.sum(np.heaviside(A[:, i], 0) *  dgx_j[:, j] )))
            P = P -   dgx_j[j, i]  *dxjxi
            #P = P - dgx_j[j, i]  *  (-dgx_j[i, j]/( dFx[j] ))
            #P = P
        Q = - dFx_tau[i]
        if abs(P/Q)<=1:
            tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
            nu = np.arccos(-P/Q)/tau
            tau_list.append(tau)
            #print(nu,tau)
    return tau_list 

def eigenvector_zeroeigenvalue(A, arguments, xs):
    """TODO: Docstring for eigenvector.

    :A: TODO
    :arguments: TODO
    :xs: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    xs_T = xs.reshape(len(xs), 1)
    denominator = D + E * xs + H * xs_T
    "A should be transposed to A_ji"
    gx_i = np.sum(A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
    gx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
    tau_sol = []
    nu_sol = []
    for initial_condition in np.array(np.meshgrid(tau_list, nu_list)).reshape(2, int(np.size(tau_list) * np.size(nu_list))).transpose():
        tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, gx_i, gx_j))
        "check the solution given by fsolve built-in function."
        eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, gx_i, gx_j)
        if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
            #print(tau_solution, nu_solution)
            tau_sol.append(tau_solution)
            nu_sol.append(nu_solution)
    tau_sol = np.array(tau_sol)
    tau_positive = tau_sol[tau_sol>0]
    nu_positive = np.array(nu_sol)[tau_sol>0]
    min_index = np.argmin(tau_positive)
    tau = tau_positive[min_index]
    nu = nu_positive[min_index]
    imag = 1j
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - gx_i) - gx_j 
    eigenvalue, eigenvector = np.linalg.eig(M)
    eigenvector_zero = eigenvector[:, np.argmin(np.abs(eigenvalue))]
    return eigenvector_zero, tau

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

def evolution_compare(network_type, arguments, N, beta, betaeffect, d, seed, delay, index):
    """TODO: Docstring for evolution_compare.

    :network_type: TODO
    :dynamics: TODO
    :arguments: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    beta, _ = betaspace(A, [0])
    N_actual = np.size(A, 0)
    w_list = np.sum(A, 0)
    w = np.sort(w_list)[index]
    A_index = np.where(w_list == w)[0][0]
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    initial_condition = np.ones((N_actual)) * 5.0
    t = np.arange(0, 200, 0.01)
    dt = 0.01
    xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(0, w, beta, arguments))[-1]
    #dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
    #dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(delay, w, beta, arguments))
    dyn_multi = ddeint_Cheng(mutual_multi_delay, xs-1e-3, t, *(delay, arguments, net_arguments))
    dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, xs_decouple - 1e-3, t, *(delay, w, beta, arguments))
    #plt.plot(t[:2000], dyn_multi[:2000, A_index], '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    #plt.plot(t[:2000], dyn_decouple[:2000, 0], '-', color='tab:blue', linewidth=lw, alpha=alpha, label='decouple')

    index_neighbor = np.where(A[A_index]>0)[0]
    s = np.sum(A[index_neighbor], 1)
    #plt.plot(t[:2000], np.mean(s * dyn_multi[:2000, index_neighbor], 1)/np.mean(s), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    #plt.plot(t[:2000], np.mean(np.sum(A, 0) * dyn_multi[:2000, :], 1)/np.mean(np.sum(A, 0)), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    plt.plot(t[:2000], np.mean(dyn_multi[:2000, index_neighbor], 1), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    plt.plot(t[:2000], dyn_decouple[:2000, 1], '-', color='tab:blue', linewidth=lw, alpha=alpha, label='decouple')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.locator_params(axis='x', nbins=5)
    plt.legend(fontsize=legendsize, frameon=False)

    plt.show()
    #plt.close()

    peak_index = plot_diff(dyn_multi[:, :], xs[:], dt, 'tab:red', 'multi')
    plot_diff(dyn_decouple[:, 0], xs_decouple[0], dt, 'tab:blue', 'decouple')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$A$', fontsize =fs)
    plt.locator_params(axis='x', nbins=5)
    plt.legend(fontsize=legendsize, frameon=False)
    #plt.ylim(10**(-9),1)
    plt.show()
    return peak_index

def evolution_multi(network_type, arguments, N, beta, betaeffect, d, seed, delay, index):
    """TODO: Docstring for evolution_compare.

    :network_type: TODO
    :dynamics: TODO
    :arguments: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    beta, _ = betaspace(A, [0])
    N_actual = np.size(A, 0)
    w_list = np.sum(A, 0)
    w = np.sort(w_list)[index]
    A_index = np.where(w_list == w)[0][0]
    net_arguments = (index_i, index_j, A_interaction, cum_index)

    initial_condition = np.ones((N_actual)) * 5.0
    t = np.arange(0, 200, 0.01)
    dt = 0.01
    xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(0, w, beta, arguments))[-1]
    dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
    dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(delay, w, beta, arguments))
    #dyn_multi = ddeint_Cheng(mutual_multi_delay, xs-1e-3, t, *(delay, arguments, net_arguments))
    #dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, xs_decouple - 1e-3, t, *(delay, w, beta, arguments))
    #plt.plot(t[:2000], dyn_multi[:2000, A_index], '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    #plt.plot(t[:2000], dyn_decouple[:2000, 0], '-', color='tab:blue', linewidth=lw, alpha=alpha, label='decouple')

    index_neighbor = np.where(A[A_index]>0)[0]
    s = np.sum(A[index_neighbor], 1)
    #plt.plot(t[:2000], np.mean(s * dyn_multi[:2000, index_neighbor], 1)/np.mean(s), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    #plt.plot(t[:2000], np.mean(np.sum(A, 0) * dyn_multi[:2000, :], 1)/np.mean(np.sum(A, 0)), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    plt.plot(t[:2000], dyn_multi[:2000, index_neighbor], '-', linewidth=lw, alpha=alpha, label='multi')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.locator_params(axis='x', nbins=5)
    plt.legend(fontsize=legendsize, frameon=False)

    plt.show()
    #plt.close()

    return None

def evolution_compare_eff(dynamics, network_type, arguments, N, beta, betaeffect, d, seed, delay, index):
    """TODO: Docstring for evolution_compare.

    :network_type: TODO
    :dynamics: TODO
    :arguments: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :d: TODO
    :returns: TODO

    """

    function_multi =  globals()[dynamics + '_multi']
    function_multi_delay =  globals()[dynamics + '_multi_delay']
    function_decouple =  globals()[dynamics + '_decouple_eff']
    function_decouple_delay =  globals()[dynamics + '_decouple_eff_delay']

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    beta, _ = betaspace(A, [0])
    N_actual = np.size(A, 0)
    w_list = np.sum(A, 0)
    w = np.sort(w_list)[::-1][index]
    A_index = np.where(w_list == w)[0][0]
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    print(beta, w)

    initial_condition = np.ones((N_actual)) * 1
    t = np.arange(0, 20, 0.01)
    dt = 0.01
    xs = odeint(function_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    xs_eff = odeint(function_decouple, initial_condition[:2], t, args=(w, beta, arguments))[-1]
    dyn_multi = dde_RK45(function_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
    dyn_eff = dde_RK45(function_decouple_delay, initial_condition[:2], t, *(delay, w, beta, arguments))

    index_neighbor = np.where(A[A_index]>0)[0]
    s = np.sum(A[index_neighbor], 1)
    plt.plot(t[:], np.mean(dyn_multi[:, index_neighbor], 1), '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    plt.plot(t[:], dyn_eff[:, 1], '-', color='tab:blue', linewidth=lw, alpha=alpha, label='decouple')

    #plt.plot(t[:], dyn_multi[:, A_index], '-', color='tab:red', linewidth=lw, alpha=alpha, label='multi')
    #plt.plot(t[:], dyn_eff[:, 0], '-', color='tab:blue', linewidth=lw, alpha=alpha, label='decouple')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.locator_params(axis='x', nbins=5)
    plt.legend(fontsize=legendsize, frameon=False)
    plt.show()
    #plt.close()

    return None


network_type = 'RR'
N = 1000

beta = 1
betaeffect = 1
seed1 = np.arange(100).tolist()

network_type = 'SF'
seed_SF = np.vstack((seed1, seed1)).transpose().tolist()

network_type = 'ER'
seed_ER = seed1

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

attractor_value = 5.0




"CW"
dynamics = 'CW'
arguments = (a, b)

"SIS"
dynamics = 'SIS'
arguments = (B_SIS, )

"harvest"
dynamics = 'harvest'
arguments = (r, K, c)
tau_list = np.arange(1., 1.5, 0.1)
nu_list = np.arange(0.1, 1.5, 0.5)


"PPI"
dynamics = 'PPI'
arguments = (B_PPI, F_PPI)
tau_list = np.arange(0.1, 2, 0.5)
nu_list = np.arange(0, 2, 0.5)

"BDP"
dynamics = 'BDP'
arguments = (B_BDP, )
tau_list = np.arange(0.5, 1, 0.2)
nu_list = np.arange(1, 5, 2)


"mutual"
dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)

"genereg"
dynamics = 'genereg'
arguments = (B_gene, )
tau_list = np.arange(1, 2, 0.5)
nu_list = np.arange(0.1, 1, 0.2)




wk_list = np.arange(0.1, 20, 0.1)
#n.tau_decouple_two()
network_list = ['SF', 'ER', 'RR']
network_list = ['SF']
d_RR = [4]
d_SF = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
d_SF = [[2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]]
d_SF = [[2.5, 999, 3]]
d_ER = [100, 200, 400, 800, 1600]
d_ER = [2000, 4000, 8000]
d_ER = [2000]
#beta_list = np.arange(60, 100, 0.01)
beta_list = [0.1]
betaeffect = 0

for network_type in network_list:
    if network_type == 'SF':
        d_list = d_SF
        seed_list = seed_SF
    elif network_type == 'ER':
        d_list = d_ER
        seed_list = seed_ER
    elif network_type == 'RR':
        d_list = d_RR
        seed_list = seed_ER
    elif network_type == '2D':
        d_list = [4]
        seed_list = [0]
    for beta in beta_list:
        for d in d_list:
            n = Net_Dyn(network_type, N, beta, betaeffect, seed_list, d, dynamics, attractor_value, arguments, tau_list, nu_list)
            #n.eigen_parallel('decouple_two')
            #n.eigen_parallel('eigen')
            #n.eigen_parallel('tau_evo', 0, 2)
            #n.eigen_parallel('tau_RK', 0, 2)
            #n.tau_decouple_eff()

for beta in beta_list:
    n = Net_Dyn(network_type, N, beta, betaeffect, seed_list, d, dynamics, attractor_value, arguments, tau_list, nu_list)
    #tau = n.tau_1D()

#n.tau_decouple_wk(wk_list)


beta = 0.1
delay = 0.05

#evolution_mutual_single(delay, beta, arguments)
dynamics ='mutual'
arguments = (B, C, D, E, H, K_mutual)
dynamics = 'PPI'
arguments = (B_PPI, F_PPI)
dynamics = 'BDP'
arguments = (B_BDP, )
#evolution_mutual_multi("ER", 1000, 41, 8000, 0.13,  1, 0,  arguments)
network_type = 'ER'
d = 2000
seed = 0
network_type = 'SF'
seed = [0, 0]
d = [2.5, 999, 3]

index = 0
evolution_compare_eff(dynamics, network_type, arguments, N, beta, betaeffect, d, seed, delay, index)
