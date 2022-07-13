import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')

from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, network_generate, stable_state, ode_Cheng, ddeint_Cheng

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
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['tau'])
        des = '../data/' + dynamics + '/' + network_type + '/'
        if not os.path.exists(des):
            os.makedirs(des)
        if betaeffect:
            des_file = des + f'_N={N}_d=' + str(d) + '_beta=' + str(beta) + '_logistic.csv'
        else:
            des_file = des + network_type + f'_N={N}_d=' + str(d) + '_wt=' + str(beta) + '_logistic.csv'

        self.save_data(des_file, data, column_name)
        print(seed, tau_critical)
        return tau_critical

    def eigen_parallel(self, function):
        """TODO: Docstring for tau_critical.

        :arg1: TODO
        :returns: TODO

        """
        p = mp.Pool(cpu_number)
        if function == 'eigen':
            p.starmap_async(self.eigen_solution, [(seed, ) for seed in self.seed_list]).get()
        elif function == 'decouple_two':
            p.starmap_async(self.tau_decouple_two, [(seed, ) for seed in self.seed_list]).get()
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




        print(xs, P, Q)

        if abs(P/Q)<=1:
            tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
            nu = np.arccos(-P/Q)/tau
            print(nu,tau)
            return tau
        else:
            return 0

    def tau_decouple_eff(self):
        """TODO: Docstring for tau_kmax.

        :network_type: TODO
        :N: TODO
        :beta: TODO
        :betaeffect: TODO
        :returns: TODO

        """

        network_type, N, beta, betaeffect, d, dynamics, attractor_value, arguments, seed_list  = self.network_type, self.N, self.beta, self.betaeffect, self.d, self.dynamics, self.attractor_value, self.arguments, self.seed_list

        des = '../data/' + dynamics + '/' + network_type + '/' 
        if not os.path.exists(des):
            os.makedirs(des)
        des_file = des + f'_N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_eff_logistic.csv'

        t = np.arange(0, 1000, 0.01)
        for seed in seed_list:
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
         
            column_name = [f'seed{i}' for i in range(np.size(seed))]
            column_name.extend(['kmax', str(beta), 'wk', 'order' ])
            self.save_data(des_file, data, column_name)

            print(seed, tau)

        return None

    def tau_decouple_two(self, seed):
        """TODO: Docstring for tau_kmax.

        :returns: TODO

        """
        network_type, N, beta, d, dynamics, attractor_value, arguments, tau_list, nu_list  = self.network_type, self.N, self.beta, self.d, self.dynamics, self.attractor_value, self.arguments, self.tau_list, self.nu_list
        des = '../data/' + dynamics + '/' + network_type + '/' 
        if not os.path.exists(des):
            os.makedirs(des)
        des_file = des + f'_N={N}_d=' + str(d) + '_beta=' + str(beta) + '_decouple_two_logistic.csv'

        t = np.arange(0, 1000, 0.01)

        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, 1, seed, d)
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
     
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['kmax', 'tau', 'index'])
        self.save_data(des_file, data, column_name)
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

def harvest_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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

def genereg_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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

def SIS_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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

def BDP_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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

def PPI_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    #x[np.where(x<0)] = 0  # Negative x is forbidden
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

def CW_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :returns: derivative of x 

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
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
    """TODO: Docstring for harvest_single_delay.

    :arg1: TODO
    :returns: TODO

    """
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    dxdt = B + x * (1 - xd/K) * ( x/C - 1) + beta * x**2/(D+E*x+H*x)
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

def mutual_multi_delay(f, x0, t, dt, d, arguments, net_arguments):
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
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt



def evolution_multi(network_type, arguments, N, beta, betaeffect, d, seed, delay, initial_value):
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
    N_actual = np.size(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    initial_condition = np.ones((N_actual)) * initial_value
    t = np.arange(0, 500, 0.01)
    xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))[-1]
    dyn_beta = betaspace(A, dyn_multi)
    if np.max(np.abs(dyn_multi-xs))< 1e-2:
        xs = dyn_multi[-1]
    else:
        xs = -1 * np.ones(N_actual)
    data = np.hstack((seed, xs))
    des = f'../data/mutual/' + network_type + '/xs/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect == 0:
        des_file = des + f'N={N}_d={d}_wt={beta}_delay={delay}_x0={initial_value}.csv'
    else:
        des_file = des + f'N={N}_d={d}_beta={beta}_delay={delay}_x0={initial_value}.csv'
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, mode='a', index=None, header=None)

    #dyn_multi = ddeint_Cheng(mutual_multi_delay, xs-1e-3, t, *(delay, arguments, net_arguments))
    #dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, xs_decouple - 1e-3, t, *(delay, w, beta, arguments))
    return None

def parallel_evolution(network_type, arguments, N, beta, betaeffect, d, seed_list, delay, initial_value):
    """TODO: Docstring for parallel_evolution.

    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(evolution_multi, [(network_type, arguments, N, beta, betaeffect, d, seed, delay, initial_value) for seed in seed_list]).get()
    p.close()
    p.join()
    return None



network_type = 'RR'
N = 100

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




"BDP"
dynamics = 'BDP'
arguments = (B_BDP, )
tau_list = np.arange(0.5, 1, 0.2)
nu_list = np.arange(1, 5, 2)


"CW"
dynamics = 'CW'
arguments = (a, b)

"SIS"
dynamics = 'SIS'
arguments = (B_SIS, )

"genereg"
arguments = (B_gene, )
dynamics = 'genereg'
tau_list = np.arange(1, 2, 0.5)
nu_list = np.arange(0.1, 1, 0.2)

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

"mutual"
dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)


wk_list = np.arange(0.1, 20, 0.1)
#n.tau_decouple_two()
network_list = ['SF', 'ER', 'RR']
network_list = ['SF']
d_RR = [4]
d_SF = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
d_SF = [[3, 99, 3]]
d_ER = [100, 200, 400, 800, 1600]
d_ER = [200]
beta_list = [0.1, 0.5, 1.5, 2, 2.5, 3]

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
            #n.tau_decouple_eff()
            #tau = n.tau_1D()

#n.tau_decouple_wk(wk_list)
network_type = 'SF'
network_type = 'ER'
N = 1000
beta = 4
betaeffect = 1
seed = 0
seed =[1, 1]
d = 2000
d = [3, 99, 3]
A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
N_actual = np.size(A, 0)
net_arguments = (index_i, index_j, A_interaction, cum_index)
initial_condition = np.ones((N_actual)) * attractor_value
t = np.arange(0, 1000, 0.01)
xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
#tau = decouple_chain_rule(A, arguments, xs)
#eigenvector, tau = eigenvector_zeroeigenvalue(A, arguments, xs)
delay = 0.25
index = 98
#peak_index = evolution_compare(network_type, arguments, N, beta, betaeffect, d, seed, delay, index)
initial_value = 5.0
evolution_multi(network_type, arguments, N, beta, betaeffect, d, seed, delay, initial_value)
