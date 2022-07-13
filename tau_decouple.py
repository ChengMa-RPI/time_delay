import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
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
K = 5
D = 5 
E = 0.9
H = 0.1

fs = 22
ticksize = 16
legendsize = 14
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

def mutual_single_delay(f, x0, t, dt, d1, d2, d3, N, delay_node, index_i, index_j, A_interaction, cum_index, arguments):
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
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_f[delay_node] = B + x[delay_node] * (1 - xd1[delay_node]/K) * ( xd2[delay_node]/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def decouple_two_delay(f, x0, t, dt, d1, d2, d3, w, beta, arguments):
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
    sum_g = np.array([w * x[0] * x[1] / (D + E * x[0] + H * x[1]), beta * x[1] * x[1] / (D + E * x[1] + H * x[1])])
    dxdt = sum_f + sum_g
    return dxdt


def one_single_delay(f, x0, t, dt, d1, d2, d3, w, x_fix, arguments):
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
    sum_g = w * x * x_fix / (D + E * x + H * x_fix)

    dxdt = sum_f + sum_g
    return dxdt

def one_kmax(x, w, x_fix, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """

    B, C, D, E, H, K = arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = w * x * x_fix / (D + E * x + H * x_fix)

    dxdt = sum_f + sum_g
    return dxdt

def evolution(network_type, N, beta, betaeffect, seed, arguments, d1, d2, d3, d=None):
    """TODO: not useful.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N = np.size(A, 0)
    initial_condition = np.ones(N) * 5 
    t = np.arange(0, 50,0.001)
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
    M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - gx_i) - gx_j 
    eigenvalue, eigenvector = np.linalg.eig(M)
    zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
    return np.array([np.real(zeropoint), np.imag(zeropoint)])

def tau_eigenvalue(network_type, N, beta, betaeffect, nu_set, tau_set, arguments, seed, d):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    degree_weighted = np.sum(A, 0)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    xs = xs_high

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    xs_T = xs.reshape(len(xs), 1)
    denominator = D + E * xs + H * xs_T
    gx_i = np.sum(A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
    gx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
    #gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    #gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2

    tau_sol = []
    for initial_condition in np.array(np.meshgrid(tau_set, nu_set)).reshape(2, int(np.size(tau_set) * np.size(nu_set))).transpose():
        tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, degree_weighted, gx_i, gx_j))
        eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, degree_weighted, gx_i, gx_j)
        if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
            tau_sol.append(tau_solution)
            print(initial_condition, tau_solution)
        else:
            tau_sol.append(1)
    tau_sol = np.array(tau_sol)
    tau_critical= np.min(tau_sol[tau_sol>0])
    #index_critical = np.where(np.abs(tau_sol - tau_critical[i])<1e-10)[0]
    data = np.hstack((seed, tau_critical))
 
    column_name = [f'seed{i}' for i in range(np.size(seed))]
    column_name.extend([ str(beta) ])

    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_beta=' + str(beta) + '={beta}_logistic.csv'
    else:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_wt=' + str(beta) + '={beta}_logistic.csv'

    if not os.path.exists(des_file):
        df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
        df.to_csv(des_file, index=None, mode='a')
    else:
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    print(seed)

    return None

def tau_eigen_parallel(network_type, N, arguments, beta_set, betaeffect, seed_list, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        p = mp.Pool(cpu_number)
        p.starmap_async(tau_eigenvalue, [(network_type, N, beta, betaeffect, nu_set, tau_set, arguments, seed, d) for seed in seed_list]).get()
        p.close()
        p.join()

        t2 = time.time()

    return None

def tau_evolution(network_type, N, beta, betaeffect, seed, arguments, delay1, delay2, criteria_delay, criteria_dyn, d):
    """TODO: Docstring for tau_evolution.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    print(seed)
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

    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + network_type + f'_N={N}_d=' + str(d) + '_evolution.csv'
    if not os.path.exists(des_file):
        df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
        df.to_csv(des_file, index=None, mode='a')
    else:
        df = pd.DataFrame(data.reshape(1, np.size(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    print('good')

    return delay2

def tau_evolution_parallel(network_type, N, beta, betaeffect, seed_list, arguments, delay1, delay2, criteria_delay, criteria_dyn, d):
    """TODO: Docstring for tau_evolution_parallel.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """

    p = mp.Pool(cpu_number)
    p.starmap_async(tau_evolution, [(network_type, N, beta, betaeffect, seed, arguments, delay1, delay2, criteria_delay, criteria_dyn, d) for seed in seed_list]).get()
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

def evolution_analysis(network_type, N, beta, betaeffect, seed, d, delay):
    """TODO: Docstring for evolution_oscillation.

    :arg1: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    beta_eff, _ = betaspace(A, [0])
    degree= np.sum(A>0, 0)
    index = np.argmax(degree)
    N = np.size(A, 0)
    initial_condition = np.ones(N) * 5
    initial_condition = xs_high - 0.0001
    dt = 0.001
    t = np.arange(0, 50, dt)
    t1 = time.time()
    dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, 0, 0, N, index_i, index_j, A_interaction, cum_index, arguments))
    t2 = time.time()
    plot_diff(dyn_multi, xs_high, dt, 'tab:red', 'multi-delay')
    dyn_single = ddeint_Cheng(mutual_single_delay, initial_condition, t, *(delay, 0, 0, N, [index], index_i, index_j, A_interaction, cum_index, arguments))
    t3 = time.time()
    plot_diff(dyn_single[:, index], xs_high[index], dt, 'tab:blue', 'single-delay')
    w = np.sum(A[index])
    #xs_eff = fsolve(mutual_1D, np.mean(initial_condition), args=(0, beta_eff, arguments))
    xs_eff = np.mean(xs_high)
    #xs_eff = np.mean(np.setdiff1d(xs_high, xs_high[index]))
    xs_high_max = ddeint_Cheng(one_single_delay, np.array([xs_high[index]]), t, *(0, 0, 0, w, xs_eff, arguments))[-1]
    initial_condition = np.ones(1) * 5
    initial_condition = xs_high_max - 0.0001

    t4 = time.time()
    dyn_one = ddeint_Cheng(one_single_delay, initial_condition, t, *(delay, 0, 0, w, xs_eff, arguments))[:, 0]
    t5 = time.time()
    print(t2-t1, t3-t2, t5-t4)
    plot_diff(dyn_one, xs_high_max, dt, 'tab:green', 'one-component')
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
        if i +1 < size:
            plt.semilogy(peak_index_positive*dt, peak_positive, '.', color=color, alpha = alpha)
        else:
            plt.semilogy(peak_index_positive*dt, peak_positive, '.', color=color, label=label, alpha = alpha)

    return None

def tau_decouple(network_type, N, d, beta, betaeffect, arguments, seed_list):
    """TODO: Docstring for tau_kmax.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_beta=' + str(beta) + '_logistic.csv'
    else:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_wt=' + str(beta) + '_logistic.csv'

    for seed in seed_list:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
        degree = np.sum(A>0, 0)
        x_fix = np.mean(xs_high)
        index_list = np.argsort(degree)[-10:]
        tau = []
        for index in index_list:
            w = np.sum(A[index])
            xs = ddeint_Cheng(one_single_delay, np.ones(1)*5, np.arange(0, 100, 0.01), *(0, 0, 0, w, x_fix, arguments))[-1]
            #xs = fsolve(one_kmax, np.ones(1) * 10, args=(w, x_fix, arguments))
            P =  - (w * x_fix)/(D + E * xs + H * x_fix) + (w * E * xs * x_fix)/(D + E * xs + H * x_fix)**2 -(1-xs/K) * (2*xs/C-1)
            Q = xs/K*(xs/C-1)
            if abs(P/Q)<=1:
                tau.append( np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) )
        tau = np.min(tau)
        data = np.hstack((seed, degree.max(), tau))
     
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['kmax', str(beta) ])

        if not os.path.exists(des_file):
            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau)

    return None

def tau_decouple_eff(network_type, N, d, beta, betaeffect, arguments, seed_list):
    """TODO: 10 largest degree to decide critical point.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_eff_beta=' + str(beta) + '_logistic.csv'
    else:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_eff_wt=' + str(beta) + '_logistic.csv'

    for seed in seed_list:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        beta_eff, _ = betaspace(A, [0])
        xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
        wk = np.sum(A, 0)
        x_fix = odeint(mutual_1D, np.ones(1) * 5, np.arange(0, 200, 0.01), args=(beta_eff, arguments))[-1]
        index_list = np.argsort(wk)[-10:]
        tau_list = np.ones(len(index_list)) * 100
        for index, i in zip(index_list, range(len(index_list))):
            w = np.sum(A[index])
            xs = ddeint_Cheng(one_single_delay, np.ones(1)*5, np.arange(0, 200, 0.01), *(0, 0, 0, w, x_fix, arguments))[-1]
            P =  - (w * x_fix)/(D + E * xs + H * x_fix) + (w * E * xs * x_fix)/(D + E * xs + H * x_fix)**2 -(1-xs/K) * (2*xs/C-1)
            Q = xs/K*(xs/C-1)
            if abs(P/Q)<=1:
                tau_list[i] = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) 
        tau = np.min(tau_list)
        tau_index = index_list[np.where(tau == tau_list)[0][0]]
        data = np.hstack((seed, wk.max(), tau, wk[tau_index], np.where(np.sort(wk)[::-1]==wk[tau_index])[0][-1]))
     
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['kmax', str(beta), 'wk', 'order' ])

        if not os.path.exists(des_file):
            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau)

    return None

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

def tau_two(network_type, N, d, beta, betaeffect, arguments, seed_list):
    """TODO: Docstring for tau_kmax.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_two_beta=' + str(beta) + '_logistic.csv'
    else:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_two_wt=' + str(beta) + '_logistic.csv'

    for seed in seed_list:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
        beta_eff, _ = betaspace(A, [0])
        degree = np.sum(A>0, 0)
        x_fix = np.mean(xs_high)
        index_list = np.argsort(degree)[-10:]
        tau_individual = []
        for index in index_list:
            w = np.sum(A[index])
            xs = ddeint_Cheng(decouple_two_delay, np.ones(2)*5, np.arange(0, 100, 0.01), *(0, 0, 0, w, beta_eff, arguments))[-1]
            fx = (1-xs/K) * (2*xs/C-1)
            fxt = -xs/K*(xs/C-1)
            g11 = w * (xs[1] /(D + E*xs[0] + H*xs[1]) - E*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
            g12 = w * (xs[0] /(D + E*xs[0] + H*xs[1]) - H*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
            g21 = 0
            g22 = beta_eff * (2*xs[1]/(D+E*xs[1]+H*xs[1]) - xs[1]**2 * (E+H) /(D+E*xs[1]+H*xs[1])**2)
            g_matrix = np.array([[g11, g12], [g21, g22]])
            tau_sol = []
            for initial_condition in np.array(np.meshgrid(tau_set, nu_set)).reshape(2, int(np.size(tau_set) * np.size(nu_set))).transpose():
                tau_solution, nu_solution = fsolve(eigen_two_decouple, initial_condition, args=(fx, fxt, g_matrix))
                eigen_real, eigen_imag = eigen_two_decouple(np.array([tau_solution, nu_solution]), fx, fxt, g_matrix)
                if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                    tau_sol.append(tau_solution)
            tau_sol = np.array(tau_sol)
            if np.size(tau_sol[tau_sol>0]):
                tau_individual.append(np.min(tau_sol[tau_sol>0]))



        tau = np.min(tau_individual)
        data = np.hstack((seed, degree.max(), tau))
     
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['kmax', str(beta) ])

        if not os.path.exists(des_file):
            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau)

    return None

def tau_two_single_delay(network_type, N, d, beta, betaeffect, arguments, seed_list):
    """TODO: Docstring for tau_kmax.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_two_single_beta=' + str(beta) + '_logistic.csv'
    else:
        des_file = des + network_type + f'_N={N}_d=' + str(d) + '_decouple_two_single_wt=' + str(beta) + '_logistic.csv'

    for seed in seed_list:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
        xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
        beta_eff, _ = betaspace(A, [0])
        degree = np.sum(A>0, 0)
        x_fix = np.mean(xs_high)
        index_list = np.argsort(degree)[-10:]
        tau_individual = []
        for index in index_list:
            w = np.sum(A[index])
            xs = ddeint_Cheng(decouple_two_delay, np.ones(2)*5, np.arange(0, 100, 0.01), *(0, 0, 0, w, beta_eff, arguments))[-1]
            #fx = np.array([(1-xs[0]/K) * (2*xs[0]/C-1), (1-xs[1]/K) * (2*xs[1]/C-1) -xs[1]/K*(xs[1]/C-1)])
            #fxt = np.array([-xs[0]/K*(xs[0]/C-1), 0])
            g11 = w * (xs[1] /(D + E*xs[0] + H*xs[1]) - E*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
            g12 = w * (xs[0] /(D + E*xs[0] + H*xs[1]) - H*xs[0]*xs[1] /(D + E*xs[0] + H*xs[1])**2)
            g21 = 0
            g22 = beta_eff * (2*xs[1]/(D+E*xs[1]+H*xs[1]) - xs[1]**2 * (E+H) /(D+E*xs[1]+H*xs[1])**2)
            g_matrix = np.array([[g11, g12], [g21, g22]])
            tau_sol = []
            for initial_condition in np.array(np.meshgrid(tau_set, nu_set)).reshape(2, int(np.size(tau_set) * np.size(nu_set))).transpose():
                tau_solution, nu_solution = fsolve(eigen_two_decouple, initial_condition, args=(fx, fxt, g_matrix))
                eigen_real, eigen_imag = eigen_two_decouple(np.array([tau_solution, nu_solution]), fx, fxt, g_matrix)
                if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                    tau_sol.append(tau_solution)
            tau_sol = np.array(tau_sol)
            if np.size(tau_sol[tau_sol>0]):
                tau_individual.append(np.min(tau_sol[tau_sol>0]))



        tau = np.min(tau_individual)
        data = np.hstack((seed, degree.max(), tau))
     
        column_name = [f'seed{i}' for i in range(np.size(seed))]
        column_name.extend(['kmax', str(beta) ])

        if not os.path.exists(des_file):
            df = pd.DataFrame(data.reshape(1, np.size(data)), columns = column_name)
            df.to_csv(des_file, index=None, mode='a')
        else:
            df = pd.DataFrame(data.reshape(1, np.size(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
        print(seed, tau)

    return None

def evolution_single(network_type, N, beta, betaeffect, seed, arguments, d1, d2, d3, d):
    """TODO: Docstring for evolution.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    x_fix = np.mean(xs_high)
    initial_condition = np.ones(1) * 5 
    t = np.arange(0, 500,0.001)
    t1 = time.time()

    degree= np.sum(A>0, 0)
    index = np.argmax(degree)
    w = np.sum(A[index])
    initial_condition = np.array([xs_high[index]])
    dyn_all = ddeint_Cheng(one_single_delay, initial_condition, t, *(d1, d2, d3, w, x_fix, arguments))
    
    xs = ddeint_Cheng(one_single_delay, initial_condition, t, *(0, 0, 0, w, x_fix, arguments))[-1]

    B, C, D, E, H, K = arguments
    P =  - (w * x_fix)/(D + E * xs + H * x_fix) + (w * E * xs * x_fix)/(D + E * xs + H * x_fix)**2 -(1-xs/K) * (2*xs/C-1)

    Q = xs/K*(xs/C-1)
    tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) 

    t2 = time.time()
    print(t2-t1)
    plt.plot(t, dyn_all, alpha = alpha)
    #plt.plot(t, dyn_all, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$ x $', fontsize =fs)
    #plt.show()
    return dyn_all, tau

def critical_wk(beta, betaeffect, wk_list, arguments):
    """TODO: Docstring for critical_wk.

    :arg1: TODO
    :returns: TODO

    """

    B, C, D, E, H, K = arguments
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    if betaeffect:
        des_file = des + 'beta=' + str(beta) + '_logistic.csv'
    else:
        des_file = des + 'wt=' + str(beta) + '_logistic.csv'

    x_fix = odeint(mutual_1D, np.ones(1) * 5, np.arange(0, 200, 0.01), args=(beta, arguments))[-1]
    tau = np.zeros((len(wk_list)))
    for i, wk in zip(range(len(wk_list)), wk_list):
        xs = ddeint_Cheng(one_single_delay, np.ones(1)*5, np.arange(0, 200, 0.01), *(0, 0, 0, wk, x_fix, arguments))[-1]
        #xs = fsolve(one_kmax, np.ones(1) * 10, args=(w, x_fix, arguments))
        P =  - (wk * x_fix)/(D + E * xs + H * x_fix) + (wk * E * xs * x_fix)/(D + E * xs + H * x_fix)**2 -(1-xs/K) * (2*xs/C-1)
        Q = xs/K*(xs/C-1)
        if abs(P/Q)<=1:
            tau[i] = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q))
    data = np.vstack((wk_list, tau))
     
    if not os.path.exists(des_file):
        df = pd.DataFrame(data.transpose())
        df.to_csv(des_file, index=None, header=None, mode='a')
    else:
        df = pd.DataFrame(data.transpose())
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None


imag = 1j
N = 1000

tau_set = np.array([0.316])
nu_set = np.array([5.2])
tau_set = np.arange(0.2, 0.5, 0.1)
nu_set = np.arange(1, 10, 1)
arguments = (B, C, D, E, H, K)
d = 0.8
d = 3
d_set = [900]
d_set = [200, 400, 600, 900, 1600, 2500]
d_set = [5, 10, 15, 20, 25, 50]
d_set = [2, 2.5, 3, 3.5, 4, 4.5, 5]
d_set = [200, 300, 400, 500]
network_type = 'real'
network_type = 'RR'
network_type = 'BA'
network_type = 'star'
network_type = 'ER'
network_type = '2D'
network_type = 'SF'
seed_set = np.arange(0, 500, 1).tolist()
betaeffect = 1
beta_set = np.arange(1, 2, 1)
N_list = [1000]

kmin = [1]
gamma = [3]
d_list = np.hstack((np.meshgrid(kmin, gamma)[0], np.meshgrid(kmin, gamma)[1]))
d_list = [[3, 2500-2, 1]]
seed2 = [0]
seed1 = np.arange(100).tolist()
seed_list = np.hstack((np.meshgrid(seed1, seed2)[0], np.meshgrid(seed1, seed2)[1])).tolist()
seed_list = np.vstack((seed1, seed1)).transpose().tolist()

d_list = [[3, 49, 3]]
N_list = [50]
'''
t1 = time.time()
for N in N_list:
    for d in d_list:
        tau_eigen_parallel(network_type, N, arguments, beta_set, betaeffect, seed_list, nu_set = nu_set, tau_set = tau_set, d = d)
t2 = time.time()
print(t2 -t1)
'''

betaeffect = 1
beta = 1
d1 = 0.31
d2 = 0
d3 = 0
seed = 499

seed = [99, 99]
delay1 = 0.2
delay2 = 0.3
criteria_delay = 1e-3
criteria_dyn = 1e-9
d = [3, 999, 3]

'''
t1 = time.time()
tau = tau_evolution_parallel(network_type, N, beta, betaeffect, seed_list, arguments, delay1, delay2, criteria_delay, criteria_dyn, d)
t2 = time.time()
print(t2 - t1)
'''


delay = 0.35
N = 1000
betaeffect = 1
beta = 1
beta_list = [0.05, 0.1, 0.2]
beta_list = [0.1, 0.05, 0.2, 0.01]
beta_list = [0.1]
d_list = [[2.5, 999, 3], [3, 999, 3], [3.5, 999, 3], [4, 999, 3]]
seed = [187, 187]
#dyn_multi, xs_multi = evolution_analysis(network_type, N, beta, betaeffect, seed, d, delay)
#dyn_all = evolution(network_type, N, beta, betaeffect, seed, arguments, delay, 0, 0, d)

for d in d_list:
    for beta in beta_list:
        #tau = tau_kmax(network_type, N, d, beta, betaeffect, arguments, seed_list)
        tau = tau_two(network_type, N, d, beta, betaeffect, arguments, seed_list)
        #tau_decouple_eff(network_type, N, d, beta, betaeffect, arguments, seed_list)
        #tau_two_single_delay(network_type, N, d, beta, betaeffect, arguments, seed_list)
        #tau = tau_decouple(network_type, N, d, beta, betaeffect, arguments, seed_list)

#dyn, tau = evolution_single(network_type, N, beta, betaeffect, seed, arguments, delay, 0, 0, d)
beta = 1
betaeffect = 1
wk_list = np.arange(0.1, 20, 0.1)

#critical_wk(beta, betaeffect, wk_list, arguments)
