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







def mutual_single(x, t, beta, arguments):
    """original dynamics N species interaction.

    :x: 1 dynamic variable
    :t: time series
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + beta * x**2 / (D + (E+H) * x)
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


def mutual_one_delay(f, x0, t, dt, d, N, delay_node, index_i, index_j, A_interaction, cum_index, arguments):
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
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_f[delay_node] = B + x[delay_node] * (1 - xd[delay_node]/K) * ( x[delay_node]/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
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

def mutual_decouple_two_delay(f, x0, t, dt, d, w, beta, arguments):
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
    sum_g = np.array([w * x[0] * x[1] / (D + E * x[0] + H * x[1]), beta * x[1] * x[1] / (D + E * x[1] + H * x[1])])
    dxdt = sum_f + sum_g
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
            plt.semilogy(peak_index_positive*dt, peak_positive, '.-', alpha = alpha, color=color)
        else:
            plt.semilogy(peak_index_positive*dt, peak_positive, '.-', label=label, alpha = alpha, color=color)

    return 

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
    dyn_multi = ddeint_Cheng(mutual_multi_delay, initial_condition, t, *(delay, arguments, net_arguments))
    dyn_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(delay, w, beta, arguments))
    xs_decouple = ddeint_Cheng(mutual_decouple_two_delay, initial_condition[:2], t, *(0, w, beta, arguments))[-1]
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

    plot_diff(dyn_multi[:, A_index], xs[A_index], dt, 'tab:red', 'multi')
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
    return None
    
def eigenvalue_tau(x, tau, A, fx, fxt, gx_i, gx_j):
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
    mu, nu = x
    M = np.diagflat(mu + nu * imag - fx - fxt * np.exp(- (mu + nu * imag) * tau) - gx_i) - gx_j 
    eigenvalue, eigenvector = np.linalg.eig(M)
    zeropoint = eigenvalue[np.argmin(np.abs(eigenvalue))]
    return np.array([np.real(zeropoint), np.imag(zeropoint)])

def eigenvalue_tau_mutual(tau, A, fx, fxt, gx_i, gx_j, mu_list, nu_list, des_file):
    """TODO: Docstring for eigenvector.

    :A: TODO
    :arguments: TODO
    :xs: TODO
    :returns: TODO

    """

    mu_sol = []
    nu_sol = []
    for initial_condition in np.array(np.meshgrid(mu_list, nu_list)).reshape(2, int(np.size(mu_list) * np.size(nu_list))).transpose():
        mu_solution, nu_solution = fsolve(eigenvalue_tau, initial_condition, args=(tau, A, fx, fxt, gx_i, gx_j))
        "check the solution given by fsolve built-in function."
        eigen_real, eigen_imag = eigenvalue_tau(np.array([mu_solution, nu_solution]), tau, A, fx, fxt, gx_i, gx_j)
        if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
            #print(tau_solution, nu_solution)
            mu_sol.append(mu_solution)
            nu_sol.append(nu_solution)
    mu_max = np.max(mu_sol)
    print(tau, mu_max)
    data = np.hstack((tau, mu_max))
    df = pd.DataFrame(data.reshape(1, len(data)))
    df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_eigenvalue_tau_mutual(network_type, N, beta, betaeffect, seed, d, tau_list):
    """TODO: Docstring for eigenvector.

    :A: TODO
    :arguments: TODO
    :xs: TODO
    :returns: TODO

    """

    mu_list = np.arange(-3, 1, 0.2)
    nu_list = np.arange(1, 10, 1)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, seed, d)
    N_actual = np.size(A, 0)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    initial_condition = np.ones((N_actual)) * 5.0
    t = np.arange(0, 1000, 0.01)
    xs = odeint(mutual_multi, initial_condition, t, args=(arguments, net_arguments))[-1]

    des_file = '../data/mutual/' + network_type + '/' + f'N={N}_d={d}_beta={beta}_eigenvalue_tau.csv'
    mu_min = []
    B, C, D, E, H, K = arguments
    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    xs_T = xs.reshape(len(xs), 1)
    denominator = D + E * xs + H * xs_T
    "A should be transposed to A_ji"
    gx_i = np.sum(A * (xs_T/denominator - E * xs * xs_T/denominator ** 2 ), 0)
    gx_j = A * (xs/denominator - H * xs * xs_T/denominator ** 2 )
    p = mp.Pool(cpu_number)
    p.starmap_async(eigenvalue_tau_mutual, [(tau, A, fx, fxt, gx_i, gx_j, mu_list, nu_list, des_file) for tau in tau_list]).get()
    p.close()
    p.join()
    return  None



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




"mutual"
dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
tau_list = np.arange(0.2, 0.5, 0.1)
nu_list = np.arange(1, 10, 1)


wk_list = np.arange(0.1, 20, 0.1)
#n.tau_decouple_two()
network_list = ['SF', 'ER', 'RR']
network_list = ['ER']
d_RR = [4]
d_SF = [[2.5, 99, 3], [3, 99, 3], [3.5, 99, 3], [4, 99, 3]]
d_SF = [[3, 99, 3]]
d_ER = [100, 200, 400, 800, 1600]
d_ER = [200]
beta_list = [4]
beta_list = [1, 4]

network_type = 'ER'
network_type = 'SF'
N = 100
beta = 4
betaeffect = 1
seed = 0
seed =[0, 0]
d = 200
d = [3, 99, 3]

tau_list= np.arange(0.01, 0.4, 0.01)
parallel_eigenvalue_tau_mutual(network_type, N, beta, betaeffect, seed, d, tau_list)
