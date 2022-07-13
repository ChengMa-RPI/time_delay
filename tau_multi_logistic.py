import sys
sys.path.insert(1, '/home/mac/RPI/research/')

import sympy as sp
import numpy as np 
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
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, mutual_1D, mutual_multi, network_generate, stable_state


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 

fs = 18
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1
beta = 1
c1 = -1 / K / C
c2 = 1/K + 1/C 


def ddeint_Cheng(f, y0, tspan, *args):
    """Solve ordinary differential equation by simple integration

    :f: function that governs the deterministic part
    :g: before t=0 
    :tspan: simulation period
    :returns: solution of y 

    """
    N = len(tspan)
    d = np.size(y0)
    dt = (tspan[N-1] - tspan[0])/(N - 1)
    # allocate space for result
    y = np.zeros((N, d), dtype=type(y0[0]))
    y[0] = y0
    for n in range(N-1):
        tn = tspan[n]
        yn = y[n]
        y[n+1] = yn + f(y, y0, tn, dt, *args) * dt
    return y


def mutual_lattice(x, t, N, index, degree, A_interaction, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :degree: the degree of lattice  
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - x/K) * ( x/C - 1) +  x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)
    return dxdt

def tau_multi_interaction(N, nu_set, tau_set, arguments, c, low=0.1, high=10):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    dynamics = mutual_lattice
    B, C, D, E, H, K = arguments
    imag = 1j
    G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    Degree = np.sum(A, 0)
    degree = Degree[0]
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs = xs_high

    fx = -3 /(K*C) * xs**2 + 2 * (1/C + 1/K) * xs - 1
    gx = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gxt = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2
    #l = sp.Symbol('l')
    #M = sp.Matrix( l * np.eye(N) + A * sp.exp(-l *tau))
    for  nu in nu_set:
        for tau in tau_set:
            M = np.diagflat(nu * imag - fx) - np.diagflat(Degree * gx) - A * gxt * np.exp(- nu * tau * imag) 
            det = np.linalg.det(M)
            distance = np.real(det) **2 + np.imag(det) **2
            if distance  < 1:
                print(nu, tau)
    return distance 

def tau_multi_K(N, nu_set, tau_set, arguments, c, low=0.1, high=10):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    dynamics = mutual_lattice
    B, C, D, E, H, K = arguments
    imag = 1j
    G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    degree = 4
    A = np.array(nx.adjacency_matrix(G).todense()) * c/degree
    Degree = np.sum(A>0, 0)
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs = xs_high

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2

    distance = np.zeros((np.size(tau_set), np.size(nu_set)))
    for tau, i in zip(tau_set, range(np.size(tau_set))):
        for  nu, j in zip(nu_set, range(np.size(nu_set))):
            M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - Degree * gx_i) - A * gx_j * c/degree
            # M = np.array([[nu * imag - fx[0] - fxt[0] * np.exp(- nu * tau * imag) - gx[0] * c]])
            det = np.linalg.det(M)
            dist = np.real(det) **2 + np.imag(det) **2
            distance[i, j] = dist
    dis_min = np.argmin(distance)   
    tau_critical = tau_set[int(np.floor(dis_min/np.size(nu_set)))] 
    return tau_critical 

def matrix_variable(x, A, fx, fxt, Degree, gx_i, gx_j):
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
    det = np.linalg.det(M)
    det_real = np.real(det)
    det_imag = np.imag(det)
    return np.array([det_real, det_imag])

def tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d=None, low=0.1, high=10):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    Degree_weighted = np.sum(A, 0)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments, d)
    xs = xs_high

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2

    tau_sol = np.ones((np.size(tau_set), np.size(nu_set))) * 10
    for tau, i in zip(tau_set, range(np.size(tau_set))):
        for nu, j in zip(nu_set, range(np.size(nu_set))):
            initial_condition = np.array([tau, nu])
            tau_solution, nu_solution = fsolve(matrix_variable, initial_condition, args=(A, fx, fxt, Degree_weighted, gx_i, gx_j))
            det_real, det_imag = matrix_variable(np.array([tau_solution, nu_solution]), A, fx, fxt, Degree_weighted, gx_i, gx_j)
            if abs(det_real) < 1e-2 and abs(det_imag) < 1e-2:
                tau_sol[i, j] = tau_solution
                #print(tau, nu, det_real, det_imag, tau_solution, nu_solution)
    return tau_sol

def coeff(R, alpha, beta):
    """coefficient of ODE function after 1st order approximation 

    :R: fixed point 
    :alpha: 
    :beta: TODO
    :returns: TODO

    """
    c3 = alpha * beta 
    c4 = beta * E + H 
    P = - (B * c4 - D + (2 * (D/C + c3 - c4 ) + D / K) * R + (3 * c4 / C + 2 * (c4 / K - D / C / K)) * R ** 2 - c4 / C / K * 3 * R ** 3 ) / (D + c4 * R)
    Q = -(R/K - R **2 /C/ K)
    return P, Q

def coeff_interaction_j(arguments, c, low=0.1, high=10):
    """TODO: Docstring for coeff_interaction.

    :R: TODO
    :alpha: TODO
    :beta: TODO
    :returns: TODO

    """

    dynamics = mutual_lattice
    N = 9
    B, C, D, E, H, K = arguments
    G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    A = np.array(nx.adjacency_matrix(G).todense())
    Degree = np.sum(A, 0)
    degree = Degree[0]
    xs_low, xs_high = stable_state(A, degree, dynamics, c, low, high, arguments)
    xs = xs_high[0]
    print(xs)
    beta = c
    P = (xs * (-1 + xs/C))/K - (xs * (1 - xs/K))/C - (-1 + xs/C) * (1 - xs/K) + (beta * E * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs)
    Q = (beta * H * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs)
    return P, Q

def coeff_K(arguments, beta, low=0.1, high=10):
    """TODO: Docstring for coeff_interaction.

    :R: TODO
    :alpha: TODO
    :beta: TODO
    :returns: TODO

    """
    network_type = '2D'
    seed = 0
    d = 0
    N = 9
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    beta_eff, _ = betaspace(A, [0])
    weight = beta/ beta_eff
    A = A * weight

    B, C, D, E, H, K = arguments
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, low, high, arguments)
    xs = xs_high[0]
    P = (beta * E * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs) + (beta * H * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs) -(1-xs/K) * (2*xs/C-1)
    Q = xs/K*(xs/C-1)
    tau1 = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) 
    f = lambda x: Q * np.exp(1 + P*x)*x - 1
    initial_condition = np.array([0.1])
    tau2 = fsolve(f, initial_condition)

    return P, Q, tau1, tau2

def mutual_lattice_delay(f, x0, t, dt, d1, d2, d3, N, index, degree, A_interaction, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    '''
    x = f(t)
    xd1 = f(t-d1)
    xd2 = f(t-d2)
    xd3 = f(t-d3)
    '''
    x = f[int(t/dt)]
    xd1 = np.where(t>d1, f[int((t-d1)/dt)], x0)
    xd2 = np.where(t>d2, f[int((t-d2)/dt)], x0)
    xd3 = np.where(t>d3, f[int((t-d3)/dt)], x0)

    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - xd1/K) * ( xd2/C - 1) +  x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)

    return dxdt

def mutual_multi_delay_original(f, t, d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    x = f(t)
    xd1 = f(t-d1)
    xd2 = f(t-d2)
    xd3 = f(t-d3)

    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    sum_f = B + x * (1 - xd1/K) * ( xd2/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    dxdt = sum_f + x * np.array([sum_g[i:j].sum() for i, j in zip(cum_index[:-1], cum_index[1:])])
    return dxdt

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

    cumsum_g = np.add.reduceat(sum_g, cum_index[:-1])
    #print(np.where(np.abs(original-cumsum_g)!=0),cum_index)
    dxdt = sum_f + x * cumsum_g 
    return dxdt

def tau_multi_critical(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_critical = np.zeros(np.size(beta_set))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        tau_sol = np.ravel(tau_eigenvalue(network_type, N, beta, nu_set, tau_set, arguments, seed, d=d))
        tau_critical[i] = np.min(tau_sol[tau_sol>0])
        t2 = time.time()
        print(i, t2 - t1, tau_critical)

    A, A_interaction, index_i, index_j, cum_inde = network_generate(network_type, N, beta, seed, d)
    A = np.heaviside(A, 0)
    degree = np.sum(A, 0)
    hetero = np.sum((degree - np.mean(degree))**2)/ N

    data = np.hstack((seed, np.mean(degree), hetero, tau_critical))
    data = pd.DataFrame(data.reshape(1, np.size(data)))
    data.to_csv('../report/report101920/' + network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None, index=None, mode='a')

    plt.plot(beta_set, tau_critical, linewidth=lw, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.show()
    return tau_critical

def tau_multi_solution(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_all = []
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        tau_matrix = tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d)
        tau_matrix = np.round(tau_matrix, 3)
        tau_sol = np.ravel(tau_matrix)
        plt.hist(tau_sol, bins = np.arange(0, 0.5, 0.005), linewidth=lw, alpha = alpha)
        plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.ylabel('frequency', fontsize= fs)
        plt.xlabel('$\\tau_c$', fontsize =fs)
        plt.savefig('../report/report090820/' + network_type +'_N=' +  str(N) + '_beta=' +  str(beta) + 'hist_sol.png')
        plt.close('all')
        #plt.show()
        tau_all.append(np.unique(np.round(tau_sol[tau_sol>0], 3)).tolist())
        tau_c = np.min(tau_matrix[tau_matrix>0])
        index = np.where(tau_c == tau_matrix)
        tau_good = tau_set[index[0]]
        nu_good = nu_set[index[1]]
        print(beta, tau_good, nu_good)

    for beta, i in zip(beta_set, range(np.size(beta_set))):
        plt.plot(beta * np.ones(np.size(tau_all[i])), tau_all[i], 'o')
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.show()


    return tau_all

def tau_multi_solution_pattern(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_all = []
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        tau_matrix = tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d)
        tau_matrix = np.round(tau_matrix, 3)
        tau_unique_true = np.sort(np.unique(np.ravel(tau_matrix)))
        tau_matrix[tau_matrix<0] = 1
        bound = min(5, np.sum(tau_unique_true>0)-1)
        tau_matrix[tau_matrix>tau_unique_true[tau_unique_true>0][bound]] = 2
        tau_min = np.min(tau_matrix)
        initial_index = np.where(tau_matrix == tau_min)
        tau_initial = tau_set[initial_index[0]]
        nu_initial = nu_set[initial_index[1]]
        '''
        bound = tau_unique_true[:5]
        for k in range(np.size(bound)-1):
            tau_matrix[(tau_matrix > bound[k]) & (tau_matrix<bound[k+1])] = k+3
        '''

        tau_unique = np.sort(np.unique(np.ravel(tau_matrix)))
        
        color=itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan'))
        for tau_sol in tau_unique:
            if tau_sol == 1:
                label = '$negative \\tau$'
            elif tau_sol == 2:
                label = 'large $\\tau$'
            else:
                label = '$\\tau$='+ str(tau_sol)

            index = np.where(tau_matrix==tau_sol)
            x = tau_set[index[0]]
            y = nu_set[index[1]]
            plt.plot(x, y, '.', color=next(color), label=label)
        plt.xlabel('$\\tau_{0}$', fontsize= fs)
        plt.ylabel('$\\nu_0$', fontsize =fs)
        plt.subplots_adjust(left=0.18, right=0.78, wspace=0.25, hspace=0.25, bottom=0.20, top=0.85)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(fontsize=legendsize, loc='upper right', bbox_to_anchor=(1.40, 1.0), framealpha=0)
        plt.locator_params(axis='x', nbins=5)



        tau_all.append(tau_unique_true)
        #sns.heatmap(tau_matrix, cmap="YlGnBu", linewidth=0, vmin=tau_matrix.min(), vmax=tau_matrix.max(), xticklabels=xlabel, yticklabels=ylabel, cbar_kws={"drawedges":False, 'label': ' $\\langle \\tau \\rangle$'} )

        #plt.savefig('../report/report092220/' + network_type +'_N=' +  str(N) + '_beta=' +  str(beta) + '_seed=' + str(seed) + '.png')
        #plt.close('all')
        plt.show()

    return tau_all, tau_initial, nu_initial

def tau_critical(arguments, beta_set, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau1_critical = np.zeros(np.size(beta_set))
    tau2_critical = np.zeros(np.size(beta_set))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        P, Q, tau1, tau2 = coeff_K(arguments, beta, low=0.1, high=10)
        tau1_critical[i] = tau1
        tau2_critical[i] = tau2
        t2 = time.time()
        print(i, t2 - t1)
    plt.plot(beta_set, tau1_critical, linewidth=lw, alpha = alpha, label='$\\tau_1$')
    plt.plot(beta_set, tau2_critical, linewidth=lw, alpha = alpha, label='$\\tau_2$')
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.show()
    return tau1_critical, tau2_critical

def evolution(network_type, N, beta, seed, arguments, d1, d2, d3, d=None):
    """TODO: Docstring for evolution.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)


    g = lambda t: np.ones(N) * 5 
    t = np.arange(0, 1000,0.01)
    #dyn_all = ddeint(mutual_multi_delay_original, g, t, fargs=(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
    t1 = time.time()
    dyn_all = ddeint_Cheng(mutual_multi_delay, np.ones(N) *5, t, *(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
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

def plot_network_effect(network_set, d_set, N_set, beta_set):
    """TODO: Docstring for plot_network_effect.

    :network_set: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """
    for network_type, i in zip(network_set, range(len(network_set))):
        d = d_set[i]
        N = N_set[i]
        data = np.array(pd.read_csv('../report/report091620/'+ network_type + f'_N={N}' + '_logistic.csv', header=None).iloc[:, :])
        #plt.plot(beta_set, data[:, 3:].transpose(), '-o', linewidth=lw, alpha = alpha, label=network_type)
        plt.plot(beta_set, data[:, 3:].transpose(), '-o', linewidth=lw, alpha = alpha)
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend(np.round(data[:, 2], 2), fontsize=legendsize)
    plt.show()

def plot_hetero_effect(network_type, d_set, N, beta):
    """TODO: Docstring for plot_network_effect.

    :network_set: TODO
    :d: TODO
    :N: TODO
    :returns: TODO

    """
    degree = []
    for d in d_set:
        data = np.array(pd.read_csv('../report/report101920/'+ network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None).iloc[:, :])
        seed = data[:, 0]
        degree.append(np.mean(data[:, 1]))
    degree = np.array(degree)
    degree_unique = np.unique(degree)
    for i in degree_unique:
        index = np.where(degree == i)[0]
        data = []
        for j in index:
            d = d_set[j]
            data.extend(np.array(pd.read_csv('../report/report101920/'+ network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None).iloc[:, :]))
        data = np.array(data)
        #hetero = data[:, 2]/(i)**2
        hetero = data[:, 2]
        tau_all = data[:, 3:].transpose()
        beta_set = np.arange(1, np.size(tau_all, 0)+1, 1)
        tau = tau_all[np.where(beta==beta_set)[0][0]]

        #plt.plot(beta_set, data[:, 3:].transpose(), '-o', linewidth=lw, alpha = alpha, label=network_type)
        plt.plot(hetero, tau, 'o', linewidth=lw, alpha = alpha, label=f'$\\langle k \\rangle$ ={round(i,2)}')
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('heterogeneity', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    #plt.legend( fontsize=legendsize, frameon=False, loc='lower left')
    plt.legend( fontsize=legendsize, frameon=False, loc='upper right')
    #plt.show()

def verify1(network_type, N, beta, seed=0):
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed)
    det = lambda a: np.linalg.det(a *np.identity(N) + A)
    initial_condition = np.arange(-10, 1, 0.01)
    solution_set = []
    for i in initial_condition:
        solution = fsolve(det, i)
        if abs(det(solution)) < 1e-10:
            solution_set.append(solution)
    solution_set = np.unique(solution_set)

    return solution_set

def verify2(x, fx, fxt, Degree, gx_i, gx_j, a):
    tau, nu = x
    g = nu * imag - fx - fxt * np.exp(- nu * tau * imag) - Degree * gx_i + a * gx_j 
    g_real = np.real(g)
    g_imag = np.imag(g)
    return np.array([g_real, g_imag])

def verify3(network_type, arguments, N, beta, nu_set, tau_set, seed=0, d=None, low=0.1, high=10):
    a_set = verify1(network_type, N, beta, seed=0)
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    Degree = np.sum(A, 0)[0]
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, low, high, arguments, d)
    xs = np.mean(xs_high)

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2

    tau_all = []
    for a in a_set:
        tau_sol = np.ones((np.size(tau_set), np.size(nu_set))) * 10
        for tau, i in zip(tau_set, range(np.size(tau_set))):
            for nu, j in zip(nu_set, range(np.size(nu_set))):
                initial_condition = np.array([tau, nu])
                tau_solution, nu_solution = fsolve(verify2, initial_condition, args=(fx, fxt, Degree, gx_i, gx_j, a))
                g_real, g_imag = verify2(np.array([tau_solution, nu_solution]), fx, fxt, Degree, gx_i, gx_j, a)
                if abs(g_real) < 1e-2 and abs(g_imag) < 1e-2:
                    tau_sol[i, j] = tau_solution
        tau_all.append(np.unique(tau_sol[tau_sol>0]))
    return a_set, tau_all

def eigenvalue_Matrix(network_type, arguments, N, beta, seed, d, nu_set, tau_set, low=0.1, high=10):
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    Degree = np.sum(A, 0)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments, d)
    xs = xs_high

    fx = (1-xs/K) * (2*xs/C-1)
    fxt = -xs/K*(xs/C-1)
    gx_i = xs/(D + E*xs + H*xs) - E * xs**2 / (D + E*xs + H*xs)**2
    gx_j = xs/(D + E*xs + H*xs) - H * xs**2 / (D + E*xs + H*xs)**2
    eigenvalue_set = np.zeros((np.size(tau_set), np.size(nu_set)))
    for tau , i in zip(tau_set, range(np.size(tau_set))):
        for nu, j in zip(nu_set, range(np.size(nu_set))):
            M = np.diagflat(nu * imag - fx - fxt * np.exp(- nu * tau * imag) - Degree * gx_i) - A * gx_j 
            eigenvalue, eigenvector = np.linalg.eig(M)
            eigenvalue_set[i, j] = np.min(np.abs(eigenvalue))
            #print(np.sort(np.real(eigenvalue)), np.sort(np.imag(eigenvalue)))
    sns.heatmap(eigenvalue_set, vmin=np.min(eigenvalue_set), vmax=np.max(eigenvalue_set))
    
    return eigenvalue_set

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
    #eigenvalue, eigenvector = sparse_eig(M, k=1, which='SM', tol=1e-10)
    #eigenvalue, eigenvector = slin.eig(M)
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
    Degree_weighted = np.sum(A, 0)
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
            tau_solution, nu_solution = fsolve(eigenvalue_zero, initial_condition, args=(A, fx, fxt, Degree_weighted, gx_i, gx_j))
            eigen_real, eigen_imag = eigenvalue_zero(np.array([tau_solution, nu_solution]), A, fx, fxt, Degree_weighted, gx_i, gx_j)
            if abs(eigen_real) < 1e-5 and abs(eigen_imag) < 1e-5:
                tau_sol[i, j] = tau_solution
            t2=time.time()
            # print(tau, nu, t2-t1, tau_solution)
    return tau_sol


fs = 22
ticksize = 20
legendsize= 14
alpha = 0.8
lw = 3
marksize = 8

imag = 1j
N = 2500

tau_set = np.array([0.316])
nu_set = np.array([5.2])
tau_set = np.arange(0.28, 0.38, 0.01)
nu_set = np.arange(1, 10, 0.1)
tau_set = np.arange(0.2, 0.4, 0.1)
nu_set = np.arange(1, 10, 1)
arguments = (B, C, D, E, H, K)
degree = 4
seed = 2
d = 0.8
d = 3
d_set = [900]
d_set = [200, 400, 600, 900, 1600, 2500]
d_set = [5, 10, 15, 20, 25, 50]
d_set = [2, 2.5, 3, 3.5, 4, 4.5, 5]
d_set = [3]
#tau_solution = tau_multi_K_eigen(N, nu_set, tau_set, arguments, c, low=0.1, high=10)
network_type = 'real'
network_type = 'RR'
network_type = '2D'
network_type = 'BA'
network_type = 'ER'
network_type = 'SF'
seed_set = [2, 10, 4, 3, 13, 14, 9, 5, 7, 8, 6, 11, 12]
seed_set = np.arange(0, 100, 1).tolist()
beta_set = np.arange(1, 2, 1)
beta = 1
d1 = 0.32
d2 = 0
d3 = 0

for d in d_set:
    for seed in seed_set:
        tau_c = tau_multi_critical(network_type, N, arguments, beta_set, seed, nu_set = nu_set, tau_set = tau_set, d = d)

# tau1, tau2 = tau_critical(arguments, beta_set)

#dyn_all = evolution(network_type, N, beta, seed, arguments, d1, d2, d3, d)


network_set = ['2D', 'ER', 'BA', 'real']
network_set = ['ER']
N_set = [100]
#plot_network_effect(network_set, d_set, N_set, beta_set)

#plot_hetero_effect(network_type, d_set, N, beta)

'''
t1 = time.time()
eigenvalue_set = eigenvalue_Matrix(network_type, arguments, N, beta, seed, d, nu_set, tau_set)
t2 = time.time()
print(t2-t1)

for N in N_set:
    for seed in seed_set:
        tau_all = tau_multi_solution_pattern(network_type, N, arguments, beta_s:w
        et, seed,nu_set = nu_set, tau_set = tau_set, d = d)

t1 = time.time()
tau = tau_eigenvalue(network_type, N, beta, nu_set, tau_set, arguments, seed, d=d)
t2 = time.time()
print(t2-t1)


'''
