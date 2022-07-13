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

def load_data(net_type):
    """TODO: Docstring for gen_data.

    :arg1: TODO
    :returns: TODO

    """

    if net_type == 1 or net_type == 2:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/ANEMONE_FISH_WEBS_Coral_reefs2007.mat')

    elif net_type == 3 or net_type == 4:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_ANT_WEBS_Bluthgen_2004.mat')

    elif net_type == 5 or net_type == 6:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Clements_1923.mat')

    elif net_type == 7 or net_type == 8:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Elberling_1999.mat')

    elif net_type == 9 or net_type == 10:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_BEES_WASPS_Santos_2010.mat')

    elif net_type == 11 or net_type == 12:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_POLLINATOR_Robertson_1929.mat')

    elif net_type == 13 or net_type == 14:
        data = scipy.io.loadmat('/home/mac/RPI/research/unires/sim_matlab/NuRsE-master/figure2/M_real_data/PLANT_SEED_DISPERSER_Schleuning_2010.mat')

    else:
        print('wrong network type!')
        return None

    if net_type%2 == 1:
        A = data['A']  # adjacency matrix of plant network  
        M = data['M']
        N = np.size(A, 0)

    else:
        A = data['B']  # adjacency matrix of pollinator network  
        M = data['M']
        N = np.size(A, 0)

    return A, M, N

def A_from_data(net_type, M):
    """project network from bipartite network
    :net_type: if odd, construct adjacency network from 'A'; if even, construct adjacency network from 'B'
    :M: bipartite network
    :returns: project network for plant

    """
    m, n = M.shape
    M_1mn = M.reshape(1, m, n)  # reshape M to 3D matrix 
    if net_type == 1:
        M_nm1 = np.transpose(M_1mn, (2,1,0))  # M_nm1 is n * m * 1 matrix 
        M_3d = M_nm1 + M 
        M_0 = M_nm1 * M  # if the element of M_0 is 0, there is no common species shared with two plants 
        "suppose unweighted interaction network, which means it is blind for bees to choose which plant should pollinate."
        k = M.sum(-1)  # total node weight of species in the other network B 
        A = np.sum(M_3d * np.heaviside(M_0, 0) / k.reshape(m, 1), axis=1) 
    elif net_type == 0:
        M_nm1 = np.transpose(M_1mn, (1,2,0))  # M_nm1 is n * m * 1 matrix 
        M_3d = M_nm1 + M.T
        M_0 = M_nm1 * M.T  # if the element of M_0 is 0, there is no common species shared with two plants 
        k = M.sum(0)
        A = np.sum(M_3d * np.heaviside(M_0, 0) / k.reshape(n, 1), axis=1) 

    else:
        print('wrong net type')
        return None
    np.fill_diagonal(A, 0)  # set diagonal of adjacency matrix 0 
    return A 

def Gcc_A_mat(A, initial, remove):
    """find the survive nodes which is in giant connected components for a given adjacency matrix
    
    :A: TODO
    :returns: TODO

    """
    G = nx.from_numpy_matrix(A)
    G.remove_nodes_from(remove)
    # A_update = np.delete(A_copy, rm, 0)
    # A_update = np.delete(A_update, rm, 1)
    # initial_update = np.delete(initial, rm)
    "only consider giant connected component? "
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    survive = list(Gcc)
    A_update = A[survive, :][:, survive] 
    initial_update = initial[survive]
    return A_update, initial_update


def mutual_1D(x, t, c, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c * x**2 / (D + (E+H) * x)
    return dxdt

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

def mutual_multi(x, t, N, index_i, index_j, A_interaction, cum_index, arguments):
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
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])

    dxdt = sum_f + x * np.array([sum_g[i:j].sum() for i, j in zip(cum_index[:-1], cum_index[1:])])

    return dxdt

def stable_state(A, A_interaction, index_i, index_j, cum_index, low, high, arguments, d=None):
    """calculate stables states for a given interaction matrix and dynamics-main.mutual

    :A: Adjacency matrix
    :degree: degree of lattice 
    :returns: stable states for all nodes x_l, x_h

    """
    dynamics = mutual_multi
    t = np.arange(0, 1000, 0.01)
    N = np.size(A, 0)
    xs_low = odeint(dynamics, np.ones(N) * low, t, args=(N, index_i, index_j, A_interaction, cum_index, arguments))[-1]
    xs_high = odeint(dynamics, np.ones(N) * high, t, args=(N, index_i, index_j, A_interaction, cum_index, arguments))[-1]
    return xs_low, xs_high

def betaspace(A, x):
    """calculate  beta_eff and x_eff from A and x

    :A: adjacency matrix
    :x: state vector
    :returns: TODO

    """
    s_out = A.sum(0)
    s_in = A.sum(-1)
    if sum(s_out) == 0:
        return 0, x[0] 
    else:
        beta_eff = np.mean(s_out * s_in) / np.mean(s_out)
        if np.ndim(x) == 1:
            x_eff = np.mean(x * s_out)/ np.mean(s_out)
        elif np.ndim(x) == 2:  # x is matrix form 
            x_eff = np.mean(x * s_out, -1)/np.mean(s_out)
        return beta_eff, x_eff


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

def network_generate(network_type, N, beta, seed, d=None):
    """TODO: Docstring for network_generate.

    :arg1: TODO
    :returns: TODO

    """
    if network_type == '2D':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    elif network_type == 'RR':
        G = nx.random_regular_graph(d, N, seed)
    elif network_type == 'ER':
        #G = nx.fast_gnp_random_graph(N, d, seed)
        m = d
        G = nx.gnm_random_graph(N, m, seed)
    elif network_type == 'BA':
        m = d
        G = nx.barabasi_albert_graph(N, m, seed)
    elif network_type == 'real':
        A, M , N = load_data(seed)
        A = A_from_data(seed%2, M)
    if network_type != 'real':
        A = np.array(nx.adjacency_matrix(G).todense()) 
    else:
        if nx.is_connected(G) == False:
            print('more than one component')
            return None
    beta_eff, _ = betaspace(A, [0])
    weight = beta/ beta_eff
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    return A, A_interaction, index_i, index_j, cum_index

def tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d=None, low=0.1, high=10):
    """TODO: Docstring for character_multi.

    :x: TODO
    :tau: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, seed, d)
    Degree_weighted = np.sum(A, 0)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, low, high, arguments, d)
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
                print(tau, nu, det_real, det_imag, tau_solution, nu_solution)
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

def mutual_lattice_delay(f, t, d1, d2, d3, N, index, degree, A_interaction, arguments):
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
    x_tile = np.broadcast_to(x, (N,N))  # copy vector x to N rows
    x_j = x_tile[index].reshape(N, degree) # select interaction term j with i
    dxdt = B + x * (1 - xd1/K) * ( xd2/C - 1) +  x * np.sum(A_interaction * x_j / (D + E * x.reshape(N, 1) + H * x_j), -1)

    return dxdt

def mutual_multi_delay(f, t, d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments):
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

def tau_multi_critical(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_critical = np.zeros(np.size(beta_set))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        t1 = time.time()
        tau_sol = np.ravel(tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d))
        tau = min ( i for i in tau_sol if i>0)
        tau_critical[i] = tau
        t2 = time.time()
        print(i, t2 - t1)

    A, A_interaction, index_i, index_j, cum_inde = network_generate(network_type, N, 4, seed, d)
    A = np.heaviside(A, 0)
    degree = np.sum(A, 0)
    hetero = np.sum((degree - np.mean(degree))**2)/ N

    data = np.hstack((seed, np.mean(degree), hetero, tau_critical))
    data = pd.DataFrame(data.reshape(1, np.size(data)))
    data.to_csv('../report/report091620/' + network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None, index=None, mode='a')

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
    beta_eff, _ = betaspace(A, [0])
    weight = beta/ beta_eff
    A = A * weight

    g = lambda t: np.ones(N) * 5 
    t = np.arange(0, 100,0.01)
    dyn_all = ddeint(mutual_multi_delay, g, t, fargs=(d1, d2, d3, N, index_i, index_j, A_interaction, cum_index, arguments))
    plt.plot(t, np.mean(dyn_all, 1), alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$\\langle x \\rangle$', fontsize =fs)
    plt.show()

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
        data = np.array(pd.read_csv('../report/report091620/'+ network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None).iloc[:, :])
        seed = data[:, 0]
        degree.append(np.mean(data[:, 1]))
    degree = np.array(degree)
    degree_unique = np.unique(degree)
    for i in degree_unique:
        index = np.where(degree == i)[0]
        data = []
        for j in index:
            d = d_set[j]
            data.extend(np.array(pd.read_csv('../report/report091620/'+ network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None).iloc[:, :]))
        data = np.array(data)
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
    plt.legend( fontsize=legendsize, frameon=False)
    #plt.show()



fs = 22
ticksize = 20
legendsize= 15
alpha = 0.8
lw = 3
marksize = 8

imag = 1j
N = 400
tau_set = np.arange(0.12, 0.13, 0.002)
nu_set = np.arange(7, 8, 0.1)
tau_set = np.array([0.2848])
nu_set = np.array([5.6856])
tau_set = np.arange(0.1, 0.3, 0.02)
nu_set = np.arange(1, 10, 1)
arguments = (B, C, D, E, H, K)
degree = 4
seed = 5
d = 0.8
d = 21
d_set = [18, 21, 27]
#tau_solution = tau_multi_K_eigen(N, nu_set, tau_set, arguments, c, low=0.1, high=10)
network_type = 'real'
network_type = 'BA'
network_type = 'ER'
network_type = 'RR'
network_type = '2D'
seed_set = [2, 10, 4, 3, 13, 14, 9, 5, 7, 8, 6, 11, 12]
seed_set = np.arange(0, 10, 1).tolist()
beta_set = np.arange(1, 10, 1)
beta_set = np.arange(1, 2, 1)
beta = 1
d1 = 0.30
d2 = 0
d3 = 0
'''
for seed in seed_set:
    tau_c = tau_multi_critical(network_type, N, arguments, beta_set, seed, nu_set = nu_set, tau_set = tau_set, d = d)
'''
# tau1, tau2 = tau_critical(arguments, beta_set)

evolution(network_type, N, beta, seed, arguments, d1, d2, d3, d)


network_set = ['2D', 'ER', 'BA', 'real']
d_set = [0, 18, 6, 4]
N_set = [9, 9, 9, 9]
network_set = ['2D']
d_set = [2, 3, 4, 5, 6, 7]
d_set = [18, 21]
N_set = [400]
seed_set = [0]
#plot_network_effect(network_set, d_set, N_set, beta_set)
#plot_hetero_effect(network_type, d_set, N, beta)
'''
for N in N_set:
    for seed in seed_set:
        tau_all = tau_multi_solution_pattern(network_type, N, arguments, beta_set, seed,nu_set = nu_set, tau_set = tau_set, d = d)
'''
