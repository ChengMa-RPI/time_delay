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

# R, K, C, alpha, beta, D, E, H = symbols('R K C alpha beta D E H ')
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

def eigenfunction(x, eigen, fx, fxt, degree, gx_i):
    """TODO: Docstring for eigenfuntion.

    :tau: TODO
    :nu: TODO
    :: TODO
    :returns: TODO

    """
    tau, nu = x
    Re = -fx - fxt * np.cos(nu * tau) - degree* gx_i
    Im = nu + fxt * np.sin(nu * tau)
    f1 = np.real(eigen) - Re
    f2 = np.imag(eigen) - Im
    return np.array([f1, f2])

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
    if network_type == 'SL':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    elif network_type == 'RR':
        G = nx.random_regular_graph(d, N, seed)
    elif network_type == 'ER':
        p = d 
        G = nx.fast_gnp_random_graph(N, p, seed)
    elif network_type == 'BA':
        m = d
        G = nx.barabasi_albert_graph(N, m, seed)
    A = np.array(nx.adjacency_matrix(G).todense()) 
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

    tau_sol = np.ones((np.size(tau_set), np.size(nu_set)))
    for tau, i in zip(tau_set, range(np.size(tau_set))):
        for nu, j in zip(nu_set, range(np.size(nu_set))):
            initial_condition = np.array([tau, nu])
            tau_solution, nu_solution = fsolve(matrix_variable, initial_condition, args=(A, fx, fxt, Degree_weighted, gx_i, gx_j))
            det_real, det_imag = matrix_variable(np.array([tau_solution, nu_solution]), A, fx, fxt, Degree_weighted, gx_i, gx_j)
            if abs(det_real) < 1e-2 and abs(det_imag) < 1e-2:
                tau_sol[i, j] = tau_solution
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

def coeff_K(arguments, c, low=0.1, high=10):
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
    beta = c
    P = (beta * E * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs) + (beta * H * xs * xs)/(D + E * xs + H * xs)**2 - (beta * xs)/(D + E * xs + H * xs) -(1-xs/K) * (2*xs/C-1)
    Q = xs/K*(xs/C-1)
    tau = np.arccos(-P/Q) /Q/np.sin(np.arccos(-P/Q)) 
    return P, Q, tau

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

def tau_critical(network_type, N, arguments, beta_set, seed, d=None, nu_set=None, tau_set=None, low=0.1, high=10):
    """TODO: Docstring for tau_critical.

    :arg1: TODO
    :returns: TODO

    """
    tau_critical = np.zeros(np.size(beta_set))
    for beta, i in zip(beta_set, range(np.size(beta_set))):
        #P, Q, tau = coeff_K(arguments, c, low=0.1, high=10)
        t1 = time.time()
        tau_sol = np.ravel(tau_multi_K_eigen(network_type, N, beta, nu_set, tau_set, arguments, seed, d))
        tau = min ( i for i in tau_sol if i>0)
        tau_critical[i] = tau
        t2 = time.time()
        print(i, t2 - t1)
    data = np.hstack((seed, tau_critical))
    data = pd.DataFrame(data.reshape(1, np.size(data)))
    data.to_csv('../report/report011720/' + network_type + f'_N={N}_d=' + str(d).replace('.', '') + '_logistic.csv', header=None, index=None, mode='a')
    plt.plot(beta_set, tau_critical, linewidth=lw, alpha = alpha)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.show()
    return tau_critical

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
        data = np.array(pd.read_csv('../report/report011720/'+ network_type + f'_N={N}_d=' +str(d).replace('.', '') + '_logistic.csv', header=None).iloc[:, :])
        plt.plot(beta_set, data[0, 1:], '-o', linewidth=lw, alpha = alpha, label=network_type)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\beta_{eff}$', fontsize= fs)
    plt.ylabel('$\\tau_c$', fontsize =fs)
    plt.legend(fontsize=legendsize)
    plt.show()


def allee_effect(f, t, d, beta):
    """TODO: Docstring for fallee_effect.

    :arg1: TODO
    :returns: TODO

    """
    x = f(t)
    xd = f(t-d)
    dxdt = x * ( -beta + (1 + beta)*x - xd * x)
    return dxdt

fs = 22
ticksize = 20
legendsize= 15
alpha = 0.8
lw = 3
marksize = 8

g = lambda t: np.array([0.8, 0.8])
t = np.arange(0, 100, 0.01)
beta = 0.2
d = 2.25
dyn_all = ddeint(allee_effect, g, t, fargs=(d, beta))
plt.plot(t, dyn_all, alpha = alpha)
plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.xlabel('$t$', fontsize= fs)
plt.ylabel('$\\langle x \\rangle$', fontsize =fs)
plt.show()





