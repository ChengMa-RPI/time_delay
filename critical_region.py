import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp

from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core
import scipy.stats as stats

B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

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


fs = 24
ticksize = 20
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))



def xs_group_partition_bifurcation(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method, des):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """

    A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    if 'high' in dynamics:
        dynamics_multi = globals()[dynamics[:dynamics.find('_high')] + '_multi']
    else:
        dynamics_multi = globals()[dynamics + '_multi']

    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
        for j, m in enumerate(m_list):
            group_index = group_index_from_feature_Kmeans(feature, m)
            A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(N_actual), group_index)
            initial_condition_reduction_deg_part = np.ones(len(A_reduction_deg_part)) * attractor_value
            xs_reduction_deg_part = odeint(dynamics_multi, initial_condition_reduction_deg_part, t, args=(arguments, net_arguments_reduction_deg_part))[-1]
            data = np.hstack((weight, xs_reduction_deg_part))
            des_file = des + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
            df = pd.DataFrame(data.reshape(1, len(data)))
            df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def xs_multi_bifurcation(network_type, N, seed, d, weight_list, attractor_value, des):
    """TODO: Docstring for random_partition.

    :arg1: TODO
    :returns: TODO

    """
    if network_type == 'degree_seq':
        d_record = d[:10]
    else:
        d_record = d
    if network_type == 'net_deg':
        kmean, kmin, kmax, lim_increase, lim_decrease
        A_unit, A_interaction, index_i, index_j, cum_index = generate_random_graph(N, seed, beta_pres, d)
    else:
        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A_unit)
    N_actual = len(A_unit)
    k = np.sum(A_unit>0, 0)
    t = np.arange(0, 1000, 0.01)
    initial_condition = np.ones(N_actual) * attractor_value
    if 'high' in dynamics:
        dynamics_multi = globals()[dynamics[:dynamics.find('_high')] + '_multi']
    else:
        dynamics_multi = globals()[dynamics + '_multi']

    for i, weight in enumerate(weight_list):
        A = A_unit * weight
        net_arguments = (index_i, index_j, weight * A_interaction, cum_index)
        xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
        des_file = des + f'N={N}_d=' + str(d_record) + f'_seed={seed}.csv'
        data = np.hstack((weight, xs_multi))
        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return None

def generate_SF(N, seed, gamma, kmax, kmin):
    """generate scale-free network using configuration model with given gamma, kmin, kmax. 

    :N: TODO
    :seed: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :returns: TODO

    """
    p = lambda k: k ** (float(-gamma))
    k = np.arange(kmin, N, 1)
    pk = p(k) / np.sum(p(k))
    random_state = np.random.RandomState(seed[0])
    if kmax == N-1 or kmax == N-2:
        degree_seq = random_state.choice(k, size=N, p=pk)
    elif kmax == 0 or kmax == 1:
        degree_try = random_state.choice(k, size=1000000, p=pk)
        k_upper = int(np.sqrt(N * np.mean(degree_try)))
        k = np.arange(kmin, k_upper+1, 1)
        pk = p(k) /np.sum(p(k))
        degree_seq = random_state.choice(k, size=N, p=pk)

    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        degree_seq[-1] = np.random.RandomState(seed=seed[0]+N+i).choice(k, size=1, p=pk)

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
        if kmax == 1 or kmax == N-2:
            break
    A = nx.to_numpy_array(G)
    degrees = np.sum(A, 0)
    beta_cal = np.mean(degrees ** 2) / np.mean(degrees)
    h1 = beta_cal - np.mean(degrees)
    h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / np.mean(degrees)
    kmean = np.mean(degrees)
    return degrees, beta_cal, kmean, h1, h2

def generate_ER(N, seed, d):
    """TODO: Docstring for generate_ER.

    :N: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    G = nx.gnm_random_graph(N, d, seed)
    A = nx.to_numpy_array(G)
    degrees = np.sum(A, 0)
    beta_cal = np.mean(degrees ** 2) / np.mean(degrees)
    h1 = beta_cal - np.mean(degrees)
    h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / np.mean(degrees)
    kmean = np.mean(degrees)
    return degrees, beta_cal, kmean, h1, h2


def degree_pres_beta(N, seed, kmean, kmin, kmax, beta_pres, lim_increase, lim_decrease, error_tol):
    """TODO: Docstring for network_pres_beta.

    :N: TODO
    :kmean: TODO
    :beta_pres: TODO
    :returns: TODO

    """
    degree = np.ones(N) * kmean
    beta_cal = np.mean(degree ** 2) / np.mean(degree)
    error = np.abs(beta_pres - beta_cal) / beta_pres
    random_state = np.random.RandomState(seed)
    iter_num = 0
    while error > error_tol:
        iter_num += 1
        num_once = int(np.ceil(error * 20))
        degree_sort = np.sort(degree)
        degree_changeable = degree_sort[np.where((degree_sort < kmax) & (degree_sort > kmin ) )[0]]
        degree_kmin = degree_sort[np.where(degree_sort <= kmin)[0]]
        degree_kmax = degree_sort[np.where(degree_sort >= kmax)[0]]
        p_decrease = np.linspace(1, lim_decrease, len(degree_changeable))[::-1]
        p_increase = np.logspace(1, lim_increase, len(degree_changeable))
        node_increase = random_state.choice(len(degree_changeable), size=num_once, replace=True, p=p_increase/sum(p_increase))
        node_decrease = random_state.choice(len(degree_changeable), size=num_once, replace=True, p=p_decrease/sum(p_decrease))
        for i in node_increase:
            degree_changeable[i] += 1
        for i in node_decrease:
            degree_changeable[i] -= 1
        degree = np.hstack((degree_kmin, degree_kmax, degree_changeable))
        beta_cal = np.mean(degree ** 2) / np.mean(degree)
        error = np.abs(beta_pres - beta_cal) / beta_pres
    print(iter_num)
    return degree

def generate_random_graph(N, seed, beta_pres, d, error_tol=1e-3):
    """generate scale-free network using configuration model with given gamma, kmin, kmax. 

    :N: TODO
    :seed: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :returns: TODO

    """
    kmean, kmin, kmax, lim_increase, lim_decrease = d
    degree = degree_pres_beta(N, seed[0], kmean, kmin, kmax, beta_pres, lim_increase, lim_decrease, error_tol)
    degree_seq = np.array(degree, int)
    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        degree_seq[-1] = np.random.RandomState(seed=seed[1]+N+i).choice(k, size=1, p=pk)

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
        if kmax == 1 or kmax == N-2:
            break
    A = nx.to_numpy_array(G)

    if betaeffect:
        beta_eff, _ = betaspace(A, [0])
        weight = beta/ beta_eff
    else:
        weight = beta
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))

    return A, A_interaction, index_i, index_j, cum_index


def select_network_sample(network_type, N, seed, d, beta_pres, error_tol=0.03):
    """TODO: Docstring for select_network_sample.

    :network_type: TODO
    :N: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :seed: TODO
    :beta_pres: TODO
    :error_tol: TODO
    :returns: TODO

    """
    des = '../data/beta_pres_networks/'   
    des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
    if not os.path.exists(des):
        os.makedirs(des)
    data_collect = 0
    if network_type == 'SF':
        gamma, kmax, kmin = d
        degrees, beta_cal, kmean, h1, h2 = generate_SF(N, seed, gamma, kmax, kmin)
        data = np.hstack((gamma, seed[0], kmin, np.max(degrees), np.mean(degrees), h1, h2, beta_cal))

    if np.abs(beta_cal - beta_pres) / beta_pres < error_tol:
        data_collect = 1
        df = pd.DataFrame(data.reshape(1, len(data)))
        df.to_csv(des_file, index=None, header=None, mode='a')
    return data_collect

def xs_multi_network_sample(network_type, N, beta_pres, weight_list, attractor_value, generate_index_list):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/beta_pres_networks/'   
    des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
    net_data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    p = mp.Pool(cpu_number)

    des_save = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/' 
    if not os.path.exists(des_save):
        os.makedirs(des_save)


    if network_type == 'SF':
        gamma_list, seed_list, kmin_list, kmax_list, kmean_list, h1_list, h2_list, beta_cal_list = net_data.transpose()
        p.starmap_async(xs_multi_bifurcation, [(network_type, N, [int(seed), int(seed)], [gamma, N-1, int(kmin)], weight_list, attractor_value, des_save) for seed, gamma, kmin in zip(seed_list[generate_index_list], gamma_list[generate_index_list], kmin_list[generate_index_list] ) ]) .get()


    p.close()
    p.join()
 
    return None

def xs_multi_network_deg(network_type, N, beta_pres, weight_list, attractor_value, kmean_list, kmin_list, kmax_list, lim_increase_list, lim_decrease_list):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    kmean, kmin, kmax, lim_increase, lim_decrease
    p.starmap_async(xs_multi_bifurcation, [(network_type, N, seed, [kmean, kmin, kmax, lim_increase, lim_decrease], weight_list, attractor_value) for kmean, kmin, kmax, lim_increase, lim_decrease in zip(kmean_list, kmin_list, kmax_list, lim_increase_list, lim_decrease_list ) ]) .get()
    p.close()
    p.join()
 
    return None


def xs_group_network_sample(network_type, N, beta_pres, weight_list, m_list, attractor_value, space, tradeoff_para, method, generate_index_list):
    """TODO: Docstring for data_network_sample.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/beta_pres_networks/'   
    des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
    net_data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    p = mp.Pool(cpu_number)
    des_save = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans' 
    if method == 'kcore' or method == 'node_state':
        des_save += '/'
    elif method == 'degree':
        des_save += '_space=' + space + '/'
    elif method == 'kcore_degree' or method == 'kcore_KNN_degree':
        des_save += '_space=' + space + f'_tradeoffpara={tradeoff_para}/'
    if not os.path.exists(des_save):
        os.makedirs(des_save)

    if network_type == 'SF':
        gamma_list, seed_list, kmin_list, kmax_list, kmean_list, h1_list, h2_list, beta_cal_list = net_data.transpose()
        p.starmap_async(xs_group_partition_bifurcation, [(network_type, N, [int(seed), int(seed)], [gamma, N-1, int(kmin)], weight_list,  m_list, attractor_value, space, tradeoff_para, method, des_save) for seed, gamma, kmin in zip(seed_list[generate_index_list], gamma_list[generate_index_list], kmin_list[generate_index_list] ) ]) .get()
    p.close()
    p.join()
 
    return None



def critical_point(network_type, N, seed, d, weight_list, critical_type, threshold_value):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_multi = data[index, 1:]
    y_multi = np.array([betaspace(A, xs_multi[i])[-1] for i in range(len(weight_list))])
    #y_multi = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])
    if critical_type == 'survival_ratio':
        transition_index = np.where(np.sum(xs_multi > 5, 1) / N > threshold_value) [0][0]
    else:
        transition_index = np.where(y_multi > threshold_value)[0][0]
    critical_weight = weight_list[transition_index]
    return y_multi, critical_weight


def critical_point_plot(network_type, N, beta_pres, weight_list, plot_type, critical_type, threshold_value):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """
    des = '../data/beta_pres_networks/'   
    des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    if network_type == 'SF':
        gamma_list, seed_list, kmin_list, kmax_list, kmean_list, h1_list, h2_list, beta_cal_list = data.transpose()
        plot_index =np.arange(118)
        y_multi_list = np.zeros((len(plot_index), len(weight_list) )) 
        critical_w_list = np.zeros((len(plot_index)))

        if plot_type == 'h2':
            plot_list = h2_list
            xlabel = '$h_2$'
        elif plot_type == 'h1':
            plot_list = h1_list
            xlabel = '$h_1$'
        elif plot_type == 'kmean':
            plot_list = kmean_list
            xlabel = '$\\langle k \\rangle $'

        for i, gamma, kmin, seed in zip(range(len(plot_index)), gamma_list[plot_index], kmin_list[plot_index], seed_list[plot_index]):
            d = [gamma, N-1, int(kmin)]
            seed = [int(seed), int(seed)]
            y_multi, critical_w = critical_point(network_type, N, seed, d, weight_list, critical_type, threshold_value)
            critical_w_list[i] = critical_w
            y_multi_list[i] = y_multi
            #plt.plot(weight_list, y_multi, label=f'h={h2_list[plot_index[i]]}')

            kmin_unique = np.unique(kmin_list[plot_index])
            color_i = colors[np.where(kmin == kmin_unique)[0][0]]
            if i == np.where(kmin_list[plot_index] == kmin)[0][0]:
                plt.plot(plot_list[plot_index[i]], critical_w, '.', markersize=10, color=color_i, label='$k_{min}=$'+f'${int(kmin)}$')
            else:
                plt.plot(plot_list[plot_index[i]], critical_w, '.', markersize=10, color=color_i)


    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$w_c$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + plot_type + '_' + critical_type + f'={threshold_value}.png'
    plt.savefig(save_des, format='png')
    plt.close()

    return None

def network_parameter_relation(network_type, N, beta_pres, weight_list, plot_type, critical_type, threshold_value):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """
    des = '../data/beta_pres_networks/'   
    des_file =des + f'beta_pres={beta_pres}_' + network_type + f'_N={N}.csv'
    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    if network_type == 'SF':
        gamma_list, seed_list, kmin_list, kmax_list, kmean_list, h1_list, h2_list, beta_cal_list = data.transpose()

        plot_index =np.arange(75)

        for i, gamma, kmin, seed in zip(range(len(plot_index)), gamma_list[plot_index], kmin_list[plot_index], seed_list[plot_index]):
            d = [gamma, N-1, int(kmin)]
            seed = [int(seed), int(seed)]

            kmin_unique = np.unique(kmin_list[plot_index])
            color_i = colors[np.where(kmin == kmin_unique)[0][0]]
            if i == np.where(kmin_list[plot_index] == kmin)[0][0]:
                plt.plot(kmean_list[plot_index[i]], h1_list, '.', markersize=10, color=color_i, label='$k_{min}=$'+f'${int(kmin)}$')
            else:
                plt.plot(kmean_list[plot_index[i]], h1_list, '.', markersize=10, color=color_i)


    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$w_c$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + plot_type + '_' + critical_type + f'={threshold_value}.png'
    plt.savefig(save_des, format='png')
    plt.close()

    return None



    


 


N = 1000
network_type = 'SF'
kmax = N-1
kmin = 5
number_trial = 100
beta_pres = 20
gamma_list = [3.6, 3.7, 3.8, 3.9, 4]
    
for gamma in gamma_list:
    data_collects = 0
    for i in range(number_trial):
        seed = [i, i]
        d = [gamma, kmax, kmin]
        #data_collect = select_network_sample(network_type, N, seed, d, beta_pres, error_tol=0.03)
        #data_collects += data_collect
    if data_collects == 0:
        print(gamma)
        break

gamma_list = [2.1, 2.2, 2.3, 2.4, 2.3, 2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.8, 2.6, 2.7, 2.8, 2.9, 3]
kmin_list = [2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
seed_list = np.tile([93, 82, 61, 70, 74, 0, 69, 1, 60, 9, 10, 1, 80, 71, 44, 30, 1], (2, 1)).transpose().tolist()
kmax = 0
for gamma, kmin, seed in zip(gamma_list, kmin_list, seed_list):
    #degrees, beta_cal, kmean, h1, h2 = generate_SF(N, seed, gamma, kmax, kmin)
    #print(gamma, kmin, beta_cal, kmean, h1, h2)
    pass




dynamics = 'genereg'
arguments = (B_gene, )
attractor_value = 20



dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1

dynamics = 'CW'
arguments = (a, b)
attractor_value = 0

dynamics = 'CW'
arguments = (a, b)
attractor_value = 1000



beta = 1
betaeffect = 0 



network_type = 'SF'
space = 'log'
N = 1000


weight_list = np.round(np.arange(0.01, 0.4, 0.01), 5)
tradeoff_para = 0.5
method = 'degree'

gamma_list = [2.1, 2.2, 2.3, 2.4, 2.3, 2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.8, 2.6, 2.7, 2.8, 2.9, 3, 3.6, 5.6]
kmin_list = [2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 10, 15]
seed_list = np.tile([93, 82, 61, 70, 74, 0, 69, 1, 60, 9, 10, 1, 80, 71, 44, 30, 1, 47, 87], (2, 1)).transpose().tolist()

for gamma, kmin, seed in zip(gamma_list, kmin_list, seed_list):
    d = [gamma, 0, kmin]
    #xs_group_partition_bifurcation(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method)
    #xs_multi_bifurcation(network_type, N, seed, d, weight_list, attractor_value)
    pass

cpu_number = 4
weight_list = np.round(np.arange(0.01, 1, 0.01), 5)
generate_index_list = [0, 1, 2, 3]
#xs_multi_network_sample(network_type, N, beta_pres, weight_list, attractor_value, generate_index_list)
m_list = np.arange(1, 50, 1)
#xs_group_network_sample(network_type, N, beta_pres, weight_list, m_list, attractor_value, space, tradeoff_para, method, generate_index_list)
plot_type_list = ['h1', 'h2', 'kmean']
critical_type_list = ['survival_ratio', 'survival_ratio', 'yglobal', 'yglobal']
threshold_value_list = [0.5, 0.7, 2.5, 5]
for plot_type in plot_type_list:
    for critical_type, threshold_value in zip(critical_type_list, threshold_value_list):
        #critical_point_plot(network_type, N, beta_pres, weight_list, plot_type, critical_type, threshold_value)
        pass

"""
beta_list = np.zeros((number_trial))
kmean_list = np.zeros((number_trial))
h1_list = np.zeros((number_trial))
h2_list = np.zeros((number_trial))
kmax_list = np.zeros((number_trial))
for i in range(number_trial):
    seed = [i, i]
    degrees, beta_cal, kmean, h1, h2 = generate_SF(N, seed, gamma, kmax, kmin)
    beta_list[i] = beta_cal
    kmean_list[i] = kmean
    h1_list[i] = h1
    h2_list[i] = h2
    kmax_list[i] = np.max(degrees)
index = np.argsort(np.abs(beta_list - beta_pres))
"""

"""
kmin = 5
kmax = 200
kmean = 15
error_tol = 1e-3
lim_increase = 3
lim_decrease = 1.

seed = [0, 0]
#degrees, beta_cal, kmean, h1, h2 = generate_random_graph(N, seed, kmean, kmin, kmax, beta_pres, lim_increase, lim_decrease, error_tol)
network_type = 'net_deg'
kmean_list = [15, 15, 15, 15]
kmin_list = [2, 3, 4, 5]
kmax_list = [200, 200, 200, 200]
lim_increase_list = [9, 9, 9, 9]
lim_decrease_list = [1, 1, 1, 1]
beta_pres = 20
weight_list = np.round(np.arange(0.01, 0.4, 0.01), 5)
#xs_multi_network_deg(network_type, N, beta_pres, weight_list, attractor_value, kmean_list, kmin_list, kmax_list, lim_increase_list, lim_decrease_list)
"""

"""
for kmean, kmin, kmax, lim_increase, lim_decrease in zip(kmean_list, kmin_list, kmax_list, lim_increase_list, lim_decrease_list):
    d = [kmean, kmin, kmax, lim_increase, lim_decrease]
    A, _, _, _, _ = generate_random_graph(N, seed, beta_pres, d)
    degrees = np.sum(A, 0)
    h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / np.mean(degrees)
    des_xs_multi = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/'
    des_file = des_xs_multi + f'N={N}_d=' + str(d) + f'_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_multi = data[index, 1:]
    y_multi = np.array([betaspace(A, xs_multi[i])[-1] for i in range(len(weight_list))])
    plt.plot(weight_list, y_multi, label=f'h={round(h2, 3)}_kmin={kmin}')
    plt.legend()
"""
