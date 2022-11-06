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
import matplotlib.gridspec as gridspec
import itertools
import seaborn as sns
import multiprocessing as mp

from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core
import scipy.stats as stats
import time

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
ticksize = 18
labelsize = 30
anno_size = 14
subtitlesize = 15
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

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



def network_critical_point(network_type, N, seed, d, weight_list, critical_type, threshold_value):
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

def group_critical_point(network_type, N, seed, d, m, space, tradeoff_para, method, weight_list, critical_type, threshold_value):
    """TODO: Docstring for critical_point.

    :network_type: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, m)
    xs_reduction_multi = np.zeros((len(weight_list), N_actual))
    des_reduction = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/' + method + '_kmeans_space=' + space + '/'
    des_file = des_reduction + f'N={N}_d=' + str(d) + f'_number_groups={m}_seed={seed}.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
    xs_i = data[index, 1:]
    for i, group_i in enumerate(group_index):
        xs_reduction_multi[:, group_i] = np.tile(xs_i[:, i], (len(group_i), 1)).transpose()

    y_reduction = np.array([betaspace(A, xs_reduction_multi[i])[-1] for i in range(len(weight_list))])
    #y_multi = np.array([np.mean(xs_multi[i]) for i in range(len(weight_list))])
    if critical_type == 'survival_ratio':
        transition_index = np.where(np.sum(xs_reduction_multi > 5, 1) / N > threshold_value) [0][0]
    else:
        transition_index = np.where(y_reduction > threshold_value)[0][0]
    critical_weight = weight_list[transition_index]
    return y_reduction, critical_weight


def critical_region_plot(network_type_list, N, d_list, seed_list, m, weight_list_list, space_list, plot_type, critical_type, threshold_value):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """

    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']
    delta_wc = np.zeros((len(d_list)))
    h1_list = np.zeros((len(d_list)))
    h2_list = np.zeros((len(d_list)))
    kmean_list = np.zeros((len(d_list)))
    for i, network_type, d, seed, weight_list, space in zip(range(len(d_list)), network_type_list, d_list, seed_list, weight_list_list, space_list):
        y_multi_list = np.zeros((len(d_list), len(weight_list) )) 
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        degrees = np.sum(A, 0)
        beta_cal = np.mean(degrees ** 2) / np.mean(degrees)
        kmean = np.mean(degrees)
        h1 = beta_cal - kmean
        h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / kmean
        h1_list[i] = h1
        h2_list[i] = h2
        kmean_list[i] = kmean

        y_multi, wc_multi = network_critical_point(network_type, N, seed, d, weight_list, critical_type, threshold_value)
        y_reduction, wc_reduction = group_critical_point(network_type, N, seed, d, m, space, tradeoff_para, method, weight_list, critical_type, threshold_value)
        delta_wc[i] = np.abs(wc_multi - wc_reduction) / wc_multi
        #delta_wc[i] =  wc_multi
    if plot_type == 'h2':
        plot_list = h2_list
        xlabel = '$h_2$'
    elif plot_type == 'h1':
        plot_list = h1_list
        xlabel = '$h_1$'
    elif plot_type == 'kmean':
        plot_list = kmean_list
        xlabel = '$\\langle k \\rangle $'


    for i, d, seed, in zip(range(len(d_list)), d_list, seed_list):
        plt.plot(plot_list[i], delta_wc[i], '.', markersize=15, color=colors[0])

    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('$\\Delta w_c$', fontsize=fs)
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + plot_type + '_' + critical_type + f'={threshold_value}.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()

    return None

def critical_region_scatter(network_type_list, N, d_list, seed_list, m, weight_list_list, space_list, hetero_type, critical_type, threshold_value):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """

    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']
    delta_wc = np.zeros((len(d_list)))
    h1_list = np.zeros((len(d_list)))
    h2_list = np.zeros((len(d_list)))
    kmean_list = np.zeros((len(d_list)))
    for i, network_type, d, seed, weight_list, space in zip(range(len(d_list)), network_type_list, d_list, seed_list, weight_list_list, space_list):
        y_multi_list = np.zeros((len(d_list), len(weight_list) )) 
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        degrees = np.sum(A, 0)
        beta_cal = np.mean(degrees ** 2) / np.mean(degrees)
        kmean = np.mean(degrees)
        h1 = beta_cal - kmean
        h2 = np.sum(np.abs(degrees.reshape(len(degrees), 1) - degrees)) / N**2 / kmean
        h1_list[i] = h1
        h2_list[i] = h2
        kmean_list[i] = kmean

        #y_multi, wc_multi = network_critical_point(network_type, N, seed, d, weight_list, critical_type, threshold_value)
        y_reduction, wc_reduction = group_critical_point(network_type, N, seed, d, m, space, tradeoff_para, method, weight_list, critical_type, threshold_value)
        #delta_wc[i] = np.abs(wc_multi - wc_reduction) / wc_multi
        delta_wc[i] =  wc_reduction
    if hetero_type == 'h2':
        hetero_list = h2_list
        ylabel = '$h_2$'
    elif hetero_type == 'h1':
        hetero_list = h1_list
        ylabel = '$h_1$'

    xlabel = '$\\langle k \\rangle $'

    #plt.scatter(np.array(kmean_list) * np.array(delta_wc), np.array(hetero_list) * np.array(delta_wc), s=delta_wc * 20, color=colors[0])
    plt.scatter(np.array(kmean_list) * np.array(delta_wc), np.array(hetero_list) * np.array(delta_wc), s=delta_wc * 20)

    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + plot_type + '_' + critical_type + f'={threshold_value}.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()

    return kmean_list, hetero_list, delta_wc

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

def critical_boundary(dynamics, network_type, m, hetero_type, critical_type, threshold_value):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """

    colors = ['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99']
    des = '../data/' + dynamics  + '/' + network_type + '/xs_bifurcation/'
    if m == N:
        des_file = des + 'wc_multi/critical_type=' + critical_type + f'_threshold_value={threshold_value}.csv'
    else:
        des_file = des + 'wc_group/critical_type=' + critical_type + f'_threshold_value={threshold_value}_m={m}.csv'
    data = pd.read_csv(des_file, header=None)
    kmean, h1, h2, wc = np.array(data.loc[:, 2:]).transpose()
    if hetero_type == 'h2':
        hetero_list = h2 
        ylabel = '$h_2$'
    elif hetero_type == 'h1':
        hetero_list = h1 * wc
        ylabel = '$h_1$'

    xlabel = '$\\langle k \\rangle $'

    plt.scatter(np.array(kmean) * np.array(wc), hetero_list, alpha=0.1)

    plt.locator_params(nbins=4)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.18, top=0.95)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel(ylabel, fontsize=fs)
    plt.legend(fontsize=17, frameon=False, markerscale=1.5) 
    save_des = '../manuscript/dimension_reduction_v1_111021/' + dynamics + '_' + network_type +  '_beta_pres_critical_point/' +   dynamics + '_' + network_type + '_' + plot_type + '_' + critical_type + f'={threshold_value}.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()

    return kmean, h1, h2, wc

def df_select_d_sort_seed(d_df, seed_df, d_list):
    index_select = []
    if type(d_df[0]) == str:
        d_df = [eval(i) for i in d_df]
        seed_df = [eval(i) for i in seed_df]
    for d_i in d_list:
        index_i = []
        seed_i = []
        for (i, d_df_i), seed_df_i in zip(enumerate(d_df), seed_df):
            if d_df_i == d_i:
                index_i.append(i) 
                if type(seed_df_i) == list:
                    seed_i.append(seed_df_i[0])
                else:
                    seed_i.append(seed_df_i)
        index_sort = np.argsort(seed_i)
        index_select.extend(np.array(index_i)[index_sort])
    return index_select


def df_wc_N_m(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list):
    """TODO: Docstring for critical_point_samples.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :: TODO
    :returns: TODO
    """
    df = pd.DataFrame()
    for network_type, d_list in zip(network_type_list, d_list_list):
        des = '../data/' + dynamics  + '/' + network_type + '/xs_bifurcation/'
        for critical_type, threshold_values in zip(critical_type_list, threshold_value_list):
            if critical_type == 'survival_ratio':
                R_type = 's='
            else:
                R_type = 'y='
            for threshold_value in threshold_values:
                df_i = pd.DataFrame()
                des_multi_file = des + 'wc_multi/critical_type=' + critical_type + f'_threshold_value={threshold_value}.csv'
                des_multi_file = des + 'wc_multi_rdw=0.01/critical_type=' + critical_type + f'_threshold_value={threshold_value}.csv'
                data_multi = pd.read_csv(des_multi_file, header=None)
                d_multi, seed_multi = np.array(data_multi.iloc[:, :2]).transpose()
                index_multi = df_select_d_sort_seed(d_multi, seed_multi, d_list)
                data_multi = data_multi.iloc[index_multi]
                d_multi, seed_multi, kmean_multi, h1_multi, h2, wc_multi = np.array(data_multi).transpose()
                beta_multi = h1_multi + kmean_multi
                df_i['N'] = wc_multi
                df_i['network'] = network_type
                df_i['h1'] = h1_multi
                df_i['kmean'] = kmean_multi
                df_i['beta_log'] = np.log10(np.array(beta_multi, float) ) 
                df_i['R'] = R_type + str(threshold_value)
                df_i['d'] = d_multi
                df_i['seed'] = seed_multi
                for m in m_list:
                    des_group_file = des + 'wc_group/critical_type=' + critical_type + f'_threshold_value={threshold_value}_m={m}.csv'
                    des_group_file = des + 'wc_group_rdw=0.01/critical_type=' + critical_type + f'_threshold_value={threshold_value}_m={m}.csv'
                    data_group = pd.read_csv(des_group_file, header=None)
                    d_group, seed_group = np.array(data_group.iloc[:, :2]).transpose()
                    index_group = df_select_d_sort_seed(d_group, seed_group, d_list)
                    data_group = data_group.iloc[index_group]
                    d_group, seed_group, kmean_group, h1_group, h2, wc_group = np.array(data_group).transpose()
                    df_i[m] = wc_group
                df = pd.concat((df, df_i))
    return df 

def facet_scatter(x, y, c, **kwargs):
    kwargs.pop('color')
    plt.scatter(x, y, c=c, **kwargs)

def connect_line(x1, x2, y1, y2, **kwargs):
    plt.plot([x1, x2], [y1, y2], **kwargs)

def change_title_name(ax, title_letter, size_ratio=1):
    get_title = ax.get_title().split('|')
    R_type =  get_title[0].split('=')[-2].strip(' ') 
    R_value = get_title[0].split('=')[-1].strip(' ')
    rename_title = '$R_' + R_type + ' = ' + R_value + '$ ' + '|' + get_title[1]
    R_sci = [int(i) for i in '{:.0e}'.format(float(R_value)).split('e')]
    if R_type == 'y':
        color='tab:blue'
    else:
        color = 'tab:green'
    ax.set_title(f'({title_letter})' + '  ' + rename_title, size=subtitlesize*0.85*size_ratio, color=color)
    return None

def facetgrid_wc_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list, size_ratio=1):
    """TODO: Docstring for facetgrid_wc_N_m.

    :df: TODO
    :returns: TODO

    """
    df = df_wc_N_m(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)

    df = pd.melt(df, id_vars=['N', 'R', 'beta_log', 'h1', 'kmean'], value_vars=m_list, var_name='m', value_name='wc_group' )
    df['relative_error'] = np.array(np.abs(df['N'] - df['wc_group']) / (df['N'] + df['wc_group']), float )  
    df_error = df.pivot_table(values='relative_error', index=['m'], columns=['R'], aggfunc=np.mean)
    relative_error = [df_error.loc[j, critical_type[0] + f'={i}'] for critical_type, threshold_values in zip(critical_type_list, threshold_value_list) for i in threshold_values for j in m_list]

    wc_together = np.array(df[['N', 'wc_group']], float)

    g = sns.FacetGrid(df, row='R', col='m')
    vmin, vmax = np.min(df['beta_log']), np.max(df['beta_log'])
    cmap = sns.color_palette('flare', as_cmap=True)
    g = g.map(facet_scatter, 'N', 'wc_group', 'beta_log', alpha=0.1, vmin=vmin, vmax=vmax, cmap=cmap)
    cax = g.fig.add_axes([0.89, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    g.fig.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize)
    cax.set_ylabel('$\\beta(w=1)$', fontsize=labelsize)

    axes = g.fig.axes
    letters = list('abcdefghijklmnopqrstuvwxyz')[:int(len(axes)**0.5)]
    letters = list('abcdefghijklmnopqrstuvwxyz')[:len(m_list)]
    numbers = np.arange(1, int(len(axes)**0.5)+1).tolist()
    numbers = np.arange(1, len(threshold_value_list) * len(threshold_value_list[0])+1).tolist()
    title_list = itertools.product(numbers, letters)

    for ax, error, title_i in zip(axes, relative_error, title_list):
        ax.plot([np.min(wc_together), np.max(wc_together)], [np.min(wc_together), np.max(wc_together)], '--k', alpha=0.3)
        ax.text(0.6 * np.max(wc_together), 0.2 * np.max(wc_together), 'Err=' + '{:#.2g}'.format(error), size=anno_size *size_ratio)
        ax.tick_params(axis='both', which='major', labelsize=ticksize*size_ratio) 
        change_title_name(ax, title_i[1] + str(title_i[0]), size_ratio)
    xlabel = '$ w_c^{(original)} $'
    ylabel = '$ w_c^{(reduction)} $'
    g.set_axis_labels('', '')
    g.fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize, rotation=90)
    g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + ''.join(network_type_list) +  '_wc_N_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()
    return g

def facetgrid_beta_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list, size_ratio=1, title_modification=0):
    """TODO: Docstring for facetgrid_wc_N_m.

    :df: TODO
    :returns: TODO

    """
    df = df_wc_N_m(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)
    df = df.rename(columns={i: str(i) for i in m_list}) # change column name type
    m_new = [str(i) for i in m_list[1:]] + ['N']
    df = pd.melt(df, id_vars=['1', 'R', 'beta_log', 'h1', 'kmean', 'network'], value_vars=m_new, var_name='m', value_name='wc_group' )
    df['h1_1'] = df['h1'] * df['1']
    df['kmean_1'] = df['kmean'] * df['1']
    df['h1_m'] = df['h1'] * df['wc_group']
    df['kmean_m'] = df['kmean'] * df['wc_group']
    beta_approx = np.mean( (df['h1_1'] + df['kmean_1']) )
    df['distance'] = np.array(np.abs(df['1'] - df['wc_group'])*  (df['h1']** 2 + df['kmean']**2 ) **0.5/beta_approx , float)
    # calculate distance -- mean and std
    df_distance = df.pivot_table(values='distance', index=['m'], columns=['R'], aggfunc=np.mean)
    df_std = df.pivot_table(values='distance', index=['m'], columns=['R'], aggfunc=np.std)
    distance_mean = [df_distance.loc[j, critical_type[0] + f'={i}'] for critical_type, threshold_values in zip(critical_type_list, threshold_value_list) for i in threshold_values for j in m_new ]
    distance_std = [df_std.loc[j, critical_type[0] + f'={i}'] for critical_type, threshold_values in zip(critical_type_list, threshold_value_list) for i in threshold_values for j in m_new ]

    """
    g = sns.FacetGrid(df, row='R', col='m', hue='network')
    g = g.map(plt.scatter, 'kmean_1', 'h1_1', s=10, alpha=0.1)
    g = g.map(plt.scatter, 'kmean_m', 'h1_m', s=10, alpha=0.1)
    """
    g = sns.FacetGrid(df, row='R', col='m')
    vmin, vmax = np.min(df['beta_log']), np.max(df['beta_log'])
    cmap = sns.color_palette('flare', as_cmap=True)
    g = g.map(facet_scatter, 'kmean_1', 'h1_1', 'beta_log', alpha=0.1, vmin=vmin, vmax=vmax, cmap=cmap)
    g = g.map(facet_scatter, 'kmean_m', 'h1_m', 'beta_log', alpha=0.1, vmin=vmin, vmax=vmax, cmap=cmap)
    g = g.map(connect_line, 'kmean_m', 'kmean_1',  'h1_m', 'h1_1', linestyle='--', linewidth=0.5, color='k', alpha=0.05)
    cax = g.fig.add_axes([0.89, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    g.fig.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize)
    cax.set_ylabel('$\\beta (w=1)$', fontsize=labelsize)

    x_max = np.max(np.array(df[['kmean_1', 'kmean_m']]) ) 
    y_max = np.max(np.array(df[['h1_1', 'h1_m']]) ) 
    axes = g.fig.axes
    letters = list('abcdefghijklmnopqrstuvwxyz')[:int(len(axes)**0.5)]
    letters = list('abcdefghijklmnopqrstuvwxyz')[:len(m_list)]
    numbers = np.arange(1, int(len(axes)**0.5) + 1).tolist()
    numbers = np.arange(1 + title_modification, len(threshold_value_list) * len(threshold_value_list[0])+1 + title_modification).tolist()
    title_list = itertools.product(numbers, letters)

    for ax, d_i, std_i, title_i in zip(axes, distance_mean, distance_std, title_list):
        di_str = '{:#.2g}'.format(d_i) 
        decimal_num = len(''.join(di_str.split('.')[-1]))
        ax.text(0.45 * x_max, 0.8 * y_max, '$\\mathcal{l}=$' + di_str + '$\\pm ${0:#.{1}f}'.format(std_i, decimal_num), size=anno_size*size_ratio)
        ax.tick_params(axis='both', which='major', labelsize=ticksize*size_ratio) 
        change_title_name(ax, title_i[1] + str(title_i[0]), size_ratio)
    xlabel = '$ \\langle k \\rangle ^{\\mathrm{wt}} $'
    ylabel = '$ h^{\\mathrm{wt}} $'
    g.set_axis_labels('', '')
    g.fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize, rotation=90)
    g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + ''.join(network_type_list) + '_beta_wc_m_1.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()

    return g

def df_distance_error(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list):
    """TODO: Docstring for similarity_N_m.

    :dynamics: TODO
    :network_type: TODO
    :d: TODO
    :seed: TODO
    :: TODO
    :returns: TODO

    """
    df = df_wc_N_m(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)
    df = df.rename(columns={i: str(i) for i in m_list}) # change column name type
    m_new = [str(i) for i in m_list]
    beta_approx = ((df['h1'] + df['kmean']) * df['1']).mean()
    df = pd.melt(df, id_vars=['N', 'R', 'd', 'seed', 'beta_log', 'h1', 'kmean'], value_vars=m_new, var_name='m', value_name='wc_group' )
    df['h1_N'] = df['h1'] * df['N']
    df['kmean_N'] = df['kmean'] * df['N']
    df['h1_m'] = df['h1'] * df['wc_group']
    df['kmean_m'] = df['kmean'] * df['wc_group']
    df['distance'] = np.array(np.abs(df['N'] - df['wc_group'])*  (df['h1']** 2 + df['kmean']**2 ) **0.5/beta_approx , float)
    df['error'] = np.array(np.abs(df['N'] - df['wc_group'])/ (df['N'] + df['wc_group'] ), float)
    #df['distance'] = np.array((df['N'] - df['wc_group'])*  (df['h1']** 2 + df['kmean']**2 ) **0.5/beta_approx , float)
    #df['error'] = np.array((df['N'] - df['wc_group'])/ (df['N'] + df['wc_group'] ), float)
    """
    "parse gamma and kmin"
    kmin_list, gamma_list = [], []
    for i in df['d']:
        str2list = i[1:-1].split(',')
        gamma = float(str2list[0])
        kmin = int(str2list[-1])
        gamma_list.append(gamma)
        kmin_list.append(kmin)
    df['kmin'] = kmin_list
    df['gamma'] = gamma_list
    """

    return df

def plot_ave_distance_error_m(dynamics, network_type, m_list, critical_type_list, threshold_value_list, plot_type):
    """TODO: Docstring for plot_ave_distance_error_m.

    :arg1: TODO
    :returns: TODO

    """
    df = df_distance_error(dynamics, network_type, m_list, critical_type_list, threshold_value_list)
    plot_df = df.pivot_table(values=plot_type, index=['m'], columns=['R'], aggfunc=np.mean)
    if plot_type == 'distance':
        ylabel = 'distance $\\langle l \\rangle $'
    else:
        ylabel = 'Error'

    plot_df.index = plot_df.index.astype(int)
    plot_df.reset_index(inplace=True)
    plot_df.columns.name = None
    plot_df = pd.melt(plot_df, id_vars=['m'], value_vars=plot_df.columns[1:], var_name='R', value_name=plot_type)
    g = sns.scatterplot(data=plot_df, x='m', y=plot_type, hue='R', style='R', s=100)
    legend_markers, legend_labels = g.get_legend_handles_labels() # get legend makers and labels 
    g.legend(legend_markers, [f'$R_{legend_i}$' for legend_i in legend_labels], frameon=False, markerscale=1.2, fontsize=14)
    g.set(xscale='log', yscale='log')
    #g.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 2, 5)))
    #g.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    #g.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    g.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.tick_params(labelsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('$m$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type + '_' + plot_type +  '_m.png'
    plt.savefig(save_des, format='png')
    plt.close()
    #plt.show()
    return None

def plot_std_distance_error_m(dynamics, network_type, m_list, critical_type_list, threshold_value_list, plot_type):
    """TODO: Docstring for plot_ave_distance_error_m.

    :arg1: TODO
    :returns: TODO

    """
    if plot_type == 'distance':
        ylabel = 'distance $\\langle l \\rangle $'
    else:
        ylabel = 'Error'
    df = df_distance_error(dynamics, network_type, m_list, critical_type_list, threshold_value_list)
    df['m'] = df['m'].astype(int)
    df_stat = df[['m', plot_type]].groupby('m').describe().sort_values(by=['m'])
    std = df_stat[(plot_type, 'std')]
    mean = df_stat[(plot_type, 'mean')]
    g = sns.lineplot(data=df, x='m', y=plot_type, ci=None)
    g.fill_between(m_list, mean-std, mean+std, alpha=0.3)
    g.set(xscale='log')

    plt.tick_params(labelsize=16)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('$m$', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + network_type + '_' + plot_type +  '_m_std.png'
    plt.savefig(save_des, format='png')
    plt.close()
    #plt.show()
    return None

def facetgrid_distance_error_m(dynamics_list, network_type, m_list, critical_type_list, threshold_value_list_list):
    """TODO: Docstring for facetgrid_distance_error_m.

    :dynamics: TODO
    :network_type: TODO
    :returns: TODO

    """
    df = pd.DataFrame()
    for dynamics, threshold_value_list in zip(dynamics_list, threshold_value_list_list):
        df_dynamics = df_distance_error(dynamics, network_type, m_list, critical_type_list, threshold_value_list)[['R', 'm', 'distance', 'error']]
        df_dynamics['dynamics'] = dynamics
        df = pd.concat((df, df_dynamics))
    df = df.melt(id_vars=['dynamics', 'R', 'm'], value_vars=['distance', 'error'], var_name='plot_type', value_name='dis_err')
    df = df.pivot_table(values='dis_err', index=['m'], columns=['dynamics', 'R', 'plot_type'], aggfunc=np.mean)
    df.reset_index(inplace=True)
    df = pd.melt(df, id_vars=['m'], value_name='dis_err')
    df['m'] = df['m'].astype(int)
    df['dis_err'] = df['dis_err'].astype(float)

    g = sns.FacetGrid(df, row='plot_type', col='dynamics', hue='R', legend_out=False)
    g.map(sns.scatterplot, 'm', 'dis_err', style=df['R'])
    axes = g.fig.axes
    letters = list('abcdefghijklmnopqrstuvwxyz')[:int(len(axes))]
    numbers = np.arange(1, int(len(axes))+1).tolist()
    title_list = itertools.product(numbers, letters)
    for ax, title_i in zip(axes, letters):
        get_title = ax.get_title().split('|')
        plot_type =  get_title[0].split('=')[-1].strip(' ') 
        dynamics_name = get_title[1].split('=')[-1].strip(' ')
        rename_title = dynamics_name + ' | ' + plot_type  
        ax.set_title(f'({title_i})' + '  ' + rename_title, size=subtitlesize*0.8)
        ax.tick_params(axis='both', which='major', labelsize=ticksize*0.8) 
    g.set(xscale='log', yscale='log')
    xlabel = '$m$'
    ylabel = 'similarity value'
    g.set_axis_labels('', '')
    g.fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
    g.fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + network_type +  '_dis_err.png'
    plt.savefig(save_des, format='png')
    plt.close()
    #plt.show()
    return None

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def subplots_distance_error_m(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list):
    """TODO: Docstring for facetgrid_distance_error_m.

    :dynamics: TODO
    :network_type: TODO
    :returns: TODO

    """
    fig, axes = plt.subplots(2, len(dynamics_list), sharex=True, sharey=True, figsize=(3*len(dynamics_list), 2*3))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for (j, dynamics), threshold_value_list in zip(enumerate(dynamics_list), threshold_value_list_list):
        df = df_distance_error(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)[['R', 'm', 'distance', 'error']]
        for i, plot_type in enumerate(['distance', 'error']):
            if plot_type == 'distance':
                ylabel = 'Distance $\\langle l \\rangle $'
            else:
                ylabel = 'Error'
            ax = axes[i, j]
            simpleaxis(ax)
            ax.annotate(f'({letters[i * len(dynamics_list) + j]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
            plot_df = df.pivot_table(values=plot_type, index=['m'], columns=['R'], aggfunc=np.mean)
            plot_df.index = plot_df.index.astype(int)
            plot_df.reset_index(inplace=True)
            plot_df.columns.name = None
            plot_df = pd.melt(plot_df, id_vars=['m'], value_vars=plot_df.columns[1:], var_name='R', value_name=plot_type)
            sns.scatterplot(data=plot_df, x='m', y=plot_type, hue='R', style='R', s=100, ax=ax)
            legend_markers, legend_labels = ax.get_legend_handles_labels() # get legend makers and labels 
            if i == 0:
                ax.legend(legend_markers, [f'$R_{legend_i}$' for legend_i in legend_labels], ncol=2, frameon=False, markerscale=1.0, loc='lower left', fontsize=legendsize*0.5)
                if 'high' in dynamics or dynamics == 'genereg':
                    title_name = dynamics[:dynamics.find('_high')] + ' (H-->L)'
                else:
                    title_name = dynamics + ' (L-->H)'
                ax.set_title(title_name, size=labelsize*0.5)
            else:
                ax.get_legend().remove()

            ax.set(xscale='log', yscale='log')
            #ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, size=labelsize*0.6)
            ax.tick_params(labelsize=16*0.8)
    xlabel = '$ m $'
    ylabel = 'Similarity Score'
    #fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.7, rotation=90)
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    plt.subplots_adjust(left=0.12, right=0.98, wspace=0.25, hspace=0.25, bottom=0.12, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + ''.join(network_type_list) + '_subplots_dis_err.png'
    plt.savefig(save_des, format='png')
    plt.close()
    return None

def subplots_distance_error_m_individual(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list):
    """TODO: Docstring for facetgrid_distance_error_m.

    :dynamics: TODO
    :network_type: TODO
    :returns: TODO

    """
    fig, axes = plt.subplots(2, len(dynamics_list), sharex=True, sharey=True, figsize=(3*len(dynamics_list), 2*3))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for (j, dynamics), threshold_value_list in zip(enumerate(dynamics_list), threshold_value_list_list):
        df = df_distance_error(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)[['R', 'm', 'd', 'seed', 'beta_log', 'distance', 'error']]
        df['m'] = df['m'].astype(int)
        for i, plot_type in enumerate(['distance', 'error']):
            df_plot = df.pivot_table(values=plot_type, index=['d', 'seed', 'beta_log'], columns=['m'])
            df_plot.reset_index(inplace=True)
            beta_log = np.array(df_plot['beta_log'], float)
            df_plot = df_plot.transpose()[3:]
            data_plot = np.array(df_plot, float)
            if plot_type == 'distance':
                ylabel = 'Distance $\\langle l \\rangle $'
            else:
                ylabel = 'Error'
            ax = axes[i, j]
            simpleaxis(ax)
            ax.annotate(f'({letters[i * len(dynamics_list) + j]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
            #sns.scatterplot(data=df, x='m', y=plot_type, hue='beta_log', s=10, ax=ax)
            cmap = sns.color_palette('flare', as_cmap=True)
            
            for y, beta_i in zip(data_plot.transpose(), beta_log):
                ax.plot(m_list, y, '--', linewidth=0.1, color=cmap((beta_i-beta_log.min())/(beta_log.max() - beta_log.min())) )
            #lines = [list(zip(m_list, y)) for y in data_plot.transpose()]
            #colored_lines = mpl.collections.LineCollection(lines, array=beta_log, cmap=cmap, linewidths=0.5)
            #ax.add_collection(colored_lines)
            if i == 0:
                if 'high' in dynamics or dynamics == 'genereg':
                    title_name = dynamics[:dynamics.find('_high')] + ' (H-->L)'
                else:
                    title_name = dynamics + ' (L-->H)'
                ax.set_title(title_name, size=labelsize*0.5)

            ax.set(xscale='log')
            ax.set_yscale('symlog', linthreshy=1e-1)
            #ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, size=labelsize*0.6)
            ax.tick_params(labelsize=16*0.8)
    vmin, vmax = beta_log.min(), beta_log.max()
    cax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize)
    cax.set_ylabel('$\\beta(w=1)$', fontsize=labelsize)

    xlabel = '$ m $'
    ylabel = 'Similarity Score'
    #fig.text(x=0.01, y=0.5, verticalalignment='center', s=ylabel, size=labelsize*0.7, rotation=90)
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + ''.join(network_type_list) + '_subplots_dis_err.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    return None

def subplots_distance_error_std_m(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list):
    """TODO: Docstring for facetgrid_distance_error_m.

    :dynamics: TODO
    :network_type: TODO
    :returns: TODO

    """
    fig, axes = plt.subplots(2, len(dynamics_list), sharex=True, sharey=True, figsize=(3 * len(dynamics_list), 6))
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for (j, dynamics), threshold_value_list in zip(enumerate(dynamics_list), threshold_value_list_list):
        df = df_distance_error(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)[['R', 'm', 'distance', 'error']]
        df['m'] = df['m'].astype(int)
        for i, plot_type in enumerate(['distance', 'error']):
            df_stat = df[['m', plot_type]].groupby('m').describe().sort_values(by=['m'])
            std = df_stat[(plot_type, 'std')]
            mean = df_stat[(plot_type, 'mean')]
            if plot_type == 'distance':
                ylabel = 'Distance $\\langle l \\rangle $'
            else:
                ylabel = 'Error'
            ax = axes[i, j]
            simpleaxis(ax)
            ax.annotate(f'({letters[i * len(dynamics_list) + j]})', xy=(-0.1, 1.03), xycoords="axes fraction", size=labelsize*0.6)
            #sns.lineplot(data=df, x='m', y=plot_type, ci=None, ax=ax)
            ax.plot(m_list, mean)
            ax.fill_between(m_list, mean-std, mean+std, alpha=0.3)
            legend_markers, legend_labels = ax.get_legend_handles_labels() # get legend makers and labels 
            if i == 0:
                if 'high' in dynamics or dynamics == 'genereg':
                    title_name = dynamics[:dynamics.find('_high')] + ' (H-->L)'
                else:
                    title_name = dynamics + ' (L-->H)'
                ax.set_title(title_name, size=labelsize*0.5)

            ax.set(xscale='log')
            #ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel(ylabel, size=labelsize*0.6)
            ax.tick_params(labelsize=16*0.8)
    xlabel = '$ m $'
    fig.text(x=0.5, y=0.01, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
    plt.subplots_adjust(left=0.12, right=0.98, wspace=0.25, hspace=0.25, bottom=0.12, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + ''.join(network_type_list) +  '_subplots_dis_err_std_noabs.png'
    plt.savefig(save_des, format='png')
    plt.close()
    return None




def facetgrid_wc_compare_beta_hk(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list):
    """TODO: Docstring for facetgrid_wc_N_m.

    :df: TODO
    :returns: TODO

    """

    size_ratio = 0.8
    critical_type_list = ['ygl', 'survival_ratio']
    threshold_value_list = [[5], [0.5]]
    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 1], width_ratios=[1, 0.02])
    g0 = facetgrid_wc_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list, size_ratio)
    g1 = facetgrid_beta_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list, size_ratio, 2)
    mg0 = SeabornFig2Grid(g0, fig, gs[0, 0])
    mg1 = SeabornFig2Grid(g1, fig, gs[2, 0])

    cmap = sns.color_palette('flare', as_cmap=True)
    #cax = fig.add_subplot(gs[:, -1])
    cax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    points = plt.scatter([], [], c=[], vmin=0.03, vmax=np.log10(100), cmap=cmap)
    plt.colorbar(points, cax=cax)
    cbar_ticks = cax.get_yticks()
    cax.get_yaxis().set_ticklabels(np.array(10**(cbar_ticks), int), size=ticksize*size_ratio)
    cax.set_ylabel('$\\beta(w=1)$', fontsize=labelsize*size_ratio)
    fig.text(x=0.02, y=0.25, verticalalignment='center', s='$ h^{\\mathrm{wt}} $', size=labelsize*0.8, rotation=90)
    fig.text(x=0.51, y=0.01, horizontalalignment='center', s='$ \\langle k \\rangle ^{\\mathrm{wt}} $', size=labelsize*0.8)

    fig.text(x=0.02, y=0.75, verticalalignment='center', s='$ w_c^{(m)} $', size=labelsize*0.8, rotation=90)
    fig.text(x=0.5, y=0.50, horizontalalignment='center', s='$ w_c^{(N)} $', size=labelsize*0.8)

    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
    save_des = '../manuscript/dimension_reduction_v2_020322/' + dynamics + '_' + ''.join(network_type_list) +  '_wc_N_m.png'
    #plt.savefig(save_des, format='png')
    #plt.close()
    #plt.show()
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
        break

gamma_list = [2.1, 2.2, 2.3, 2.4, 2.3, 2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.8, 2.6, 2.7, 2.8, 2.9, 3]
kmin_list = [2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6]
seed_list = np.tile([93, 82, 61, 70, 74, 0, 69, 1, 60, 9, 10, 1, 80, 71, 44, 30, 1], (2, 1)).transpose().tolist()
kmax = 0
for gamma, kmin, seed in zip(gamma_list, kmin_list, seed_list):
    #degrees, beta_cal, kmean, h1, h2 = generate_SF(N, seed, gamma, kmax, kmin)
    #print(gamma, kmin, beta_cal, kmean, h1, h2)
    pass








dynamics = 'CW_high'
arguments = (a, b)
attractor_value = 1000


dynamics = 'CW'
arguments = (a, b)
attractor_value = 0

dynamics = 'genereg'
arguments = (B_gene, )
attractor_value = 100

dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_value = 0.1


beta = 1
betaeffect = 0 



network_type = 'SF'
space = 'log'
N = 1000


weight_list = np.round(np.arange(0.01, 1, 0.01), 5)
tradeoff_para = 0.5
method = 'degree'

gamma_list = [2.1, 2.2, 2.3, 2.4, 2.3, 2.4, 2.5, 2.6, 2.5, 2.6, 2.7, 2.8, 2.6, 2.7, 2.8, 2.9, 3, 3.6, 5.6]
kmin_list = [2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 10, 15]
seed_list = np.tile([93, 82, 61, 70, 74, 0, 69, 1, 60, 9, 10, 1, 80, 71, 44, 30, 1, 47, 87], (2, 1)).transpose().tolist()

m_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gamma_list = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
kmin_list = [3, 4, 5]
seed_list = np.arange(50) 

#for gamma, kmin, seed in zip(gamma_list, kmin_list, seed_list):
for gamma in gamma_list:
    for kmin in kmin_list:
        for seed in seed_list:
            d = [gamma, 999, kmin]
            #xs_group_partition_bifurcation(network_type, N, seed, d, weight_list, m_list, attractor_value, space, tradeoff_para, method)
            #xs_multi_bifurcation(network_type, N, seed, d, weight_list, attractor_value)
            pass

cpu_number = 4
weight_list = np.round(np.arange(0.01, 1.01, 0.01), 5)
generate_index_list = [6, 17, 22, 43, 60, 73]
generate_index_list = [6]
weight_list = [4]
#xs_multi_network_sample(network_type, N, beta_pres, weight_list, attractor_value, generate_index_list)
m_list = np.arange(1, 50, 1)
m_list = [5]
#xs_group_network_sample(network_type, N, beta_pres, weight_list, m_list, attractor_value, space, tradeoff_para, method, generate_index_list)
plot_type_list = ['h1', 'h2', 'kmean']
critical_type_list = ['survival_ratio', 'survival_ratio', 'yglobal', 'yglobal']
threshold_value_list = [0.5, 0.7, 2.5, 5]
for plot_type in plot_type_list:
    for critical_type, threshold_value in zip(critical_type_list, threshold_value_list):
        #critical_point_plot(network_type, N, beta_pres, weight_list, plot_type, critical_type, threshold_value)
        pass


weight_list = np.round(np.arange(0.01, 0.5, 0.01), 5)
d_list = [[2.5, 999, 3]] * 5
seed_list = np.tile(np.arange(5), (2, 1)).transpose().tolist()
network_type = 'ER'
space = 'linear'
d_list = [8000] * 5
seed_list = np.arange(5).tolist()


network_type_list = ['ER'] * 15
space_list = ['linear'] * 15
d_list = [2000] * 5 + [4000] * 5 + [8000] * 5
weight_list_list = [ np.round(np.arange(0.01, 1.5, 0.01), 5) ]  * 5 + [ np.round(np.arange(0.01, 1, 0.01), 5) ]  * 5 + [ np.round(np.arange(0.01, 0.5, 0.01), 5) ]  * 5
seed_list = np.arange(5).tolist() + np.arange(5).tolist() + np.arange(5).tolist() 

network_type_list = ['SF'] * 3000
space_list = ['log'] * 3000
d_list = sum([[[gamma, 999, kmin]] * 50 for gamma in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4] for kmin in [3, 4, 5]], [])
weight_list_list = [np.round(np.arange(0.01, 0.4, 0.01), 5) ] * 1500 + [np.round(np.arange(0.01, 1.01, 0.01), 5) ] * 1500
seed_list = [[i, i] for i in range(50)] * 60

network_type_list = ['ER'] * 15 + ['SF'] * 7 + ['SF'] * 6
space_list = ['linear'] * 15  + ['log'] * 7 + ['log'] * 6
d_list = [2000] * 5 + [4000] * 5 + [8000] * 5 + [[2.1, 999, 2]] * 2 + [[2.5, 0, 3]] * 5 + [[2.7, 999, 5]] + [[2.2, 999, 2]] + [[2.3, 999, 2]] + [[2.3, 999, 2]] + [[2.6, 999, 3]] + [[2.8, 999, 4]] 
weight_list_list = [ np.round(np.arange(0.01, 1.5, 0.01), 5) ]  * 5 + [ np.round(np.arange(0.01, 1, 0.01), 5) ]  * 5 + [ np.round(np.arange(0.01, 0.5, 0.01), 5) ]  * 5 + [np.round(np.arange(0.01, 0.6, 0.01), 5) ]  * 7 +  [np.round(np.arange(0.01, 0.4, 0.01), 5) ]  * 6   
seed_list = np.arange(5).tolist() + np.arange(5).tolist() + np.arange(5).tolist() + np.tile(np.arange(2), (2, 1)).transpose().tolist() + np.tile(np.arange(5), (2, 1)).transpose().tolist() + [[61, 61]] + [[9, 9]] +  [[31, 31]] + [[51, 51]]  + [[45, 45]] + [[50, 50]]



network_type = 'SF'
plot_type = 'h1'
m = N
critical_type = 'ygl'
threshold_value = 5
#critical_region_plot(network_type_list, N, d_list, seed_list, m, weight_list_list, space_list, plot_type, critical_type, threshold_value)
hetero_type = 'h1'

#kmean, h1, h2, wc = critical_boundary(dynamics, network_type, m, hetero_type, critical_type, threshold_value)


network_type = 'SF'
network_type = 'ER'
network_type_list = ['SF']
gamma_list = [round(i, 1) if i%1 > 1e-5 else int(i) for i in np.hstack(( np.arange(2.1, 5.01, 0.2), np.arange(6, 20.1, 1) )) ]
kmin_list = [3, 4, 5]
d_SF = [[gamma, N-1, kmin] for gamma in gamma_list for kmin in kmin_list]

kmean_half = np.arange(2, 11)
d_ER = sum([ [i * N] * 10 for i in kmean_half ], [] )
network_type_list = ['SF']
d_list_list = [d_SF]


critical_type_list = ['ygl', 'survival_ratio']
m_list = [1, 3, 5, 10, 20, 50]
m_list = [1, 2, 4]
m_list = [1, 2, 4, 8, 16, 32]




dynamics = 'genereg'
threshold_value_list = [[0.001, 0.01, 0.1], [0.3, 0.5, 0.7]]

dynamics = 'CW'
threshold_value_list = [[1, 3, 5], [0.3, 0.5, 0.7]]

dynamics = 'mutual'
threshold_value_list = [[1, 3, 5], [0.3, 0.5, 0.7]]


threshold_value_list_list = [[[1, 3, 5], [0.3, 0.5, 0.7]], [[1, 3, 5], [0.3, 0.5, 0.7]], [[0.001, 0.01, 0.1], [0.3, 0.5, 0.7]], [[1, 3, 5], [0.3, 0.5, 0.7]]]
dynamics_list = ['mutual', 'CW', 'genereg', 'CW_high']
for dynamics, threshold_value_list in zip(dynamics_list[:1], threshold_value_list_list[:1]):
    #facetgrid_wc_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)
    #facetgrid_beta_N_m_list(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)
    facetgrid_wc_compare_beta_hk(dynamics, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list)
    pass


m_list = np.hstack(( np.arange(1, 51),  np.unique(np.array([1.21 ** i for i in range(37)], int) )[16:] )) 
m_list = np.unique(np.array([1.21 ** i for i in range(37)], int) )
m_list = np.unique(np.array(np.round([(2**0.25) ** i for i in range(40)], 0), int) )
plot_type = 'error'
plot_type = 'distance'


threshold_value_list_list = [[[1, 3, 5], [0.3, 0.5, 0.7]], [[1, 3, 5], [0.3, 0.5, 0.7]], [[0.001, 0.01, 0.1], [0.3, 0.5, 0.7]], [[1, 3, 5], [0.3, 0.5, 0.7]]]
dynamics_list = ['mutual', 'CW', 'genereg', 'CW_high']
#facetgrid_distance_error_m(dynamics_list, network_type, m_list, critical_type_list, threshold_value_list_list)
#plot_ave_distance_error_m(dynamics, network_type, m_list, critical_type_list, threshold_value_list, plot_type)
#plot_std_distance_error_m(dynamics, network_type, m_list, critical_type_list, threshold_value_list, plot_type)


#subplots_distance_error_m(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list)

critical_type_list = ['survival_ratio'] 
threshold_value_list_list = [ [[0.5]], [[0.5]], [[0.5]], [[0.5]] ] 
#subplots_distance_error_std_m(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list)
gamma_list = [round(i, 1) if i%1 > 1e-5 else int(i) for i in np.hstack(( np.arange(2.1, 5.01, 0.2) , np.arange(6, 20.1, 2) )) ]
kmin_list = [3, 4, 5]
d_SF = [[gamma, N-1, kmin] for gamma in gamma_list for kmin in kmin_list]
d_list_list = [d_SF]
#subplots_distance_error_m_individual(dynamics_list, network_type_list, d_list_list, m_list, critical_type_list, threshold_value_list_list)
