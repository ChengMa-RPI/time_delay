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

from collections import Counter
from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core
import scipy.stats as stats
import time
from netgraph import Graph
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib import patches 

fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
color1 = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
color2 = ['brown', 'orange', 'lightgreen', 'steelblue','slategrey', 'violet']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

class Plot_Tau_C():
    def __init__(self, rows, cols,  network_type, dynamics, N, d, seed, weight, sharex=False, sharey=False, grid_pec_ratio=False):
        self.cols = cols
        self.rows = rows
        self.network_type = network_type
        self.dynamics = dynamics
        self.N = N
        self.d = d
        self.seed = seed
        self.weight = weight
        if grid_pec_ratio:
            fig, axes = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=(4 * cols, 3.5 * rows), gridspec_kw={'width_ratios':[1, 0.2, 1, 1, 1, 1]})

        else:
            fig, axes = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, figsize=(4 * cols, 3.5 * rows))
        self.fig = fig
        self.axes = axes


    def get_A_feature(self):
        """TODO: Docstring for get_A_feature.

        :arg1: TODO
        :returns: TODO

        """
        self.A_unit, self.A_interaction, self.index_i, self.index_j, self.cum_index = network_generate(self.network_type, self.N, 1, 0, self.seed, self.d)
        self.G = nx.from_numpy_array(self.A_unit)
        if network_type == 'SF':
            self.space = 'log'
        else:
            self.space = 'linear'
        tradeoff_para = 0.5
        method = 'degree'
        A = self.A_unit * self.weight
        feature = feature_from_network_topology(A, self.G, self.space, tradeoff_para, method)
        self.feature = feature

    def read_tau_c(self, method_type):
        """TODO: Docstring for read_data.

        :method_type: TODO
        :returns: TODO

        """
        filename = '../data/tau/' + self.dynamics + '/' + self.network_type + '/' + method_type + f'/N={self.N}_d={self.d}_seed={self.seed}_weight={self.weight}.csv'
        data = np.array(pd.read_csv(filename, header=None))
        m_list, tau = data.transpose()
        return m_list, tau

    def plot_tau_c_eigen_evolution(self, ax):
        simpleaxis(ax)
        m_eigen, tau_eigen = self.read_tau_c("eigen")
        m_evolution, tau_evolution = self.read_tau_c("evolution")
        ax.semilogx(m_eigen, tau_eigen, label = f'eigen', linewidth=2.5, alpha=0.8)
        ax.semilogx(m_evolution, tau_evolution, label = f'evolution', linewidth=2.5, alpha=0.8)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4) 

        return ax

    def read_xt_evolution(self, m, delay_list, detail=0):
        """TODO: Docstring for plot_xt_evolution.

        :ax: TODO
        :returns: TODO

        """
        if detail:
            des = '../data/tau/' + self.dynamics + '/' + self.network_type + '/evolution_detail_data_ygl/'
        else:
            des = '../data/tau/' + self.dynamics + '/' + self.network_type + '/evolution_data/'
        filename_part = f'm={m}_d={self.d}_seed={self.seed}_weight={self.weight}_delay='
        xs_diff_list = []
        if delay_list == None:
            delay_list = []
            for filename in os.listdir(des):
                if filename_part in filename:
                    start_index = filename.rfind("=")+1
                    end_index = filename.rfind(".")
                    delay = eval(filename[start_index:end_index])
                    delay_list.append(delay)
            delay_list = np.sort(delay_list)

        for delay in delay_list:
            filename = filename_part + f"{delay}.csv"

            data = np.array(pd.read_csv(des + filename, header=None)) 
            t, xs_delay = data[:, 0], data[:, 1:]
            xs = xs_delay[0] + 1e-2
            xs_diff = xs_delay - xs
            if not detail:
                xs_diff = np.abs(xs_diff)
            xs_diff_list.append(xs_diff)
        return delay_list, xs_diff_list, t

    def plot_xs_evolution(self, ax, m, delay_list, title_letter, title=True):
        """TODO: Docstring for plot_xs_evolution.

        :ax: TODO
        :returns: TODO

        """
        simpleaxis(ax)
        colors = color1 
        delay_list, xs_diff_list, t = self.read_xt_evolution(m, delay_list, detail=1)
        for (i, delay), xs_diff in zip(enumerate(delay_list), xs_diff_list):
            xs_diff_ave = np.abs(xs_diff)
            ax.semilogy(t[::1000], xs_diff_ave[::1000], linewidth=1.5, alpha=0.5, label = f'$\\tau={delay}$', color=colors[i])
        ax.annotate(f'({title_letter})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            if m != self.N:
                ax.set_title(f"$m={m}$", size=labelsize*0.5)
            else:
                ax.set_title(f"$m=N$", size=labelsize*0.5)

    def get_ygl_from_group(self, xs_group):
        """TODO: Docstring for get_ygl_from_group.

        :xs_group: TODO
        :m: TODO
        :returns: TODO

        """
        xs_group = np.vstack((xs_group))
        tlength, m = len(xs_group), len(xs_group[0])
        xs_individual = np.zeros((tlength, len(self.A_unit)) )
        group_index = group_index_from_feature_Kmeans(self.feature, m)

        for i, group_i in enumerate(group_index):
            xs_individual[:, group_i] = xs_group[:, i:i+1]
        y_gl = betaspace(self.A_unit * self.weight, xs_individual)[-1]
        return y_gl

    def plot_xs_evolution_m_N(self, ax, m_list, title_letter, delay=None):
        """TODO: Docstring for plot_xs_evolution.

        :ax: TODO
        :returns: TODO

        """
        simpleaxis(ax)
        colors = color2
        for i, m in enumerate(m_list):
            _, xs_m, t = self.read_xt_evolution(m, [delay])
            y_m = self.get_ygl_from_group(xs_m)
            if m == N:
                labels = f'$m=N$'
            else:
                labels = f'$m={m}$'
            ax.semilogy(t, y_m, linewidth=1.5, alpha=0.5, label = labels , color=colors[i])

        ax.annotate(f'({title_letter})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
        ax.set_title(f"$\\tau={delay}$", size=labelsize*0.5)

    def plot_xs_delay_m(self, m_evo, m_compare, delay_list):
        """TODO: Docstring for plot_xs_delay_m.

        :arg1: TODO
        :returns: TODO

        """
        self.get_A_feature()
        for i, m in enumerate(m_evo):
            ax = self.axes[i//self.cols, i%self.cols]
            self.plot_xs_evolution(ax, m, delay_list, title_letters[i])
        i += 1
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.33,0.4) ) 

        for j, delay in enumerate(delay_list):
            ax = self.axes[(i+j) // self.cols, (i+j)%self.cols]
            self.plot_xs_evolution_m_N(ax, m_compare, title_letters[i+j], delay)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.33,-0.03) ) 

        self.fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        self.fig.text(x=0.05, y=0.5, horizontalalignment='center', s="$\\Delta  y^{(\\mathrm{gl})}$", size=labelsize*0.6, rotation=90)
        self.fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_d={self.d}_seed={self.seed}_weight={self.weight}_y_delay_tau_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()
        
    def plot_tau_c_evolution_error(self, d_list, seed_list, weight_list):
        for k, d in enumerate(d_list):
            self.d = d
            ax2 = self.axes[k, -1]
            simpleaxis(ax2)
            error_list = []
            for i, seed in enumerate(seed_list):
                self.seed = seed
                ax = self.axes[k, i]
                simpleaxis(ax)
                for j, weight in enumerate(weight_list):
                    self.weight = weight
                    m_evolution, tau_evolution = self.read_tau_c("evolution")
                    ax.semilogx(m_evolution, tau_evolution, label = f'weight={weight}', linewidth=2.5, alpha=0.8)
                    error = np.abs(tau_evolution[:-1] - tau_evolution[-1]) / np.abs(tau_evolution[:-1] + tau_evolution[-1])
                    error_list.append(error)
                seq_order = k * self.cols + i
                ax.set_title(f"net {seq_order + 1}", size=labelsize*0.5)
                ax.annotate(f'({title_letters[seq_order]})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
            error_ave = np.mean(np.vstack((error_list)), 0)
            ax2.loglog(m_evolution[:-1], error_ave, linewidth=2.5, alpha=0.8, color='slategrey', linestyle='--')
            ax2.set_title(f"$\\gamma={d[0]}$", size=labelsize*0.5)
            ax2.annotate(f'({title_letters[k * self.cols + 3]})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(-1.43,1.19) ) 

        self.fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$m$", size=labelsize*0.6)
        self.fig.text(x=0.05, y=0.5, horizontalalignment='center', s="$\\tau_c$", size=labelsize*0.6, rotation=90)
        self.fig.text(x=0.735, y=0.46, horizontalalignment='center', s="Error $(\\tau_c)$", size=labelsize*0.6, rotation=90)
        self.fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        plt.savefig(save_des, format='png')
        plt.close()
        return None

    def plot_tau_c_error(self, d_list, seed_list, weight_list):
        """TODO: Docstring for plot_tau_c_error.

        :seed_list: TODO
        :weight_list: TODO
        :returns: TODO

        """
        for k, d in enumerate(d_list):
            self.d = d
            ax = self.axes[k, -1]
            simpleaxis(ax)
            error_list = []
            for i, seed in enumerate(seed_list):
                self.seed = seed
                for j, weight in enumerate(weight_list):
                    self.weight = weight
                    m_evolution, tau_evolution = self.read_tau_c("evolution")
                    error = np.abs(tau_evolution[:-1] - tau_evolution[-1]) / np.abs(tau_evolution[:-1] + tau_evolution[-1])
                    error_list.append(error)
            error_ave = np.mean(np.vstack((error_list)), 0)
                    
            ax.loglog(m_evolution[:-1], error_ave, linewidth=2.5, alpha=0.8, color='salmon')
            #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4) 

    def plot_tau_c_evo_one_net(self, ax, network_type, d, seed, weight):
        error_list = []
        "override network_type, d, seed, weight"
        self.network_type = network_type
        self.d = d
        self.seed = seed
        self.weight = weight
        m_evolution, tau_evolution = self.read_tau_c("evolution")
        ax.semilogx(m_evolution, tau_evolution, color='tab:grey', linewidth=1.5, alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        return m_evolution, tau_evolution

    def plot_tau_c_evo_diff_nets(self, network_type_list, d_list_list, seed_list_list, weight):
        """TODO: Docstring for plot_tau_c_evo_diff_nets.

        :arg1: TODO
        :returns: TODO

        """

        for (i, network_type), d_list, seed_list in zip(enumerate(network_type_list), d_list_list, seed_list_list):
            for j, d in enumerate(d_list):
                if self.rows == 1:
                    ax = self.axes[j]
                else:
                    ax = self.axes[i, j]
                simpleaxis(ax)
                if j == 0:
                    #ax.annotate(network_type, xy=(-0.5, 0.47), xycoords="axes fraction", size=labelsize*0.5)
                    pass
                if network_type == 'SF':
                    title = f'$\\gamma={d[0]}, $ ' +  '$k_{\\mathrm{min}}=$' + f'${d[-1]}$'
                else:
                    title = f'$\\langle k \\rangle = {int(2*d/self.N)}$'
                ax.set_title(title, size=labelsize*0.5)
                ax.annotate(f'({title_letters[i * self.cols + j]})', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
                tau_c = []
                for k, seed in enumerate(seed_list):
                    m_evolution, tau_evolution = self.plot_tau_c_evo_one_net(ax, network_type, d, seed, weight)
                    tau_c.append(tau_evolution)
                tau_c = np.vstack(( tau_c ))
                self.tau_m_diffseeds_errorbar(ax, m_evolution, tau_c)
        self.fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$m$", size=labelsize*0.7)
        self.fig.text(x=0.05, y=0.54, horizontalalignment='center', s="$\\tau_c$", size=labelsize*0.7)
        self.fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        plt.show()

    def tau_m_diffseeds_errorbar(self, ax, x, y):
        """TODO: Docstring for tau_m_diffseeds_errorbar.
        :returns: TODO

        """
        y_mean = np.mean(y, 0)
        y_std = np.std(y, 0)
        lower = y_mean - y_std
        upper = y_mean + y_std
        ax.plot(x, y_mean, color='tab:red', alpha=0.8, linewidth=1.5)
        #ax.plot(x, lower, color='tab:blue', alpha=0.5)
        #ax.plot(x, upper, color='tab:blue', alpha=0.5)
        ax.fill_between(x, lower, upper, color='tab:blue', alpha=0.5)
        
    def plot_ygl_dygl(self, ax, m, delay_list, title_letter, title, cmap):
        """TODO: Docstring for plot_ygl_dygl.

        :delay: TODO
        :returns: TODO

        """
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")

        delay_list, yms, t = self.read_xt_evolution(m, delay_list, detail=1)
        dt = t[1]- t[0]
        for i, delay in enumerate(delay_list):
            #y_m = self.get_ygl_from_group(xs_m[i])
            y_m = yms[i]
            dymdt = (y_m[1:] - y_m[:-1]) / dt
            simpleaxis(ax[i])
            y_infinite_index = np.where(y_m>100)[0]
            dy_infinite_index = np.where(dymdt>100)[0]
            if len(y_infinite_index):
                index = y_infinite_index[0]
                if len(dy_infinite_index):
                    index = min(dy_infinite_index[0], index)
                y_m = y_m[:index]
                dymdt = dymdt[:index]
                t_i = t[:index]
            else:
                y_m = y_m[:-1]
                t_i = t[:-1]

            interval = 10
            ax[i].scatter(y_m[::interval], dymdt[::interval], c=t_i[::interval], vmin=t[0], vmax=t[-1], cmap=cmap, s=2)
            ax[i].tick_params(axis='both', which='major', labelsize=13)
            if title:
                ax[i].set_title(f'$\\tau={delay}$', size=labelsize*0.5)
            ax[i].annotate('(' + title_letter + str(i) + ')', xy=(-0.1, 1.05), xycoords="axes fraction", size=labelsize*0.5)
            #x_start, y_start, x_end, y_end = find_pos_for_arror(y_m, dymdt)
            #helper_arrow(ax[i], x_start, y_start, x_end, y_end)



    def plot_ygl_delay_m_ygl_dygl(self, m_evo, delay_list):
        """TODO: Docstring for plot_ygl_delay_m_ygl_dygl.

        :arg1: TODO
        :returns: TODO

        """
        self.get_A_feature()
        sns_color = sns.color_palette("flare", as_cmap=True)
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        for i, m in enumerate(m_evo):
            self.plot_xs_evolution(self.axes[i, 0], m, delay_list, title_letters[i], title=False)
            if m == N:
                annotation = f'$m=N$'
            else:
                annotation = f'$m={m}$'
            self.axes[i, 0].annotate(annotation, xy=(-1.23, 0.5), xycoords="axes fraction", size=labelsize*0.6)
            self.axes[i, 1].set_axis_off()
            ax = self.axes[i, 2:]
            self.plot_ygl_dygl(ax, m, delay_list, title_letters[i], i==0, cmap)
        self.axes[3, 0].legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.63,-0.09) ) 
        cax = self.fig.add_axes([0.92, 0.2, 0.02, 0.6])
        vmin = 0
        vmax = 200
        points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax, cmap=cmap)
        self.fig.colorbar(points, cax=cax)
        cbar_ticks = cax.get_yticks()
        cax.get_yaxis().set_ticklabels(np.array((cbar_ticks), int), size=ticksize*0.8)
        cax.set_ylabel('$t$', fontsize=labelsize*0.8)

        self.fig.text(x=0.61, y=0.01, horizontalalignment='center', s="$\\Delta y^{\\mathrm{(gl)}}$", size=labelsize*0.7)
        self.fig.text(x=0.32, y=0.54, horizontalalignment='center', s="$\\frac{dy^{\\mathrm{(gl)}}}{dt}$", size=labelsize*0.7)
        self.fig.text(x=0.22, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.7)
        self.fig.text(x=0.12, y=0.54, horizontalalignment='center', s="$|\\Delta y^{\\mathrm{(gl)}}|$", size=labelsize*0.7)
        self.fig.subplots_adjust(left=0.17, right=0.9, wspace=0.35, hspace=0.35, bottom=0.1, top=0.95)
        plt.show()


def helper_arrow(ax, x_start, y_start, x_end, y_end):
    """TODO: Docstring for helper_arrow.

    :ax: TODO
    :direction: TODO
    :: TODO
    :returns: TODO

    """
    linewidth = 2
    ax.annotate('', xy=(x_end, y_end),
             xycoords='data',
             xytext=(x_start, y_start),
             textcoords='data',
             arrowprops=dict(arrowstyle= '-|>',
                 color='tab:grey',
                             lw=linewidth,
                             ls='-')
           )
    return ax

def find_pos_for_arror(y_m, dymdt):
    if np.abs(y_m[-1]) < 1e-3 and np.abs(dymdt[-1]) < 1e-3:
        #arrow at the start
        x_range = np.max(y_m) - np.min(y_m)
        y_range = np.max(dymdt) - np.min(dymdt)
        x_start = y_m[0] + x_range * 0.2
        index_start = np.where(np.abs(y_m - x_start ) < 1e-3)[0][0]
        y_start = dymdt[index_start] - y_range * 0.05
        #y_start = dymdt[0]
        x_end = x_start+ x_range * 0.4
        index_end = np.where(np.abs(y_m - x_end )  <1e-3)[0][0]
        y_end = dymdt[index_end]- y_range * 0.05
    else:
        x_start = 0
        y_start = 0
        x_end = 1
        y_end = 1
    return x_start, y_start, x_end, y_end






rows = 2
cols = 4
network_type = 'SF'
dynamics = 'mutual'
N = 1000
d = [2.5, 999, 3]
seed = [0, 0]
weight = 0.1
title_letters = list('abcdefghijklmnopqrstuvwxyz')


if __name__ == '__main__':
    
    ptc = Plot_Tau_C(rows, cols, network_type, dynamics, N, d, seed, weight, sharex=True, sharey=True, grid_pec_ratio=False)
    #ptc.plot_tau_c_eigen_evolution(ptc.axes)
    m_evo = [1, 4 ,16, 1000]
    m_compare = np.unique(np.array(np.round([(2**2) ** i for i in range(5)], 0), int) ).tolist() + [N]
    delay_list = [0.19, 0.22, 0.25, 0.28]
    #ptc.plot_xs_delay_m(m_evo, m_compare, delay_list)

    d_list = [[2.5,999, 3], [3, 999, 4]]
    seed_list = [[i, i] for i in range(3)]
    weight_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    #ptc.plot_tau_c_evolution_error(d_list, seed_list, weight_list)
    #ptc.plot_tau_c_error(d_list, seed_list, weight_list)
    network_type_list = ['ER']
    d_list_list = [[2000, 3000, 5000, 16000]]
    seed_list_list = [[i for i in range(10)]]
    network_type_list = ['SF', 'SF']
    d_list_list = [[[2.1, 999, 2], [2.5, 999, 3], [3, 999, 4], [3.8, 999, 5]], [[2.1, 999, 3], [2.5, 999, 4], [3, 999, 6], [3.8, 999, 7]]]
    seed_list_list = [[[i, i] for i in range(10)], [[i, i] for i in range(10)]]
    weight = 0.6
    ptc.plot_tau_c_evo_diff_nets(network_type_list, d_list_list, seed_list_list, weight)
    m_evo = [1, 4, 16, N]
    #m_evo = [1]
    delay_list = [0.19, 0.22, 0.25, 0.28]
    #ptc.plot_ygl_delay_m_ygl_dygl(m_evo, delay_list)
