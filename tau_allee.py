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
from mutual_framework import ddeint_Cheng

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']) 
colors = iter(('#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494') )

fs = 35
ticksize = 25
legendsize= 25
alpha = 0.8
lw = 3
marksize = 8

def logistic_allee(x, t, arguments):
    """describe thCe derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    K, C = arguments
    dxdt = x * (K - x) * ( x - C)
    return dxdt

def logistic_allee_delay(f, x0, x, t, dt, d, arguments):
    """original dynamics N species interaction.

    :x: 1 dynamic variable
    :t: time series
    :returns: derivative of x 

    """
    K, C = arguments
    x = f[int(t/dt)]
    xd = np.where(t>d, f[int((t-d)/dt)], x0)
    dxdt = x * (K - xd) * ( x - C) 
    return dxdt


K = 1
C = 0.1
initial_condition = np.array([0.5])
dt = 0.01
t = np.arange(0, 1000, dt)
arguments = (K, C)
delay = 1.7
delay_list = [0.4, 1.7, 1.8][::-1]
for delay in delay_list:
    dyn_all = ddeint_Cheng(logistic_allee_delay, initial_condition, t, *(delay, arguments))
    plt.plot(t, dyn_all, linewidth=2, label=f'$\\tau={delay}$', color=next(colors))
    plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(axis='x', tight=True, nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.xlabel('$t$', fontsize= fs)
    plt.ylabel('$x$', fontsize =fs)
    plt.legend( fontsize=legendsize, frameon=False, markerscale=2)
    figname = f'x_t_logistic_allee_tau={delay}'
    plt.savefig('/home/mac/RPI/research/Candidacy_Exam/pre_v1/' + figname + '.svg' )
    plt.close()

#plt.show()
tau_c = np.pi / 2 / K / (1-C)
tau_c2 = 1 / np.e / K / (K-C)
