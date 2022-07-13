import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import itertools
import time 

fs = 22
ticksize = 16
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#fc8d62', '#66c2a5', '#8da0cb', '#a6d854',  '#e78ac3', '#ffd92f', '#e5c494', '#b3b3b3']) 




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

def stable_state(dynamics, beta_list, initial_low, initial_high):
    """TODO: Docstring for stable_state.

    :dynamics: TODO
    :returns: TODO

    """
    t = np.arange(0, 1000, 0.01)
    dynamics_function = globals()[dynamics + '_single']  
    xs_low_list = []
    xs_high_list = []
    for beta in beta_list:
        x_low = odeint(dynamics_function, initial_low, t, args=(beta, arguments))
        xs_low = x_low[-1, 0]
        x_high = odeint(dynamics_function, initial_high, t, args=(beta, arguments))
        xs_high = x_high[-1, 0]

        xs_low_list.append(xs_low)
        xs_high_list.append(xs_high)
    
    return xs_low_list, xs_high_list



a = 3
b = 1

dynamics = 'CW'
arguments = (a, b)
beta_list = np.arange(0, 20, 0.1)
initial_low = 0.01
initial_high = 10

xs_low, xs_high = stable_state(dynamics, beta_list, initial_low, initial_high)
plt.plot(beta_list, xs_low)
plt.plot(beta_list, xs_high)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.subplots_adjust(left=0.2, right=0.98, wspace=0.25, hspace=0.25, bottom=0.18, top=0.98)
plt.locator_params(nbins=5)
plt.xlabel('$\\beta$', fontsize=fs)
plt.ylabel('$x$', fontsize=fs)
plt.legend(fontsize=legendsize, frameon=False)
#plt.show()



