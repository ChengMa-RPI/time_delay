import numpy as np
import networkx as nx 

N = 1000
gamma = 3.0
kmin = 3
seed = 0
kmax = int(kmin * N ** (1/(gamma-1)))

probability = lambda k: (gamma - 1) * kmin**(gamma-1) * k**(-gamma)
k_list = np.arange(kmin, 10 *kmax, 0.001)
p_list = probability(k_list)
p_list = p_list/np.sum(p_list)

degree_seq = np.array(np.round(np.random.RandomState(seed=seed).choice(k_list, size=N, p=p_list)), int)


G = nx.configuration_model(degree_seq, seed = 3)
G = nx.Graph(G)  # remove parallel edges
G.remove_edges_from(list(nx.selfloop_edges(G))) 
if nx.is_connected(G) == False:
    print('more than one component')
    G = G.subgraph(max(nx.connected_components(G), key=len))

A = np.array(nx.to_numpy_matrix(G))
A = np.array(nx.adjacency_matrix(G).todense()) 


