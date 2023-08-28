import sys
sys.path.insert(1, '/home/mac/RPI/research/')
import scipy as sp
import numpy.random as rn
import numpy.linalg as la
import numpy as np
import sage.all as sg
import networkx as nx
#from mutual_framework import generate_SF


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
    return G




#A = [[0, 1], [0, 2], [1, 4]]
#net = NetworkIRR(A)
#x = net.get_adjacency_matrix()
#print(x)
A_matrix = [[0] + [1] * 10, [1, 0] + [1] * 9, [1, 1, 0, 0] + [1] * 5 + [0, 1], [1]*2+[0]*3+[1]*3+[0, 0, 1], [1]*3+[0]*2+[1]*3+[0, 1, 1], [1]*5+[0]+[1]*5, [1]*6+[0]+[1]*4, [1]*7 +[0]+[1]*3, [1]*3+[0]*2+[1]*3+[0, 1, 1], [1, 1, 0, 0]+[1]*5+[0, 1], [1]*10+[0]] 

N = 1000
m = 1000 * 4
seed = 100
G = nx.gnm_random_graph(N, m, seed)
m = 1
#G = nx.barabasi_albert_graph(N, m, seed)
#gamma = 2.5
#G = generate_SF(N, [4, 0], gamma, N-1, 1)
A_matrix = nx.to_numpy_array(G)
A = []
for i in range(len(A_matrix)):
    for j in range(i, len(A_matrix)):
        if A_matrix[i][j] != 0:
            A.append([i, j])


group, orbits = sg.Graph(A).automorphism_group(orbits=True)
print(len(orbits), np.mean(np.sum(A_matrix, axis=0)))
