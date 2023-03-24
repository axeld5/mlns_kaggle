import numpy as np
import networkx as nx 

from sklearn.cluster import KMeans
from networkx.algorithms.community import greedy_modularity_communities

def get_community_based_pred(g, test_set):
    c = greedy_modularity_communities(g)
    comm_dict = get_community_dict(g, c)
    comm_pred = get_pred(comm_dict, test_set)
    return comm_pred

def get_cluster_based_pred(g, test_set, k=10):
    c = spectral_clustering(g, k)
    comm_dict = get_community_dict(g, c)
    comm_pred = get_pred(comm_dict, test_set)
    return comm_pred

def get_community_dict(g, c):
    comm_dict = {}
    for i, community in enumerate(c):  
        for node in g.nodes: 
            if node in community:
                comm_dict[node] = i
    return comm_dict

def get_pred(comm_dict, test_set):
    comm_pred = np.zeros(len(test_set))
    for i, elem in enumerate(test_set):
        src, tgt = elem[0], elem[1]
        comm_src = comm_dict[src]
        comm_tgt = comm_dict[tgt]
        if comm_src == comm_tgt:
            comm_pred[i] = 1
    return comm_pred

def spectral_clustering(G, k):
    nodelist = list(G)  
    A = np.array(nx.adjacency_matrix(G, nodelist).todense())
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals, eigvects = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    eigvects = eigvects[:,idx]
    X = eigvects[:, 0:k]
    kmeans = KMeans(n_clusters=k).fit(X)
    partition = [set() for _ in range(k)]
    for i in range(len(nodelist)):
        partition[kmeans.labels_[i]].add(nodelist[i])
    return partition