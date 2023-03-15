import random
import numpy as np 

def generate_random_walk(graph, root, L):
    """
    :param graph: networkx graph
    :param root: the node where the random walk starts
    :param L: the length of the walk
    :return walk: list of the nodes visited by the random walk
    """
    walk = [root]
    current = root 
    for l in range(L-1):
      next = np.random.choice(np.array(list(graph.neighbors(current))))
      walk = walk + [next]
      current = next
    return walk

def deep_walk(graph, N, L):
    '''
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    '''
    walks = []
    for n in range(N):
      O = list(graph.nodes())
      random.shuffle(O)
      for v in O:
        current_walk = generate_random_walk(graph, v, L)
        walks.append(current_walk)
    return walks

