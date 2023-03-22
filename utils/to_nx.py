import networkx as nx 

from typing import List 

def set_to_nx(dataset:List[str]) -> nx.Graph:
    g = nx.Graph()
    for elem in dataset:
        g.add_node(elem[0])
        g.add_node(elem[1])
        if elem[2] == "1":
            g.add_edge(elem[0], elem[1])
    return g 

def set_to_directed_nx(dataset:List[str]) -> nx.Graph:
    g = nx.DiGraph()
    for elem in dataset:
        g.add_node(elem[0])
        g.add_node(elem[1])
        if elem[2] == "1":
            g.add_edge(elem[0], elem[1])
    return g 