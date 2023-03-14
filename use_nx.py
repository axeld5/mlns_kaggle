import csv
import networkx as nx 
import pandas as pd 

from utils.loader import load_set
from utils.to_nx import set_to_nx

if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    print(g.number_of_nodes())
    print(g.number_of_edges())