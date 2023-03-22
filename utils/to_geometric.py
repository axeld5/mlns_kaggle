import pandas as pd 
import networkx as nx 
import torch
from torch_geometric.utils.convert import from_networkx

from .loader import load_set
from .to_nx import set_to_nx
from .info_to_dict import info_to_dict


def info_to_torch():
    node_information = pd.read_csv("node_information.csv", sep=",", header=None)
    node_information.drop(node_information.columns[0], axis=1, inplace=True)
    node_data = torch.from_numpy(node_information.to_numpy()).to(torch.float32)
    return node_data

def to_geometric(train:bool=True):
    set = load_set(train)
    g = set_to_nx(set)
    node_dict = info_to_dict()
    nx.set_node_attributes(g, values=node_dict, name="feat")    
    data = from_networkx(g, group_node_attrs=["feat"])
    return data