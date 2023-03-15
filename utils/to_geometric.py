import pandas as pd 
import torch
from torch_geometric.utils.convert import from_networkx

from .loader import load_set
from .to_nx import set_to_nx

def to_geometric(train:bool=True):
    set = load_set(train)
    g = set_to_nx(set)
    data = from_networkx(g)
    node_information = pd.read_csv("node_information.csv", sep=",", header=None)
    node_information.drop(node_information.columns[0], axis=1, inplace=True)
    node_data = torch.from_numpy(node_information.to_numpy()).to(torch.float32)
    data.x = node_data 
    return data