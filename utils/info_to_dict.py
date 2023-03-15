import torch
import pandas as pd 

def info_to_dict():
    node_information = pd.read_csv("node_information.csv", sep=",", header=None)
    node_names = node_information[node_information.columns[0]]
    node_information.drop(node_information.columns[0], axis=1, inplace=True)
    node_data = torch.from_numpy(node_information.to_numpy()).to(torch.float32)
    node_dict = {}
    for i, idx in enumerate(node_names):
        node_dict[str(idx)] = node_data[i].to_sparse()
    return node_dict

