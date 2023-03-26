import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.to_geometric import to_geometric

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--scaling_factor', type=float, default=1)
parser.add_argument('--training_rate', type=float, default=0.8) 
args = parser.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = to_geometric(train=True)
data.x = data.x.to_dense()
data = T.NormalizeFeatures()(data)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=10, alpha=0.15)

    def forward(self, x, edge_index):
        if args.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x

        if args.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_

        return x

class Decoder(torch.nn.Module):
    def __init__(self, h_feats, edge_index):
        super(Decoder, self).__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
    
    def forward(self, z, edge_index, sigmoid=False):
        avg = (z[edge_index[0]]+z[edge_index[1]])/2
        var = (z[edge_index[0]]-z[edge_index[1]])**2
        h = torch.cat([avg, var], 1)
        return torch.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)

channels = args.channels
train_rate = args.training_rate
val_ratio = 0
test_ratio = (1-args.training_rate)
data = train_test_split_edges(data.to(dev), val_ratio=val_ratio, test_ratio=test_ratio)

N = int(data.x.size()[0])
if args.model == 'GNAE':   
    model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
if args.model == 'VGNAE':
    model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

config = {
    "ray": True,
    "verbose": False,
    "data": data,
    "max_epochs":300,
    "save": True,
    "model": "VGNAE",
    "channels": 64,
    "scaling": 0
}

x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

train_set = load_set(train=True)

non_edges = get_non_edges(train_set)
non_edge_indexes = torch.zeros([2, len(non_edges)])
convert_dict = {}
g = set_to_nx(train_set)
for i, node in enumerate(g.nodes): 
    convert_dict[node] = i
for i, elem in enumerate(non_edges):
    non_edge_indexes[0][i] = convert_dict[elem[0]]
    non_edge_indexes[1][i] = convert_dict[elem[1]]
data.test_neg_edge_index = non_edge_indexes.to(torch.long).to(dev)
 
def train():
    model.train()
    optimizer.zero_grad()
    z  = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

for epoch in range(1,args.epochs):
    loss = train()
    loss = float(loss)
    
    with torch.no_grad():
        test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

test_set = load_set(False)
test_indexes = torch.zeros(2, len(test_set))
for i, elem in enumerate(test_set):
    test_indexes[0][i] = convert_dict[elem[0]]
    test_indexes[1][i] = convert_dict[elem[1]]
test_indexes = test_indexes.to(torch.long).to(dev)
test_pred = model.decode(model.encode(x, test_indexes), test_indexes)

import numpy as np 
pred = np.zeros(len(test_set))
for i, elem in enumerate(test_pred): 
    if elem.item() > 0.5:
        pred[i] = 1
    else:
        pred[i] = 0