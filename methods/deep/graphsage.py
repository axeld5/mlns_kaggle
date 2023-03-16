import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, n_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        first_conv = SAGEConv(in_feats, h_feats, 'mean')
        self.convs.append(first_conv)
        self.n_layers = n_layers
        for i in range(n_layers - 2):
            add_conv = SAGEConv(h_feats, h_feats, 'mean')
            self.convs.append(add_conv)
        last_conv = SAGEConv(h_feats, h_feats, 'mean')
        self.convs.append(last_conv)
        self.dropout = dropout

    def forward(self, g, in_feat):
        h = self.convs[0](g, in_feat)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for conv in self.convs[1:self.n_layers-1]:
            h = conv(g, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](g, h)
        return h