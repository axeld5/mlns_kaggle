import dgl
import torch
import itertools
import numpy as np
import dgl.data
import networkx as nx

from evaluate import evaluate_torch
from utils.get_non_edges import get_non_edges
from utils.loader import load_set
from utils.to_nx import set_to_directed_nx
from utils.info_to_dict import info_to_dict
from utils.create_submission import create_submission
from methods.deep.graphsage import GraphSAGE
from methods.deep.predictors import MLPUpgradedPredictor
from methods.deep.train_models import train

if __name__ == "__main__":
    #create dgl graph
    train_set = load_set(train=True)
    nx_g = set_to_directed_nx(train_set)
    node_dict = info_to_dict()
    nx.set_node_attributes(nx_g, values=node_dict, name="feat")
    sorted_nodes = sorted(nx_g.nodes())
    #creating convert dict to keep node embeddings
    convert_dict = {}
    for i, node in enumerate(sorted_nodes):
        convert_dict[node] = i
    g = dgl.from_networkx(nx_g, node_attrs=["feat"], idtype=torch.int32)

    #get positive attributes
    u, v = g.edges()
    eval_model = False
    if eval_model:
        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.1)
        train_size = g.number_of_edges() - test_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

        # Find all known negative edges and split them for training and testing
        non_edges = get_non_edges(train_set)
        non_edges_indic = np.ones((g.number_of_nodes(), g.number_of_nodes()))
        for elem in non_edges: 
            idx_0 = convert_dict[elem[0]] 
            idx_1 = convert_dict[elem[1]]
            non_edges_indic[idx_0][idx_1] = 0
            non_edges_indic[idx_1][idx_0] = 0
        neg_u, neg_v = np.where(non_edges_indic == 0)
        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
        train_g = dgl.remove_edges(g, eids[:test_size])
            
        #create train and test graphs
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

        #Define model, optimizer, and training step
        model = GraphSAGE(train_g.ndata['feat'].shape[1], 64, 32, n_layers=3, dropout=0.2, skip=True)
        pred = MLPUpgradedPredictor(32)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.005, weight_decay=5*10**-4)
        train(model, pred, train_g, train_pos_g, train_neg_g, optimizer, num_epochs=250)

        #make a test
        with torch.no_grad():
            model.eval()
            pred.eval()
            h = model(train_g, train_g.ndata['feat'])
            pos_score = pred(test_pos_g, h)
            neg_score = pred(test_neg_g, h)
            print('AUC', evaluate_torch(pos_score, neg_score))

    else: 
        train_pos_g = dgl.graph((u, v), num_nodes=g.number_of_nodes())
        non_edges = get_non_edges(train_set)
        non_edges_indic = np.ones((g.number_of_nodes(), g.number_of_nodes()))
        for elem in non_edges: 
            idx_0 = convert_dict[elem[0]] 
            idx_1 = convert_dict[elem[1]]
            non_edges_indic[idx_0][idx_1] = 0
            non_edges_indic[idx_1][idx_0] = 0
        neg_u, neg_v = np.where(non_edges_indic == 0)
        train_neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
        model = GraphSAGE(g.ndata['feat'].shape[1], 32, 16, n_layers=3, dropout=0.2, skip=True)
        pred = MLPUpgradedPredictor(16)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)
        train(model, pred, g, train_pos_g, train_neg_g, optimizer, num_epochs=250)

        #make a test
        with torch.no_grad():
            model.eval()
            pred.eval()
            h = model(g, g.ndata['feat'])

        #get submission
        test_set = load_set(train=False)
        test_u = np.zeros(len(test_set))
        test_v = np.zeros(len(test_set))
        for i in range(len(test_set)):
            test_u[i] = convert_dict[test_set[i][0]]
            test_v[i] = convert_dict[test_set[i][1]]
        test_g = dgl.graph((test_u, test_v), num_nodes=g.number_of_nodes())
        with torch.no_grad():
            test_score = pred(test_g, h)
        test_pred = np.zeros(test_score.shape[0])
        for i, elem in enumerate(test_score):
            if elem < 0:
                test_pred[i] = 0
            else:
                test_pred[i] = 1
        n_test = len(test_set)
        create_submission(n_test, test_pred, pred_name="gnn")