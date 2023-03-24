import csv
import networkx as nx 
import pandas as pd 
import numpy as np 
import json 
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.generate_valid_samples import generate_samples
from utils.create_submission import create_submission
from methods.traditional.similarities.global_feat_extractor import FeatureExtractor

#AdamicAdar not working, likely due to self loops
if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    non_edges = get_non_edges(train_set)
    extractor = FeatureExtractor()
    clf = LogisticRegression(C=100, l1_ratio=0.1, penalty="elasticnet", solver="saga", max_iter=10000)
    evaluate_method = False
    if evaluate_method:
        residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.8)
        train_features = extractor.feature_extract(residual_g, train_samples, train=True)
        valid_features = extractor.feature_extract(residual_g, valid_samples, train=False)
        param_dict = {}
        param_dict["penalty"] = ["elasticnet"]
        param_dict["solver"] = ["saga"]
        param_dict["max_iter"] = [10000]
        param_dict["C"] = [0.1, 1, 10, 100, 1000]
        param_dict["l1_ratio"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        clf = GridSearchCV(clf, param_dict, verbose=True)
        clf.fit(train_features, train_labels)
        print(clf.score(valid_features, valid_labels))
        print(clf.best_params_)   
    else:
        total_samples = list(g.edges) + non_edges 
        total_labels = [1 for _ in range(len(g.edges))] + [0 for _ in range(len(non_edges))]
        total_features = extractor.feature_extract(g, total_samples, train=True)
        clf.fit(total_features, total_labels)
        test_set = load_set(train=False)
        n_test = len(test_set)
        test_features = extractor.feature_extract(g, test_set, train=False) 
        pred = clf.predict(test_features)
        create_submission(n_test, pred, pred_name="submissions/supervised_pred")

