import csv
import networkx as nx 
import pandas as pd 
import numpy as np 
import json 
import torch

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.create_submission import create_submission
from methods.traditional.similarities.unsup_extractor import FeatureExtractor
from check import enrich_test
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    extractor = FeatureExtractor()
    test_set = load_set(train=False)
    test_features = extractor.feature_extract(g, test_set, True)
    test_features = np.mean(test_features, axis=1)
    pred = np.zeros_like(test_features) 
    sorted_features = np.argsort(test_features)[::-1]
    for i in range(len(sorted_features)//2):
        idx = sorted_features[i]
        pred[idx] = 1
    n_test = len(test_features)
    y = enrich_test(False)["y"].to_list() 
    print(accuracy_score(pred, y))
    create_submission(n_test, pred, pred_name="unsupervised_pred")