import csv
import networkx as nx 
import pandas as pd 
import numpy as np 
import json 
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA 
from xgboost import XGBClassifier

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.generate_valid_samples import generate_samples
from utils.create_submission import create_submission
from methods.traditional.similarities.sim_extractor import FeatureExtractor
from evaluate import evaluate_tradi

#AdamicAdar not working, likely due to self loops
if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    non_edges = get_non_edges(train_set)
    extractor = FeatureExtractor()
    clf = LogisticRegression()
    evaluate_method = True 
    if evaluate_method:
        residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.8)
        train_features = extractor.feature_extract(residual_g, train_samples, train=True)
        valid_features = extractor.feature_extract(residual_g, valid_samples, train=False)
        clf.fit(train_features, train_labels)
        valid_preds = clf.predict_proba(valid_features)[:, 1]
        evaluate_tradi(valid_labels, valid_preds)
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

