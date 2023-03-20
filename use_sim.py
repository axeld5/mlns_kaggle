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
    residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.8)
    node_information = pd.read_csv("node_information.csv", sep=",", header=None) 
    deep = True
    if deep:        
        fp = open('convert_dict.json', 'r')
        convert_dict = json.load(fp)
        info_embedding = torch.load("embedding.pt")
    else:
        node_names = node_information[node_information.columns[0]].tolist()
        convert_dict = {}
        for i, idx in enumerate(node_names):
            convert_dict[str(idx)] = i 
        node_information.drop(node_information.columns[0], axis=1, inplace=True)
        info_embedding = node_information.to_numpy()
    extractor = FeatureExtractor()
    train_features = extractor.feature_extract(residual_g, train_samples, convert_dict, info_embedding, train=True)
    valid_features = extractor.feature_extract(residual_g, valid_samples, convert_dict, info_embedding, train=False)
    clf = RandomForestClassifier()
    #clf = LogisticRegression()
    #clf = XGBClassifier()
    clf.fit(train_features, train_labels)

    valid_preds = clf.predict_proba(valid_features)[:, 1]
    evaluate_tradi(valid_labels, valid_preds)

    test_set = load_set(train=False)
    n_test = len(test_set)
    test_features = extractor.feature_extract(residual_g, test_set, convert_dict, info_embedding, train=False) 
    pred = clf.predict(test_features)
    create_submission(n_test, pred, pred_name="features_extracted_pred")

