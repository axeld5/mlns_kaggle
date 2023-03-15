import csv
import networkx as nx 
import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression 

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.generate_valid_samples import generate_samples
from utils.create_submission import create_submission
from methods.traditional.similarities.sim_extractor import feature_extractor
from evaluate import evaluate

#AdamicAdar not working, likely due to self loops
if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    non_edges = get_non_edges(train_set)
    #print(g.number_of_nodes())
    #print(g.number_of_edges())
    residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.9)
    train_features = feature_extractor(residual_g, train_samples)
    valid_features = feature_extractor(residual_g, valid_samples)
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    valid_preds = clf.predict_proba(valid_features)[:, 1]
    evaluate(valid_labels, valid_preds)

    test_set = load_set(train=False)
    n_test = len(test_set)
    test_features = feature_extractor(residual_g, test_set) 
    pred = clf.predict(test_features)
    create_submission(n_test, pred, pred_name="features_extracted_pred")

