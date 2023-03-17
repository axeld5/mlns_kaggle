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
from methods.traditional.w2v.wv_extractor import feature_extractor, get_node_embedding
from evaluate import evaluate_tradi

#AdamicAdar not working, likely due to self loops
if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    non_edges = get_non_edges(train_set)
    residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.8)
    print("samples generated")
    node_embedding = get_node_embedding(residual_g)
    print("node_embeddings gotten")
    train_features = feature_extractor(train_samples, node_embedding)
    valid_features = feature_extractor(valid_samples, node_embedding)
    print("features extracted")
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)
    print("training done")

    valid_preds = clf.predict_proba(valid_features)[:, 1]
    evaluate_tradi(valid_labels, valid_preds)

    test_set = load_set(train=False)
    n_test = len(test_set)
    test_features = feature_extractor(test_set, node_embedding) 
    pred = clf.predict(test_features)
    create_submission(n_test, pred, pred_name="word2vec_pred")