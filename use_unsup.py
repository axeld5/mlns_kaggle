import pandas as pd 
import numpy as np 

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.create_submission import create_submission
from methods.traditional.similarities.unsup_extractor import FeatureExtractor
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    extractor = FeatureExtractor()
    test_set = load_set(train=False)
    n_test = len(test_set)
    test_features = extractor.feature_extract(g, test_set, True)
    test_features = pd.DataFrame(test_features)
    test_ranks = test_features.rank(axis=0, method="min")
    test_features = np.mean(test_ranks.to_numpy(), axis=1)
    sorted_features = np.argsort(test_features)[::-1]
    pred = np.zeros(n_test)
    for i in range(int(len(sorted_features)*0.5)):
        idx = sorted_features[i]
        pred[idx] = 1
    create_submission(n_test, pred, pred_name="unsupervised_pred")