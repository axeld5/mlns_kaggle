import json 
import torch

from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier

from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.get_non_edges import get_non_edges
from utils.generate_valid_samples import generate_samples
from utils.create_submission import create_submission
from methods.traditional.similarities.deep_sim_extractor import FeatureExtractor
from evaluate import evaluate_tradi

#AdamicAdar not working, likely due to self loops
if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    non_edges = get_non_edges(train_set)
    residual_g, train_samples, train_labels, valid_samples, valid_labels = generate_samples(g, non_edges, 0.8)
    fp = open('convert_dict.json', 'r')
    convert_dict = json.load(fp)
    info_embedding = torch.load("embedding.pt")
    extractor = FeatureExtractor()
    train_features = extractor.feature_extract(residual_g, train_samples, convert_dict, info_embedding, train=True)
    valid_features = extractor.feature_extract(residual_g, valid_samples, convert_dict, info_embedding, train=False)
    #clf = LogisticRegression()
    clf = XGBClassifier()
    clf.fit(train_features, train_labels)

    valid_preds = clf.predict_proba(valid_features)[:, 1]
    evaluate_tradi(valid_labels, valid_preds)

    test_set = load_set(train=False)
    n_test = len(test_set)
    test_features = extractor.feature_extract(residual_g, test_set, convert_dict, info_embedding, train=False) 
    pred = clf.predict(test_features)
    create_submission(n_test, pred, pred_name="deep_extracted_pred")