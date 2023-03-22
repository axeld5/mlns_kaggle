import numpy as np 
import pandas as pd 

from utils.loader import load_set
from utils.create_submission import create_submission

if __name__ == "__main__":
    unsup_pred = pd.read_csv("unsupervised_pred.csv").to_numpy()
    gnn_pred = pd.read_csv("gnn.csv").to_numpy()
    sup_pred = pd.read_csv("features_extracted_pred.csv").to_numpy() 
    final_pred = 1/3*(unsup_pred + gnn_pred + sup_pred)[:, 1]
    for i in range(len(final_pred)):
        if final_pred[i] > 0.5:
            final_pred[i] = 1
        else:
            final_pred[i] = 0
    n_test = len(final_pred)
    create_submission(n_test, final_pred, "avgd")