import pandas as pd 

from sklearn.metrics import accuracy_score
from utils.create_submission import create_submission

if __name__ == "__main__":
    unsup_pred = pd.read_csv("submissions/unsupervised_pred.csv").to_numpy()
    gnn_pred = pd.read_csv("submissions/gnn.csv").to_numpy()
    sup_pred = pd.read_csv("submissions/supervised_pred.csv").to_numpy() 
    comm_pred = pd.read_csv("submissions/community_pred.csv").to_numpy()
    final_pred = (0.25*sup_pred+0.25*unsup_pred+0.25*gnn_pred+0.25*comm_pred)[:, 1]
    for i in range(len(final_pred)):
        if final_pred[i] >= 0.5:
            final_pred[i] = 1
        else:
            final_pred[i] = 0
    n_test = len(final_pred)
    create_submission(n_test, final_pred, "submissions/avgd")