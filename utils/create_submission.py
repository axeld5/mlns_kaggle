import numpy as np
import csv

def create_submission(n_test:int, predictions:np.ndarray, pred_name:str):
    submission = zip(np.arange(n_test), predictions)
    # note: Kaggle requires that you add "ID" and "category" column headers
    csv_name = pred_name+".csv"
    with open(csv_name,"w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(i for i in ["ID", "Predicted"])
        for row in submission:
            csv_out.writerow(row)
        pred.close()
