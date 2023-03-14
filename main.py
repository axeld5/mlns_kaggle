import numpy as np 

from submitting.test_loader import load_test 
from submitting.create_submission import create_submission

if __name__ == "__main__":
    test_set = load_test()
    pred = np.random.choice([0, 1], size=len(test_set))
    n_test = len(test_set)
    create_submission(n_test, pred)