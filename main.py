import numpy as np 

from utils.loader import load_set
from utils.create_submission import create_submission

if __name__ == "__main__":
    test_set = load_set(train=False)
    train_set = load_set(train=True)
    nodes = []
    for i, elem in enumerate(train_set):
        if elem[0] not in nodes: 
            nodes.append(elem[0])
        if elem[1] not in nodes:
            nodes.append(elem[1])
    print(len(nodes))
    pred = np.random.choice([0, 1], size=len(test_set))
    n_test = len(test_set)
    create_submission(n_test, pred, "random")