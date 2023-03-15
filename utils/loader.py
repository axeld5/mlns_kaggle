import csv

from typing import List

# Load test samples 
def load_set(train=False) -> List[str]:
    if train: 
        with open("train.txt", "r") as f:
            reader = csv.reader(f)
            dataset = list(reader)
        dataset = [element[0].split(" ") for element in dataset]
    else:
        with open("test.txt", "r") as f:
            reader = csv.reader(f)
            dataset = list(reader)
        dataset = [element[0].split(" ") for element in dataset]
    return dataset

