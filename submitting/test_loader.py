import csv

# Load test samples 
def load_test() -> list:
    with open("test.txt", "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)
    test_set = [element[0].split(" ") for element in test_set]
    return test_set