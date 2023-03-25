from utils.loader import load_set
from utils.to_nx import set_to_nx
from utils.create_submission import create_submission
from methods.traditional.similarities.community_maker import get_community_based_pred

if __name__ == "__main__":
    train_set = load_set(train=True)
    g = set_to_nx(train_set)
    test_set = load_set(train=False)
    n_test = len(test_set)
    community_pred = get_community_based_pred(g, test_set)
    create_submission(n_test, community_pred, pred_name="submissions/community_pred")
