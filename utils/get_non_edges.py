from typing import List 

def get_non_edges(dataset:List[str]) -> List[str]:
    non_edge_list = []
    for elem in dataset:
        if elem[2] == "0":
            non_edge_list.append([elem[0], elem[1]])
    return non_edge_list