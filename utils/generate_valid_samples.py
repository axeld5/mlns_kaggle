import numpy as np
import networkx as nx 

from typing import List

def generate_samples(graph:nx.Graph, non_edges:List[str], train_set_ratio:float):
    """
    Graph pre-processing step required to perform supervised link prediction
    Create training and valid sets
    """

    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")
       
    # --- Step 1: Generate positive edge samples for valid set ---
    residual_g = graph.copy()
    valid_pos_samples = []
      
    # Store the shuffled list of current edges of the graph
    edges = list(residual_g.edges())
    np.random.shuffle(edges)
    
    # Define number of positive valid samples desired
    valid_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    train_set_size = graph.number_of_edges() - valid_set_size
    num_of_pos_valid_samples = 0
    
    # Remove random edges from the graph, leaving it connected
    # Fill in the blanks
    for edge in edges:
        
        # Remove the edge
        residual_g.remove_edge(edge[0], edge[1])
        
        # Add the removed edge to the positive sample list if the network is still connected
        if nx.is_connected(residual_g):
            num_of_pos_valid_samples += 1
            valid_pos_samples.append(edge)
        # Otherwise, re-add the edge to the network
        else: 
            residual_g.add_edge(edge[0], edge[1])
        
        # If we have collected enough number of edges for valid set, we can terminate the loop
        if num_of_pos_valid_samples == valid_set_size:
            break
    
    # Check if we have the desired number of positive samples for valid set 
    if num_of_pos_valid_samples != valid_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

        
    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = list(residual_g.edges())
        
        
    # --- Step 3: Generate the negative samples for valid and training sets ---
    # Fill in the blanks
    np.random.shuffle(non_edges)
    
    train_neg_samples = non_edges[:train_set_size] 
    valid_neg_samples = non_edges[train_set_size:train_set_size + valid_set_size]

    
    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For valid set
    valid_samples = valid_pos_samples + valid_neg_samples
    valid_labels = [1 for _ in valid_pos_samples] + [0 for _ in valid_neg_samples]
    
    return residual_g, train_samples, train_labels, valid_samples, valid_labels