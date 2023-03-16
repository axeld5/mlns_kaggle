import tqdm
import networkx as nx 
import numpy as np 

#AdamicAdar could not be imported due to self loops
def feature_extractor(graph, samples):
    """
    Creates a feature vector for each edge of the graph contained in samples 
    """
    feature_vector = [] 
      
    # Degree Centrality measure
    deg_centrality = nx.degree_centrality(graph)
      
    # Betweeness centrality measure
    betweeness_centrality = nx.betweenness_centrality(graph)

    pagerank_analysis = nx.algorithms.link_analysis.pagerank(graph)

    cluster_analysis = nx.algorithms.cluster.clustering(graph)

    for edge in tqdm.tqdm(samples):
        source_node, target_node = edge[0], edge[1]

        # Degree Centrality
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]
        
        # Betweeness centrality measure 
        diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]

        # Preferential Attachement 
        pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]

        # Jaccard
        jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]

        # Pagerank
        diff_pagerank = pagerank_analysis[target_node] - pagerank_analysis[source_node]

        # Cluster
        diff_cluster = cluster_analysis[target_node] - cluster_analysis[source_node]
        
        # Create edge feature vector with all metric computed above
        feature_vector.append([source_degree_centrality, target_degree_centrality, 
                                diff_bt, pref_attach, jacard_coeff, diff_pagerank, diff_cluster]) 
    feature_vector = np.array(feature_vector)
    return feature_vector