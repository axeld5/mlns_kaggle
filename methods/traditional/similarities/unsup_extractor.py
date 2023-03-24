import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self, scaler=MinMaxScaler()) -> None:
        self.scaler = scaler

    def feature_extract(self, graph, samples, train):
        """
        Creates a feature vector for each edge of the graph contained in samples 
        """
        feature_vector = [] 
        epured_samples = []
        for edge in samples:
            if edge[0] != edge[1]:
                epured_samples.append(edge)
        cnc = nx.common_neighbor_centrality(graph, epured_samples) 
        cnc_list = list(cnc)   
        for edge in tqdm.tqdm(samples):
            source_node, target_node = edge[0], edge[1]

            ## EDGE FEATURES
            # Preferential Attachement 
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            # Jaccard
            jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            # Ressource Allocation Index
            rai = list(nx.resource_allocation_index(graph, [(source_node, target_node)]))[0][2]
            # Adamic Adar Index 
            aai = 0 
            if target_node != source_node: 
                aai = list(nx.adamic_adar_index(graph, [(source_node, target_node)]))[0][2]
            #features = np.concatenate((vect_features, emb_features))
            cnc = 0
            if target_node != source_node:
                for elem in cnc_list: 
                    if elem[0] == source_node and elem[1] == target_node:
                        cnc = elem[2] 
                        continue
            # Shortest Path
            sp = nx.shortest_path_length(graph, source_node, target_node)

            features = np.array([jacard_coeff, rai, pref_attach, aai, cnc, sp])
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        return feature_vector
        