import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

#AdamicAdar could not be imported due to self loops

class FeatureExtractor:
    def __init__(self, scaler=MinMaxScaler()) -> None:
        self.scaler = scaler

    def feature_extract(self, graph, samples, train):
        """
        Creates a feature vector for each edge of the graph contained in samples 
        """
        feature_vector = [] 

        for edge in tqdm.tqdm(samples):
            source_node, target_node = edge[0], edge[1]

            ## EDGE FEATURES
            # Preferential Attachement 
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            # Jaccard
            jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            # Ressource Allocation Index
            rai = list(nx.resource_allocation_index(graph, [(source_node, target_node)]))[0][2]
            #features = np.concatenate((vect_features, emb_features))
            features = np.array([jacard_coeff, rai, pref_attach])
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        if train:
            feature_vector = self.scaler.fit_transform(feature_vector)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector
        