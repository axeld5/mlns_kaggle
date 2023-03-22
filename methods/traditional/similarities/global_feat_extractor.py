import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self, scaler=MinMaxScaler()) -> None:
        self.scaler = scaler

    def feature_extract(self, graph, samples, train):
        """
        Creates a feature vector for each edge of the graph contained in samples 
        """
        feature_vector = [] 

        p_rank = nx.pagerank(graph)
        hits = nx.hits(graph)
        hubs = hits[0]
        auth = hits[1]
        simrank = nx.simrank_similarity(graph)

        for edge in tqdm.tqdm(samples):
            source_node, target_node = edge[0], edge[1]
            
            ## Global FEATURES
            #Auth
            src_hubs, tgt_hubs = hubs[source_node], hubs[target_node]
            hubs_avg, hubs_std = (src_hubs+tgt_hubs)/2, (src_hubs - tgt_hubs)**2
            src_auth, tgt_auth = auth[source_node], auth[target_node]
            auth_avg, auth_std = (src_auth+tgt_auth)/2, (src_auth - tgt_auth)**2

            #prank
            src_p_rank = p_rank[source_node]
            tgt_p_rank = p_rank[target_node]
            p_rank_avg = (src_p_rank + tgt_p_rank)/2
            p_rank_std = (src_p_rank - tgt_p_rank)**2

            #simrank 
            edge_simrank = simrank[source_node][target_node]

            #features = np.concatenate((vect_features, emb_features))
            features = np.array([hubs_avg, hubs_std, auth_avg, auth_std, p_rank_avg, p_rank_std, edge_simrank])
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        if train:
            feature_vector = self.scaler.fit_transform(feature_vector)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector