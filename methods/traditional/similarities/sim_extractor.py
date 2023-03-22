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
        # Degree Centrality measure
        deg_centrality = nx.degree_centrality(graph)
        # Betweeness centrality measure
        betweeness_centrality = nx.betweenness_centrality(graph)

        p_rank = nx.pagerank(graph)
        hits = nx.hits(graph)
        hubs = hits[0]
        authorities = hits[1]

        for edge in tqdm.tqdm(samples):
            source_node, target_node = edge[0], edge[1]
            
            ## NODE FEATURES
            # Degree Centrality
            src_dg_ctrl = deg_centrality[source_node]
            tgt_dg_ctrl = deg_centrality[target_node]
            dg_avg = (src_dg_ctrl+tgt_dg_ctrl)/2 
            dg_std = (src_dg_ctrl-tgt_dg_ctrl)**2
            
            src_btw_ctrl = betweeness_centrality[source_node]
            tgt_btw_ctrl = betweeness_centrality[target_node]
            btw_avg = (src_btw_ctrl+tgt_btw_ctrl)/2 
            btw_std = (src_btw_ctrl-tgt_btw_ctrl)**2

            #prank
            src_p_rank = p_rank[source_node]
            tgt_p_rank = p_rank[target_node]
            p_rank_avg = (src_p_rank + tgt_p_rank)/2
            p_rank_std = (src_p_rank - tgt_p_rank)**2

            #features = np.concatenate((vect_features, emb_features))
            features = np.array([dg_avg, dg_std, btw_avg, btw_std, p_rank_avg, p_rank_std])
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        if train:
            feature_vector = self.scaler.fit_transform(feature_vector)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector
        