import tqdm
import networkx as nx 
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#AdamicAdar could not be imported due to self loops


class FeatureExtractor:
    def __init__(self, scaler=StandardScaler()) -> None:
        self.scaler = scaler

    def feature_extract(self, graph, samples, convert_dict, info_embedding, train):
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

            # Degree Centrality
            src_dg_ctrl = deg_centrality[source_node]
            tgt_dg_ctrl = deg_centrality[target_node]
            src_btw_ctrl = betweeness_centrality[source_node]
            tgt_btw_ctrl = betweeness_centrality[target_node]
            
            # Betweeness centrality measure 
            #diff_bt = np.abs(betweeness_centrality[target_node] - betweeness_centrality[source_node])
            # Preferential Attachement 
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            # Jaccard
            jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            # Ressource Allocation Index
            rai = list(nx.resource_allocation_index(graph, [(source_node, target_node)]))[0][2]
            # Add embedding 
            source_emb_idx = convert_dict[source_node]
            target_emb_idx = convert_dict[target_node]
            source_emb = info_embedding[source_emb_idx, :]
            target_emb = info_embedding[target_emb_idx, :]
            info_emb = cosine_similarity(source_emb.reshape(1, -1), target_emb.reshape(1, -1))[0, 0]

            # Shortest path
            #spl = nx.shortest_path_length(graph, source_node, target_node) = 0
            spl = 0

            #prank
            src_p_rank = p_rank[source_node]
            tgt_p_rank = p_rank[target_node]

            #hits
            src_hubs = hubs[source_node]
            tgt_hubs = hubs[target_node]
            src_auth = authorities[source_node]
            tgt_auth = authorities[target_node]

            features = np.array([src_dg_ctrl, tgt_dg_ctrl, src_btw_ctrl, tgt_btw_ctrl, pref_attach, jacard_coeff, src_p_rank, tgt_p_rank, src_hubs, tgt_hubs,
                                 src_auth, tgt_auth, rai, info_emb])
            # Create edge feature vector with all metric computed above
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        if train:
            feature_vector = self.scaler.fit_transform(feature_vector)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector
        