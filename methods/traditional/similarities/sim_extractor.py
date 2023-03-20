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

            #hits
            src_hubs = hubs[source_node]
            tgt_hubs = hubs[target_node]
            hubs_avg = (src_hubs + tgt_hubs)/2
            hubs_std = (src_hubs - tgt_hubs)**2
            src_auth = authorities[source_node]
            tgt_auth = authorities[target_node]
            auth_avg = (src_auth + tgt_auth)/2
            auth_std = (src_auth - tgt_auth)**2

            ## EDGE FEATURES
            # Preferential Attachement 
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]
            # Jaccard
            jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            # Ressource Allocation Index
            rai = list(nx.resource_allocation_index(graph, [(source_node, target_node)]))[0][2]

            

            features = np.array([dg_avg, dg_std, btw_avg, btw_std, p_rank_avg, p_rank_std, hubs_avg, hubs_std, auth_avg, auth_std, pref_attach, jacard_coeff, rai])
            # Create edge feature vector with all metric computed above

            """# Add embedding 
            source_emb_idx = convert_dict[source_node]
            target_emb_idx = convert_dict[target_node]
            source_emb = info_embedding[source_emb_idx, :]
            target_emb = info_embedding[target_emb_idx, :]
            emb_avg = (source_emb + target_emb)/2
            emb_std = (source_emb - target_emb)**2"""
            
            feature_vector.append(features) 
        feature_vector = np.array(feature_vector)
        if train:
            feature_vector = self.scaler.fit_transform(feature_vector)
        else:
            feature_vector = self.scaler.transform(feature_vector)
        return feature_vector
        