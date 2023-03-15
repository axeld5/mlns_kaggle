from node2vec import Node2Vec

def feature_extractor(samples, node2embedding, feature_func=None):
    # --- Construct feature vectors for edges ---
    if feature_func is None:
        feature_func = lambda x,y: abs(x-y)
    features = [feature_func(node2embedding[edge[0]], node2embedding[edge[1]]) for edge in samples]
    return features

def get_node_embedding(res_g, dimensions=100, walk_length=30, num_walks=200, window_size=5, batch_words=4):
    n2v = Node2Vec(res_g, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks)
    model = n2v.fit(window=window_size, min_count=1, batch_words=batch_words)
    node2embedding = model.wv
    return node2embedding