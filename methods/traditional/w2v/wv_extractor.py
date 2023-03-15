from gensim.models import Word2Vec

from .deepwalk import deep_walk

def feature_extractor(samples, node2embedding, feature_func=None):
    # --- Construct feature vectors for edges ---
    if feature_func is None:
        feature_func = lambda x,y: abs(x-y)
    features = [feature_func(node2embedding[edge[0]], node2embedding[edge[1]]) for edge in samples]
    return features

def get_node_embedding(res_g, num_of_walks=100, walk_length=30, embedding_size=100, window_size=5):
    walks = deep_walk(res_g, num_of_walks, walk_length)
    model = Word2Vec(sentences=walks, vector_size=embedding_size, window=window_size, min_count=0, sg=1, hs=1)
    node2embedding = model.wv
    return node2embedding