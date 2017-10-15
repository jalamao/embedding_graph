## this is the embedding code
from dataStructures import HeterogeneousInformationNetwork
from core import stochastic_normalization,page_rank
import numpy as np

def hinmine_embedding(network):
    hin = network
    assert isinstance(hin, HeterogeneousInformationNetwork)
    n = hin.decomposed['decomposition'].shape[0]
    vectors = np.zeros((n, n))
    graph = stochastic_normalization(hin.decomposed['decomposition'])
    for index in range(n):
        pr = page_rank(graph, [index], try_shrink=True)
        norm = np.linalg.norm(pr, 2)
        if norm > 0:
            pr = pr / np.linalg.norm(pr, 2)
            vectors[index, :] = pr
    train_features = {
        'data': vectors[hin.train_indices, :],
        'target': hin.label_matrix[hin.train_indices, :],
        'target_names': [str(x) for x in hin.label_list],
        'DESCR': None
    }
    test_features = {
        'data': vectors[hin.test_indices, :],
        'target_names': [str(x) for x in hin.label_list],
        'DESCR': None
    }
    return {'train_features': train_features, 'test_features': test_features}
