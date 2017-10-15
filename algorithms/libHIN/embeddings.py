## this is the embedding code
from dataStructures import HeterogeneousInformationNetwork
from core import stochastic_normalization, page_rank
import numpy as np

def pr_kernel(index_row):
    pr = page_rank(graph, [index_row], try_shrink=True)
    norm = np.linalg.norm(pr, 2)
    if norm > 0:
        pr = pr / np.linalg.norm(pr, 2)
        return (index_row,pr)
                
def hinmine_embedding(hin,use_decomposition=True, parallel=4):

    global graph
    # embed the input network to a term matrix    
    assert isinstance(hin, HeterogeneousInformationNetwork)

    ## special treatment of the decomposed network appears here
    if use_decomposition:
        n = hin.decomposed['decomposition'].shape[0]
        graph = stochastic_normalization(hin.decomposed['decomposition'])
    
    else:

        ## this works on a raw network.
        n = len(hin.graph)
        if hin.weight_tag != False:
            converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
        else:
            converted = nx.to_scipy_sparse_matrix(hin.graph)
            
        graph = stochastic_normalization(converted)

    ## initialize
    vectors = np.zeros((n, n))

    if parallel > 0:
        import multiprocessing as mp
        p = mp.Pool(processes=parallel)
        results = p.map(pr_kernel,range(n))
        for pr_vector in results:
            vectors[pr_vector[0],:] = pr_vector[1]
    
    else:
        ## to se da paralelno!
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
