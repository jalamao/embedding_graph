## this is the embedding code
from .dataStructures import HeterogeneousInformationNetwork
from .core import stochastic_normalization, page_rank
from .infolog import emit_state
import numpy as np
import scipy.sparse as sp

def pr_kernel(index_row):
    pr = page_rank(graph, [index_row], try_shrink=True)
    norm = np.linalg.norm(pr, 2)
    if norm > 0:
        pr = pr / np.linalg.norm(pr, 2)
        return (index_row,pr)

def csr_vappend(a,b):

    ##  this is an effictient variation of vstack - for sparse result concatenation.
    
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    #return a
                
def hinmine_embedding(hin,use_decomposition=True, parallel=True,return_type="raw",verbose=False):

    if verbose:
        emit_state("Beginning embedding process..")
        
    global graph
    # embed the input network to a term matrix    
    assert isinstance(hin, HeterogeneousInformationNetwork)

    ## special treatment of the decomposed network appears here
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
        graph = stochastic_normalization(hin.decomposed['decomposition'])
    
    else:
        if verbose:
            emit_state("Using raw networks..")
        import networkx as nx
        ## this works on a raw network.
        n = len(hin.graph)
        if hin.weighted != False:
            converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
        else:
            converted = nx.to_scipy_sparse_matrix(hin.graph)

        if verbose:
            emit_state("Normalizing the adj matrix..")
        graph = stochastic_normalization(converted)

    ## initialize

    if n > 5000:
        vectors = sp.csr_matrix((n, n))
    else:
        vectors = np.zeros((n, n))

    if parallel:

        import mkl
        mkl.set_num_threads(1)
        
        if verbose:
            emit_state("Parallel embedding in progress..")
        import multiprocessing as mp
        with mp.Pool(processes=mp.cpu_count()) as p:
            results = p.map(pr_kernel,range(n))
        for pr_vector in results:
            if pr_vector != None:
                vectors[pr_vector[0],:] = pr_vector[1]
                vectors = vectors.tocsr()
    else:
        if verbose:
            emit_state("Non-Parallel embedding in progress..")
        for index in range(n):
            pr = page_rank(graph, [index], try_shrink=True)
            norm = np.linalg.norm(pr, 2)
            if norm > 0:
                pr = pr / np.linalg.norm(pr, 2)
                vectors[index, :] = pr
                
    if verbose:
        emit_state("Finished with embedding..")
    if return_type == "raw":
        return {'data' : vectors,'targets' : hin.label_matrix}

    else:
        ## return bo dodelan, verjetno zgolj dve matriki tho.
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
