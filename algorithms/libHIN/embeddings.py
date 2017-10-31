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

def hadamand_sum(embedding):

    ## construct pairwise feature constructs, representing individual edges
    ## return a map, containing index pair tuple, and a feature vector
    
    pass

def hadamand_prod():
    pass

def hadamand_L1():
    pass

def hadamand_L2():
    pass
    
def csr_vappend(a,b):

    ##  this is an effictient variation of vstack - for sparse result concatenation.
    
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    #return a
                
def hinmine_embedding(hin,use_decomposition=True, parallel=True,return_type="raw",verbose=False, generate_edge_features = None):

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
    size_threshold = 5000
    if n > size_threshold:
        vectors = sp.csr_matrix((n, n))
    else:
        vectors = np.zeros((n, n))

    if parallel:
        
        import mkl
        mkl.set_num_threads(1)

        vdim = vectors.shape
        
        if verbose:
            emit_state("Parallel embedding in progress..")
        import multiprocessing as mp
        with mp.Pool(processes=mp.cpu_count()) as p:
            results = p.map(pr_kernel,range(n))

        ## ze tukaj naredi edge embedding? output to file
        ## v matriko lahko le na koncu.. to je treba fino dodelat.
            
        for enx, pr_vector in enumerate(results):
            if pr_vector != None:
                if  size_threshold > 5000:            
                    col = range(0,vdim[0],1)
                    row = np.repeat(pr_vector[0],vdim[0])
                    val = pr_vector[1]
                    vectors = vectors +  sp.csr_matrix((val, (row,col)), shape=(vdim[0],vdim[1]), dtype=float)
                else:
                    vectors[pr_vector[0],:] = pr_vector[1]
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

    if generate_edge_features == None:
        emit_state("Generating edge-based features")

        ## select the pairwise composition function
        ## for each pair of nodes, f(n1,n2) = E
        ## edge labels are tuples of length 2
        ## for constructed edge in edges, do: join
        
    if return_type == "raw":
        #print(vectors.todense())
        return {'data' : vectors,'targets' : hin.label_matrix}
    elif return_type == "file":
        ## write to two separate files..

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
