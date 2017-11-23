## whole deepRank approach in a single file.

import networkx as nx
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, Counter
from libHIN.community_detection import *
import multiprocessing as mp ## initialize the MP part
import mkl

def emit_state(message):
    print ("-----\n{}\n-----".format(message))

def stochastic_normalization(matrix):
    matrix = matrix.tolil()
    try:
        matrix.setdiag(0)
    except TypeError:
        matrix.setdiag(np.zeros(matrix.shape[0]))
    matrix = matrix.tocsr()
    d = matrix.sum(axis=1).getA1()
    nzs = np.where(d > 0)
    d[nzs] = 1 / d[nzs]
    matrix = (sp.diags(d, 0).tocsc().dot(matrix)).transpose()
    return matrix

def generate_deep_embedding(X, target=[],
                            encoding_dim = 50,
                            reg=10e-5,
                            act="lrelu",
                            sample=1,
                            epoch=400,
                            bsize=90,
                            verbose_tag=0):
    
    from keras.layers import Input, Dense, Activation,ActivityRegularization
    from keras.layers.advanced_activations import LeakyReLU
    from keras.models import Model
    from keras import regularizers
    from keras.callbacks import EarlyStopping

    ssize = int(X.shape[1]*sample)
    idx = np.random.randint(X.shape[1], size=ssize)

    tra = X[idx]

    try:
        target = target.todense()
    except:
        ## all good..
        pass
    
    i_shape = int(X.shape[0])
    if target.any():
        o_shape = int(target.shape[1])
        tar = target[idx]
    else:
        o_shape = i_shape
        
    ## THE ARCHITECTURE ##
    input_matrix = Input(shape=(i_shape,))
    encoded = Dense(encoding_dim)(input_matrix)
    reg1 = ActivityRegularization(l1=reg)(encoded)
    if act == "lrelu":
        activation = LeakyReLU()(reg1)
    else:
        activation = Activation(act)(reg1)
    decoded = Dense(o_shape, activation='sigmoid')(activation)
    
    # this model maps an input to its reconstruction    
    autoencoder = Model(input_matrix, decoded)
    encoder = Model(input_matrix, encoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    stopping = EarlyStopping(monitor='loss', patience=20, verbose=verbose_tag, mode='auto')

    if target.any():
        autoencoder.fit(tra,
                        tar,
                        epochs=epoch,
                        batch_size=bsize,
                        shuffle=True,
                        verbose=verbose_tag,
                        callbacks=[stopping])
        
    else:
        autoencoder.fit(tra,
                        tra,
                        epochs=epoch,
                        batch_size=bsize,
                        shuffle=True,verbose=verbose_tag,
                        callbacks=[stopping])
    
    return encoder

def sparse_page_rank(matrix, start_nodes,
              epsilon=1e-6,
              max_steps=10000,
              damping=0.85,
              spread_step=10,
              spread_percent=0.5,
              try_shrink=True):
    assert(len(start_nodes)) > 0
    
    # this method assumes that column sums are all equal to 1 (stochastic normalizaition!)
    size = matrix.shape[0]
    if start_nodes is None:
        start_nodes = range(size)
        nz = size
    else:
        nz = len(start_nodes)
    start_vec = np.zeros((size, 1))
    start_vec[start_nodes] = 1
    start_rank = start_vec / len(start_nodes)
    rank_vec = start_vec / len(start_nodes)
    # calculate the max spread:
    shrink = False
    which = np.zeros(0)
    if try_shrink:
        v = start_vec / len(start_nodes)
        steps = 0
        while nz < size * spread_percent and steps < spread_step:
            steps += 1
            v += matrix.dot(v)
            nz_new = np.count_nonzero(v)
            if nz_new == nz:
                shrink = True
                break
            nz = nz_new
        rr = np.arange(matrix.shape[0])
        which = (v[rr] > 0).reshape(size)
        if shrink:
            start_rank = start_rank[which]
            rank_vec = rank_vec[which]
            matrix = matrix[:, which][which, :]
    diff = np.Inf
    steps = 0
    while diff > epsilon and steps < max_steps:  # not converged yet
        steps += 1
        new_rank = matrix.dot(rank_vec)
        rank_sum = np.sum(new_rank)
        if rank_sum < 0.999999999:
            new_rank += start_rank * (1 - rank_sum)
        new_rank = damping * new_rank + (1 - damping) * start_rank
        new_diff = np.linalg.norm(rank_vec - new_rank, 1)
        diff = new_diff
        rank_vec = new_rank
    if try_shrink and shrink:
        ret = np.zeros(size)        
        rank_vec = rank_vec.T[0] ## this works for both python versions
        ret[which] = rank_vec
        ret[start_nodes] = 0
        return ret.flatten()
    else:
        rank_vec[start_nodes] = 0
        return rank_vec.flatten()


def return_communities(net):

    G = nx.Graph()
    rows,cols = net.nonzero()
    for row,col in zip(rows,cols):
        G.add_edge(row,col)
    partitions = best_partition(G)

    cnts = Counter(partitions.values())
    sizes = {k:cnts[v] for k,v in partitions.items()}
    return sizes
    

## compute a page rank feature vector
def pr_kernel(index_row):
    pr = sparse_page_rank(graph, [index_row], try_shrink=True)
    norm = np.linalg.norm(pr, 2)
    if norm > 0:
        pr = pr / np.linalg.norm(pr, 2)
        return (index_row,pr)
    else:
        return None

def hinmine_embedding_pr(network,targets = False,verbose=False, community_information=True, deep_embedding=False,sample=0.1):

    if verbose:
        emit_state("Beginning embedding process..")
        
    global graph ## this holds the graph being processed
    n = network.shape[1]
    graph = stochastic_normalization(network) ## normalize    
    
    # if deep_embedding:        

    #     ssize = int(graph.shape[1]*sample)
    #     idx = np.random.randint(graph.shape[1], size=ssize)
    #     subtars = targets[idx]
    #     #        subgraph = graph.tocsr()[idx,:].tocsc()[:,idx]
    #     subgraph = graph.tocsr()[idx,:]
    #     tmp = nx.from_scipy_sparse_matrix(subgraph)
    #     idx_new = [x for x in tmp.nodes()]
    #     subgraph = nx.to_scipy_sparse_matrix(tmp)
    #     deep_vectors = np.zeros((len(idx), len(idx)))
        
    #     if verbose:
    #         emit_state("Training the encoder on a subset of the original graph..")

    #     deep_results = []        
    #     for index in idx_new:
    #         pr = sparse_page_rank(subgraph, [index], try_shrink=True)
    #         norm = np.linalg.norm(pr, 2)
    #         if norm > 0:
    #             pr = pr / np.linalg.norm(pr, 2)
    #             deep_results.append((index,pr))
                
    #     for pr_vector in deep_results:
    #         if pr_vector != None:
    #             deep_vectors[pr_vector[0],:] = pr_vector[1]
                    
    #     encoder = generate_deep_embedding(deep_vectors, target = subtars)
        

    ## .......................
    ## .......................
    ## Graph embedding part
    ## .......................
    ## .......................
    
    ## use parallel implementation of PR 
    mkl.set_num_threads(1) ## this ports process to individual cores

    if verbose:
        emit_state("Parallel embedding in progress..")
        
    with mp.Pool(processes=mp.cpu_count()) as p:
        results = p.map(pr_kernel,range(n)) ## those are the embedded vectors

    if verbose:
        emit_state("Finished with core embedding..")
                
    ## .......................
    ## .......................
    ## global topology - communities
    ## .......................
    ## .......................

    if community_information:
        if verbose:
            emit_state("Mapping the community information..")
        
        partition_sizes = return_communities(graph)
        for k,v in partition_sizes.items():
            for res in results:
                if res != None:
                    res[1][k]*=v

    vectors = np.zeros((n, n))
    
    for pr_vector in results:
        if pr_vector != None:
            vectors[pr_vector[0],:] = pr_vector[1]

    if deep_embedding:
        if verbose:
            emit_state("Generating the deep embedding..")
        vectors = generate_deep_embedding(vectors,target = targets).predict(vectors)


    return {'data' : vectors,'targets' : targets}



class DeepR:
    def __init__(self, method,vb=False):
        self.method = method
        self.vb = vb
    def learn_embedding(self, core_network, labels=[]):

        if self.method == "default":
            results = hinmine_embedding_pr(core_network,
                                           targets = labels,
                                           verbose=self.vb,
                                           community_information=True,
                                           deep_embedding=True)
            
        if self.method == "no_community":
            results = hinmine_embedding_pr(core_network,
                                           targets = labels,
                                           verbose=self.vb,
                                           community_information=False,
                                           deep_embedding=True)
            
        if self.method == "no_deep":
            results = hinmine_embedding_pr(core_network,
                                           targets = labels,
                                           verbose=self.vb,
                                           community_information=True,
                                           deep_embedding=False)
            
        if self.method == "no_deep_no_community":
            results = hinmine_embedding_pr(core_network,
                                           targets = labels,
                                           verbose=self.vb,
                                           community_information=False,
                                           deep_embedding=False)
        return (results,self.method)


if __name__ == "__main__":

    import scipy.io
    mat = scipy.io.loadmat("../matrix_data/Homo_sapiens.mat")
    labels= mat['group']
    core_network= mat['network']

    raw = DeepR("deepR_default")
    embeddings = raw.learn_embedding(core_network,labels)
    
    print(embeddings)
