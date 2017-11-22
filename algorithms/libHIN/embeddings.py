## this is the embedding code
from .dataStructures import HeterogeneousInformationNetwork
from .core import stochastic_normalization, page_rank
from .infolog import emit_state
import numpy as np
import scipy.sparse as sp
from .community_detection import *
from .graphlet_calculation import count_graphlets_orca
from .deep_features import deep_embedding_gp
import networkx as nx
from collections import Counter
from .node2vec_interface import get_n2v_embedding
## compute communities

def return_communities(net):

    G = nx.Graph()
    rows,cols = net.nonzero()
    for row,col in zip(rows,cols):
        G.add_edge(row,col)
    partitions = best_partition(G)

    cnts = Counter(partitions.values())
    sizes = {k:cnts[v] for k,v in partitions.items()}
    return sizes

def return_clustering_coefficients(net):

    ## triangle-based clustering
    G = nx.from_scipy_sparse_matrix(net)
    clusterings = nx.clustering(G)    
    return clusterings

def return_load_centralities(net):

    ## triangle-based clustering
    G = nx.from_scipy_sparse_matrix(net)
    centralities = nx.load_centrality(G)
    return centralities

def return_betweenness_centralities(net):

    ## triangle-based clustering
    G = nx.from_scipy_sparse_matrix(net)
    centralities = nx.betweenness_centrality(G)
    return centralities
    

## compute a page rank feature vector
def pr_kernel(index_row):
    pr = page_rank(graph, [index_row], try_shrink=True)
    norm = np.linalg.norm(pr, 2)
    if norm > 0:
        pr = pr / np.linalg.norm(pr, 2)
        return (index_row,pr)
    else:
        return None

def hinmine_embedding_n2v(hin,use_decomposition=True,return_type="matrix",verbose=False,generate_edge_features = None, from_mat=False,outfile=None,n2v_binary="./node2vec"):
    
    assert isinstance(hin, HeterogeneousInformationNetwork)
    
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
        ## if weighted != False;
        ## elementwise product with the ground thruth network
        graph = stochastic_normalization(hin.decomposed['decomposition'])
    
    else:

        if from_mat:
            graph = hin.graph
            n = hin.graph.shape[0]
            
        else:
            
            if verbose:
                emit_state("Using raw networks..")
            
            ## this works on a raw network.
            n = len(hin.graph)
            if hin.weighted != False:
                converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
            else:
                converted = nx.to_scipy_sparse_matrix(hin.graph)

            if verbose:
                emit_state("Normalizing the adj matrix..")
            graph = stochastic_normalization(converted) ## normalize

    try:
        targets = hin.label_matrix.todense() ## convert targets to dense representation..
    except:
        targets = hin.label_matrix
        pass

    n2v_embedded = get_n2v_embedding(graph,n2v_binary)
    
    ## get n2v embedding, pass on as result
    return {'data' : n2v_embedded,'targets' : targets}
    

def generate_deep_embedding(X, target=None,
                            encoding_dim = 128,
                            reg=10e-5,
                            sample=0.5,
                            act="lrelu",
                            epoch=400,
                            bsize=90):
    
    from keras.layers import Input, Dense, Activation
    from keras.layers.advanced_activations import LeakyReLU
    from keras.models import Model
    from keras import regularizers
    from keras.callbacks import EarlyStopping
    
    ssize = int(X.shape[1]*sample)
    idx = np.random.randint(X.shape[1], size=ssize)

    if sample == 1:
        tra = X
    else:
        tra = X[idx]
        
    if target.any():
        if sample == 1:
            tar = target
        else:
            tar = target[idx]

    ## sample
    i_shape = int(X.shape[0])
    o_shape = int(target.shape[1])
    
    # this is our input placeholder
    input_matrix = Input(shape=(i_shape,))
    encoded = Dense(encoding_dim,
                    activity_regularizer=regularizers.l1(reg))(input_matrix)

    if act == "lrelu":
        activation = LeakyReLU()(encoded)
    else:
        activation = Activation(act)(encoded)
        
    decoded = Dense(o_shape, activation='sigmoid')(activation)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_matrix, decoded)
    encoder = Model(input_matrix, encoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    print("finished deep model compilation..")
    stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')

    if target.any():
        autoencoder.fit(tra,tar,epochs=epoch,batch_size=bsize,shuffle=True,verbose=0,callbacks=[stopping])
        
    else:
        autoencoder.fit(tra,tra,epochs=epoch,batch_size=bsize,shuffle=True,verbose=0,callbacks=[stopping])
    
    Xo = encoder.predict(X)
    print("Encoding stage complete, current shape: {}".format(Xo.shape))
    return (Xo,encoder)

def hinmine_laplacian(hin,use_decomposition=True,return_type="matrix",from_mat=False,verbose=True):
    assert isinstance(hin, HeterogeneousInformationNetwork)
    
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
        ## if weighted != False;
        ## elementwise product with the ground thruth network
        graph = hin.decomposed['decomposition']
    
    else:

        if from_mat:
            graph = hin.graph
            n = hin.graph.shape[0]
        else:
            
            if verbose:
                emit_state("Using raw networks..")
            
            ## this works on a raw network.
            n = len(hin.graph)
            if hin.weighted != False:
                graph = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
            else:
                graph = nx.to_scipy_sparse_matrix(hin.graph)

            if verbose:
                emit_state("Normalizing the adj matrix..")

    from scipy.sparse import csgraph
    vectors = csgraph.laplacian(graph, normed=True)
    try:
        targets = hin.label_matrix.todense() ## convert targets to dense representation..
    except:
        targets = hin.label_matrix
        pass

    return {'data' : vectors.todense(),'targets':targets}
    

                      

def hinmine_deep_gp(hin,use_decomposition=True,return_type="matrix",verbose=False,generate_edge_features = None, from_mat=False,outfile=None,graphlet_binary="./orca"):

    assert isinstance(hin, HeterogeneousInformationNetwork)
    
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
        ## if weighted != False;
        ## elementwise product with the ground thruth network
        graph = stochastic_normalization(hin.decomposed['decomposition'])
    
    else:

        if from_mat:
            graph = hin.graph
            n = hin.graph.shape[0]
            
        else:
            
            if verbose:
                emit_state("Using raw networks..")
            
            ## this works on a raw network.
            n = len(hin.graph)
            if hin.weighted != False:
                converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
            else:
                converted = nx.to_scipy_sparse_matrix(hin.graph)

            if verbose:
                emit_state("Normalizing the adj matrix..")
            graph = stochastic_normalization(converted) ## normalize

    graphlets = count_graphlets_orca(graph,graphlet_binary)
    try:
        targets = hin.label_matrix.todense() ## convert targets to dense representation..
    except:
        targets = hin.label_matrix
        pass

    graphlets_embedded = deep_embedding_gp(graphlets,targets,nlayers=100)

    return {'data' : graphlets_embedded,'targets' : targets}

def hinmine_embedding_gp(hin,use_decomposition=True,return_type="matrix",verbose=False,generate_edge_features = None, from_mat=False,outfile=None,graphlet_binary="./orca",deep_embedding=True):

    assert isinstance(hin, HeterogeneousInformationNetwork)
    
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
        ## if weighted != False;
        ## elementwise product with the ground thruth network
        graph = stochastic_normalization(hin.decomposed['decomposition'])
    
    else:

        if from_mat:
            graph = hin.graph
            n = hin.graph.shape[0]
            
        else:
            
            if verbose:
                emit_state("Using raw networks..")
            
            ## this works on a raw network.
            n = len(hin.graph)
            if hin.weighted != False:
                converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
            else:
                converted = nx.to_scipy_sparse_matrix(hin.graph)

            if verbose:
                emit_state("Normalizing the adj matrix..")
            graph = stochastic_normalization(converted) ## normalize

    ## .......................
    ## .......................
    ## local topology - graphlets
    ## .......................
    ## .......................

    ## raw data - initial setup
    graphlets = count_graphlets_orca(graph,graphlet_binary)
    targets = hin.label_matrix.todense() ## convert targets to dense representation..

    from sklearn.model_selection import train_test_split

    ## train on small percent of the data
    X_train, X_test, y_train, y_test = train_test_split(graphlets, targets, test_size=0.1, random_state=42)
    
    autoencoders = []    ## model container
    
    print("Beginning with recursive embeddings..")
    
    for j in range(15): ## how many recursive embeddings
        deeper_level_embedding, encoder = generate_deep_embedding(X_test, y_test)
        autoencoders.append(encoder)
        X_test = np.concatenate((X_test,deeper_level_embedding),axis=1)

    ## encript the rest of the data
    print("Applying {} autoencoders..".format(len(autoencoders)))

    ## use trained autoencoders
    for enc in autoencoders:
        encoding = enc.predict(graphlets)
        graphlets = np.concatenate((graphlets,encoding), axis=1)

    print("Final shape:{}".format(graphlets.shape))
    return {'data' : graphlets,'targets' : hin.label_matrix, 'decision_threshold' : 0.5}

def hinmine_embedding_pr(hin,use_decomposition=True, parallel=True,return_type="matrix",verbose=False, generate_edge_features = None,from_mat=False, outfile=None,feature_permutator_first="0000",deep_embedding=False,reorder_by_communities=False,simple_input=False,simple_weighted=False):

    # fc_operators = []
    
    ## list of possible features
    topology_operators = ["clustering_information",
                          "load_centrality_information",
                          "betweenness_centrality_information",
                          "community_information"]

    ## map t|f to individual operators
    operator_bool = [True if x == "1" else False for x in feature_permutator_first]

    ## map to feature vectors
    operator_map = dict(zip(topology_operators,operator_bool))
        
    if verbose:
        emit_state("Beginning embedding process..")
        
    global graph ## this holds the graph being processes

    if simple_input: ## use as a class

        n = len(hin.nodes())
        if simple_weighted != False:
            graph = nx.to_scipy_sparse_matrix(hin,weight="weight")
        else:
            graph = nx.to_scipy_sparse_matrix(hin)
            
        graph = stochastic_normalization(graph)

        
    else: ## use within the hinmine
        # embed the input network to a term matrix    
        assert isinstance(hin, HeterogeneousInformationNetwork)

        ## .......................
        ## Use decomposed network
        ## .......................
    
        if use_decomposition:
            if verbose:
                emit_state("Using decomposed networks..")
                n = hin.decomposed['decomposition'].shape[0]
                graph = stochastic_normalization(hin.decomposed['decomposition'])

        ## .......................
        ## Use raw, weighted network
        ## .......................
        
        else:

            if from_mat:

                if verbose:
                    emit_state("Using matrix directly..")
                graph = stochastic_normalization(hin.graph)
                n = hin.graph.shape[0]
            
            else:
            
                if verbose:
                    emit_state("Using raw networks..")
            
                ## this works on a raw network.
                n = len(hin.graph)
                if hin.weighted != False:
                    converted = nx.to_scipy_sparse_matrix(hin.graph,weight=hin.weighted)
                else:
                    converted = nx.to_scipy_sparse_matrix(hin.graph)

                if verbose:
                    emit_state("Normalizing the adj matrix..")
                graph = stochastic_normalization(converted) ## normalize        
            
    ## .......................
    ## .......................
    ## Graph embedding part
    ## .......................
    ## .......................
    
    ## use parallel implementation of PR 
    if parallel:        
        import mkl
        mkl.set_num_threads(1) ## this ports process to individual cores
        if verbose:
            emit_state("Parallel embedding in progress..")
        import multiprocessing as mp ## initialize the MP part
        with mp.Pool(processes=mp.cpu_count()) as p:
            results = p.map(pr_kernel,range(n)) ## those are the embedded vectors

    ## the baseline
    else:
        if verbose:
            emit_state("Non-Parallel embedding in progress..")
        results = []
        for index in range(n):
            pr = page_rank(graph, [index], try_shrink=True)
            norm = np.linalg.norm(pr, 2)
            if norm > 0:
                pr = pr / np.linalg.norm(pr, 2)
                results.append((index,pr))
        
    if verbose:
        emit_state("Finished with embedding..")

                
    if operator_map["community_information"]:

        ## .......................
        ## .......................
        ## global topology - communities
        ## .......................
        ## .......................
        
        if verbose:
            emit_state("Mapping the community information..")
        
        partition_sizes = return_communities(graph)
        for k,v in partition_sizes.items():
            for res in results:
                if res != None:
                    res[1][k]*=v

    if operator_map["load_centrality_information"]:

        ## .......................
        ## .......................
        ## global topology - basic load paths
        ## .......................
        ## .......................
        
        centralities= return_load_centralities(graph)
        for k,v in centralities.items():
            for res in results:
                if res != None:
                    res[1][k]*=v


    if operator_map["betweenness_centrality_information"]:
        
        ## .......................
        ## .......................
        ## global topology - centrality
        ## .......................
        ## .......................
        
        centralities= return_betweenness_centralities(graph)
        max_cent = np.amax(centralities.values())
        for k,v in centralities.items():
            for res in results:
                if res != None:
                    res[1][k]*=v
                    
    if operator_map["clustering_information"]:

        ## .......................
        ## .......................
        ## global topology - basic clustering
        ## .......................
        ## .......................
        
        clusterings = return_clustering_coefficients(graph)
        for k,v in clusterings.items():
            for res in results:
                if res != None:
                    res[1][k]*=v        
                
        
    if generate_edge_features != None:
        emit_state("Generating edge-based features")
        pass
    

    if verbose:
        emit_state("Writing to output..")

    ## a se kar tu natrenira? Threshold tudi?
        
    if return_type == "matrix":
        
        size_threshold = 100000
        if n > size_threshold:
            vectors = sp.csr_matrix((n, n))
        else:
            vectors = np.zeros((n, n))
        
        for pr_vector in results:
            if pr_vector != None:
                if  n > size_threshold:
                    col = range(0,n,1)
                    row = np.repeat(pr_vector[0],n)
                    val = pr_vector[1]
                    vectors = vectors + sp.csr_matrix((val, (row,col)), shape=(vdim[0],vdim[1]), dtype=float)
                else:
                    vectors[pr_vector[0],:] = pr_vector[1]


        try:
            hin.label_matrix = hin.label_matrix.todense()
        except:
            pass

        if deep_embedding:
            if verbose:
                emit_state("Generating the deep embedding..")
            vectors, encoder = generate_deep_embedding(vectors, target = hin.label_matrix)

        if simple_input:
            return {'data' : vectors}
        else:
            return {'data' : vectors,'targets' : hin.label_matrix}

    
    elif return_type == "file":

        if outfile != None:
            f=open(outfile,'a')
            for rv in results:
                if rv != None:
                    
                    index = rv[0] ## indices
                    vals = rv[1] ## pagerank vectors
                    
                    fv = np.concatenate(([index],vals))
                    outstring = ",".join([str(x) for x in fv.tolist()])+"\n"
                    f.write(outstring)

            f.close()

        else:
            print("Please enter output file name..")
        
        pass

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

class hinmine_embedding:
    def __init__(self, method,augmentation="none"):
        self.method = method
        self.augmentation = augmentation

    def learn_embedding(self, graph, is_weighted=True, edge_f=None, no_python=None):
        if self.method == "pagerank":
            results = hinmine_embedding_pr(graph,use_decomposition=True, parallel=True,return_type="matrix", outfile=None,feature_permutator_first="0000",simple_input=True,simple_weighted=is_weighted,verbose=True)
            return (results['data'],True)
