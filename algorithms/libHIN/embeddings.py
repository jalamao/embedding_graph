## this is the embedding code
from .dataStructures import HeterogeneousInformationNetwork
from .core import stochastic_normalization, page_rank
from .infolog import emit_state
import numpy as np
import scipy.sparse as sp
from .community_detection import *
from .graphlet_calculation import count_graphlets_orca
import networkx as nx
from collections import Counter

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


def generate_deep_embedding(X, depth=1):    
    
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers

    i_shape = int(X.shape[1])
    encoding_dim = int(X.shape[1]/2)
    
    # this is our input placeholder
    input_matrix = Input(shape=(i_shape,))
#    mid_first = Dense(int(encoding_dim*1.5), activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_matrix)
#    mid_second = Dense(int(encoding_dim*1.5), activation='relu')(encoded)
    decoded = Dense(i_shape, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_matrix, decoded)
    encoder = Model(input_img, encoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X, X,
                epochs=50,
                batch_size=256,
                shuffle=True,verbose=False)

    X = encoder.predict(X)

    print("Encoding stage complete, current shape: {}".format(X.shape))
    
    return X


def hinmine_embedding_gp(hin,use_decomposition=True,return_type="matrix",verbose=False,generate_edge_features = None, from_mat=False,outfile=None,graphlet_binary="./orca",deep_embedding=True):

    assert isinstance(hin, HeterogeneousInformationNetwork)
    
    if use_decomposition:
        if verbose:
            emit_state("Using decomposed networks..")
        n = hin.decomposed['decomposition'].shape[0]
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

    from sklearn.preprocessing import PolynomialFeatures
    
    graphlets = count_graphlets_orca(graph,graphlet_binary)

    ## for level in levels concatenate

    for j in range(4): ## how many recursive embeddings
        deeper_level = generate_deep_embedding(graphlets)
        graphlets = np.concatenate((graphlets,deeper_level),axis=1)
    
    ## tukaj nekaj naredi s to matriko, enrich it.    
    
    return {'data' : graphlets,'targets' : hin.label_matrix, 'decision_threshold' : 0.5}

    
def hinmine_embedding_pr(hin,use_decomposition=True, parallel=True,return_type="matrix",verbose=False, generate_edge_features = None,from_mat=False, outfile=None,feature_permutator_first="0000",deep_embedding=False):

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

        ## this call returns a np/sp matrix of data + np label matrix. This is the best way to directly use the results, yet is memory-exhaustive O(m)~|V|^2 for a square matrix. Numpy can have problems with such sizes..

        ## this threshold specifies the data structure for final embeddings..
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

        from sklearn.linear_model import SGDClassifier
        from sklearn.model_selection import ShuffleSplit
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.metrics import f1_score
        
        if deep_embedding:
            vectors = generate_deep_embedding(vectors)
        
        v = SGDClassifier(loss="log", penalty="l2")
        v = OneVsRestClassifier(v)
        
        pre_split = ShuffleSplit(1, test_size=0.1,random_state=43)
        rs = ShuffleSplit(5, test_size=0.5,random_state=42)

        for x,y in pre_split.split(hin.label_matrix):

            parameter_data = vectors[y]
            parameter_target = hin.label_matrix[y]

        batch=0

        cmax = [0,0] ## optimization criterion
        current_optimum = 0.5 ## default
        
        for threshold in np.arange(0,0.55,0.05):            
            for train_index, test_index in rs.split(parameter_target):
        
                batch += 1
                if verbose:
                    print("Training fold: {}, current_optimum: {}".format(batch, current_optimum))
                
                ## do the splits
                train_X = parameter_data[train_index]
                train_Y = parameter_target[train_index]
                test_X = parameter_data[test_index]
                test_Y = parameter_target[test_index]
            
                model_preds = v.fit(train_X,train_Y).predict_proba(test_X)
                model_preds[model_preds > threshold] = 1
                model_preds[model_preds <= threshold] = 0
                sc_micro = f1_score(test_Y, model_preds, average='micro')
                sc_macro = f1_score(test_Y, model_preds, average='macro')

                if sc_micro > cmax[0] and sc_macro > cmax[1]:
                    current_optimum = threshold
        
                
        return {'data' : vectors,'targets' : hin.label_matrix, 'decision_threshold' : current_optimum}

    
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
