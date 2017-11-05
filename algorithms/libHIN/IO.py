## This  part of library reads various files
from .dataStructures import HeterogeneousInformationNetwork
import numpy as np

def load_hinmine_object(infile,label_delimiter=" ",weight_tag = False, targets=True):


    if ".mat" in infile:

        hin = HeterogeneousInformationNetwork(None," ")
        import scipy.io
        mat = scipy.io.loadmat(infile)
        hin.label_matrix = mat['group']
        hin.graph = mat['network']
        return hin

    else:    
        if ".gml" in infile:
            ## parse as gml object
            from networkx import read_gml ## lazy
            net = read_gml(infile)
        
        elif ".txt" in infile:
            ## parse as edgelist
            weight_tag = "weight"
            net = parse_edgelist(infile)
    
        else:
            ## parse as a previously constructed nx object
            net = infile

        hin = HeterogeneousInformationNetwork(net, label_delimiter, weight_tag, target_tag = targets)
    train_indices = []
    test_indices = []
    for index, node in enumerate(hin.node_list):
        if len(hin.graph.node[node]['labels']) > 0:
            train_indices.append(index)
        else:
            test_indices.append(index)
    hin.split_to_indices(train_indices=train_indices, test_indices=test_indices)

    if targets:
        hin.create_label_matrix()
        
    return hin

def parse_edgelist(infile):

    # This is intended for simple file parsing, when only embeddings are considered.
    # Check the examples on the node classificaiton to generate proper input for this type.
    import networkx as nx ## lazy imports
    
    G = nx.MultiDiGraph()
    with open(infile) as inf:
        for line in inf:
            parts = line.strip().split()

            if len(parts) == 2:
                ## unweighted network
                G.add_edge(parts[0],parts[1],weight=1)
                
            if len(parts) == 3:
                ## a simple weighted network
                G.add_edge(parts[0],parts[1],weight=np.absolute(float(parts[2])))
    return G
            
def generate_cv_folds(data,targets,percentage=0.7,nfold=10):
    
    nrow,ncol = data.shape ## get dataframe dimensions
    f1 = int(nrow*0.7) ## determine the random sample size

    ## to je ze loop
    for x in range(nfold):

        ## generate random partition - train
        idx = np.random.randint(nrow, size=f1)

        ## set up the test part - test
        inverse = np.ones(nrow, np.bool)
        inverse[idx] = 0

        ## assign correct partitions
        train_data = data[idx]
        test_data = data[inverse]
        train_targets = targets[idx]
        test_targets = targets[inverse]
        
        yield (train_data,test_data,train_targets,test_targets)
