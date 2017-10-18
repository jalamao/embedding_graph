## This  part of library reads various files
from networkx import read_gml
from .dataStructures import HeterogeneousInformationNetwork
import numpy as np

def load_hinmine_object(infile,label_delimiter=" "):

    if ".gml" in infile:
        net = read_gml(infile)
    else:
        net = infile

    hin = HeterogeneousInformationNetwork(net, label_delimiter)
    train_indices = []
    test_indices = []
    for index, node in enumerate(hin.node_list):
        if len(hin.graph.node[node]['labels']) > 0:
            train_indices.append(index)
        else:
            test_indices.append(index)
    hin.split_to_indices(train_indices=train_indices, test_indices=test_indices)
    hin.create_label_matrix()
    return hin

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
