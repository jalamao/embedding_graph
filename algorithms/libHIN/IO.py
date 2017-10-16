## This  part of library reads various files
from networkx import read_gml
from .dataStructures import HeterogeneousInformationNetwork

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
