## this is a basic test of hinmine capabilities

from HINMINE.library_cli import *
import numpy as np

## from HINMINE.lib.HIN import HeterogeneousInformationNetwork
## load test gml and read it further on..

heuristic = "idf"
cycle = {'node_list': [u'movie', u'person'], 'edge_list': [u'directed_by']}
example_net = load_gml("../data/imdb_gml.gml","---")
decomposed = hinmine_decompose(example_net,heuristic,cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)

data = propositionalized['train_features']
print(data)
print(len(data['target_names']))
print(np.count_nonzero(data['target']))
## minitests
