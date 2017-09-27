## this is a basic test of hinmine capabilities

from cf_netSDM.library_cli import *
import numpy as np

## load test gml and read it further on..
heuristic = "rf"
cycle = {'node_list': [u'movie', u'person'], 'edge_list': [u'directed_by', u'directed']}
example_net = load_gml("../data/imdb_gml.gml","_")
decomposed = hinmine_decompose(example_net,heuristic,cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)

data = propositionalized['train_features']
print(len(data['target_names']))
print(np.count_nonzero(data['target']))
## minitests
