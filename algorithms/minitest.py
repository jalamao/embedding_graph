## this is a basic test of hinmine capabilities

from cf_netSDM.library_cli import *

## load test gml and read it further on..
heuristic = "delta"
cycle = {'node_list': [u'movie', u'person', u'movie'], 'edge_list': [u'features', u'acts_in']}
example_net = load_gml("../data/imdb_gml.gml","_")
decomposed = hinmine_decompose(example_net,heuristic,cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)
print(propositionalized)

