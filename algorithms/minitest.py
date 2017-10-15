## this is a basic test of hinmine capabilities

from HINMINE.library_cli import *
import numpy as np

## load test gml and read it further on..
heuristic = "tf"
#cycle = {'node_list': [u'movie', u'person',u'movie'], 'edge_list': [u'features',u'acts_in']}
cycle = ['movie_____features_____person_____acts_in_____movie']
example_net = load_gml("../data/imdb_gml.gml","---")
decomposed = hinmine_decompose(example_net,heuristic,cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)
data = propositionalized
print(data)
#print(np.count_nonzero(data['data'])) ## to vrne 0, kar ni isto kot CF-workflow
