## this tests the libHIN

from libHIN.IO import load_gml ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding

## load the network
example_net = load_gml("../data/imdb_gml.gml","---")
cycle = {'node_list': [u'movie', u'person', u'movie'], 'edge_list': [u'features', u'acts_in']}

## split and re-weight
decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=cycle)
embedding = hinmine_embedding(decomposed)
print(embedding)

## avtomatic cycle detection?
## word2vec heuristika?
