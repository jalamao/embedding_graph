## this tests the libHIN

from libHIN.IO import load_gml ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding

# TODOs:
example_net = load_gml("../data/imdb_gml.gml","---")
decomposed = hinmine_decompose(example_net,heuristic="idf")
#propositionalized = hinmine_propositionalize(decomposed)
