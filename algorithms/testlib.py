## this tests the libHIN

from libHIN.IO import load_gml ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa
import networkx as nx
# ## load the network

example_net = load_gml("../data/imdb_gml.gml","---")
cycle = ['movie_____features_____person_____acts_in_____movie']
## split and re-weight
decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=cycle)
embedding = hinmine_embedding(decomposed)
print(np.count_nonzero(embedding['train_features']['data']))

# voting_graph = read_rfa()
# converted = nx.convert_node_labels_to_integers(voting_graph,first_label=0)        
# cycle = ['person_____votes_for_____person']#{'node_list': [u'person'], 'edge_list': [u'votes_for']}
# tmp_path = "tmp.gml"
# nx.write_gml(converted,tmp_path)
# example_graph = load_gml("tmp.gml"," ")
# decomposed = hinmine_decompose(example_graph,heuristic="idf",cycle=cycle)
# embedding = hinmine_embedding(decomposed)
