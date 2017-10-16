## this tests the libHIN

from libHIN.IO import load_hinmine_object ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa
import networkx as nx
import numpy as np

# ## load the network

example_net = load_hinmine_object("../data/imdb_gml.gml","---") ## add support for weight
cycle = ['movie_____features_____person_____acts_in_____movie'] ## decomposition cycle
## split and re-weight
decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=cycle)

## does embedding work as it is?3
embedding = hinmine_embedding(decomposed)
print(np.count_nonzero(embedding['train_features']['data']))

### TODOs

# 1.) parallel doesnt work well
# 2.) finalframe as one blob
# 3.) test some models finally!

## naredi, da bo delal brez dekompozicije!
# voting_graph = read_rfa("../data/smaller.txt") #rfa_all.NL-SEPARATED.txt
# converted = nx.convert_node_labels_to_integers(voting_graph,first_label=0)        
# cycle = ['person_____votes_for_____person_____voted_by_____person']
# tmp_path = "tmp.gml"
# nx.write_gml(converted,tmp_path)
# example_graph = load_hinmine_object("tmp.gml"," ")

# ## a je mozno direktno z utezmi dodat v embedding?
# decomposed = hinmine_decompose(example_graph,heuristic="idf",cycle=cycle)
# embedding = hinmine_embedding(decomposed) ## nekak weighted matrix direktno?

# print(embedding)
