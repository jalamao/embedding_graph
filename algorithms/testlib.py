## this tests the libHIN

from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa
from libHIN.label_propagation import *
import networkx as nx
import numpy as np

# ## load the network

example_net = load_hinmine_object("../data/imdb_gml.gml","---") ## add support for weight
cycle = ['movie_____features_____person_____acts_in_____movie'] ## decomposition cycle
## split and re-weight
decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=cycle)

## test label propagation
result = run_label_propagation(decomposed)
print(result)

## does embedding work as it is?3
embedding = hinmine_embedding(decomposed)
print("train shape{}, test shape {}".format(embedding['data'].shape,embedding['targets'].shape))

## CV classification

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_val_score
#import autosklearn.classification

# forest = RandomForestClassifier(n_estimators=100, random_state=1)
# #clf = MultiOutputClassifier(forest, n_jobs=4)
# scores = cross_val_score(forest, embedding['data'], embedding['targets'][:,0], cv=10)
# print(scores)

#cls = autosklearn.classification.AutoSklearnClassifier()
#scores = cross_val_score(cls, embedding['data'], embedding['targets'][:,0], cv=10)
#print(scores)
### TODOs

## kako oceniti uspesnost?
## pipeline cez vec dataset - ov.
## dodaj w2w

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
