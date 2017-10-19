## this tests the libHIN

from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa, read_bitcoin, read_web
from libHIN.label_propagation import *
import networkx as nx
import numpy as np

def test_DEC():
    ## load the network from a gml, decompose and predict labels.
    example_net = load_hinmine_object("../data/imdb_gml.gml","---") ## add support for weight
    cycle = ['movie_____features_____person_____acts_in_____movie'] ## decomposition cycle

    ## split and re-weight
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=cycle)

    ## embedding
    embedding = hinmine_embedding(decomposed, parallel=0)
    print(embedding)
    print("Finished test 1 - DEC")

def test_classification():

    ## CV classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import autosklearn.classification

    ## multi target example
    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    clf = MultiOutputClassifier(forest, n_jobs=4)
    scores = cross_val_score(clf, embedding['data'], embedding['targets'], cv=10)

    ## single target example
    cls = autosklearn.classification.AutoSklearnClassifier()
    scores = cross_val_score(cls, embedding['data'], embedding['targets'][:,0], cv=10)
    print("Finished test 2 - classification")

def test_embedding_raw():

    ## test simple embedding
    simple_net = load_hinmine_object("../data/example_weighted.txt", targets=False) ## embed only
    embedding = hinmine_embedding(simple_net,use_decomposition=False, parallel=8,verbose=True)

def test_embedding_prediction():
    ## do embedding + prediction
    pass


if __name__ == "__main__":
#    test_embedding_raw()
    test_DEC()
