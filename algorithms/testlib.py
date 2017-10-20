## this tests the libHIN

from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import hinmine_embedding ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa, read_bitcoin, read_web
from libHIN.label_propagation import *
import networkx as nx
import numpy as np

def decompose_test(fname, delim):

    example_net = load_hinmine_object(fname,delim) ## add support for weight

    ## split and re-weight
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None)

    ## embedding
    embedding = hinmine_embedding(decomposed, parallel=0)
        
    return embedding

def test_classification_imdb():

    ## CV classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier
    import autosklearn.classification
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier    


    classifiers = {'rf' : RandomForestClassifier(n_estimators=100, random_state=1),
                   'dummy' : DummyClassifier(strategy='most_frequent',random_state=13),
                    'nb' : GaussianNB(),
                   'ada' : AdaBoostClassifier(n_estimators=500),
                   'SVC' : LinearSVC(random_state=0),
                   'MLP' : MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8, 5,3,2), random_state=13),
                   'knn' : OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)),
                   'autoML' :autosklearn.classification.AutoSklearnClassifier(per_run_time_limit=15)}

    ## result container

    embedding = decompose_test("../data/imdb_gml.gml","---")
    
    results = []
    for k,v in classifiers.items():

        v = OneVsRestClassifier(v)
        scores = cross_val_score(v, embedding['data'], embedding['targets'], cv=5, scoring='f1_weighted',n_jobs=4)        
        results.append((k,np.mean(scores)))
        
    results= sorted(results, key=lambda tup: tup[1])
    for x in results:
        cls, score = x
        print("Classifier: {} performed with score of {}".format(cls,score))

    print("Finished test 2 - classification")


def test_automl():

    import autosklearn.classification
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    classifiers = {'autoML' :autosklearn.classification.AutoSklearnClassifier(per_run_time_limit=15)}

    ## result container
    embedding = decompose_test("../data/imdb_gml.gml","---")
    
    results = []
    for k,v in classifiers.items():

        v = OneVsRestClassifier(v)
        scores = cross_val_score(v, embedding['data'], embedding['targets'], cv=5, scoring='f1_weighted')        
        results.append((k,np.mean(scores)))
        
    results= sorted(results, key=lambda tup: tup[1])
    for x in results:
        cls, score = x
        print("Classifier: {} performed with score of {}".format(cls,score))

    print("Finished test 2 - classification")

def test_embedding_raw():

    ## test simple embedding
    simple_net = load_hinmine_object("../data/example_weighted.txt", targets=False) ## embed only
    embedding = hinmine_embedding(simple_net,use_decomposition=False, parallel=8,verbose=True)

def test_embedding_prediction():
    ## do embedding + prediction
    pass


if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_classification_imdb")
    parser.add_argument("--test_automl")
    parser.add_argument("--decompose_test")
    args = parser.parse_args()

    if args.decompose_test:
        decompose_test(args.decompose_test," ")
        
    if args.test_classification_imdb:
        test_classification_imdb()

    if args.test_automl:
        test_automl()
