## this tests the libHIN

from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import hinmine_embedding_pr, hinmine_embedding_gp ## basic embedding
from libHIN.decomposition import * ## basic embedding
from dataloaders import read_rfa, read_bitcoin, read_web
from collections import defaultdict
from libHIN.label_propagation import *
import networkx as nx
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import scipy.io as spi
from sklearn import preprocessing

from libHIN.deep_models import *

def decompose_test(fname, delim):

    example_net = load_hinmine_object(fname,delim) ## add support for weight
    
    ## split and re-weight
    print("Beginning decomposition..")

    # c2 = ["movie_____features_____person_____acts_in_____movie"]
   
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)
    
    ## embedding
    print("Starting embedding..")
    embedding = hinmine_embedding_pr(decomposed, parallel=True,verbose=True)
    print(embedding['data'].shape,embedding['targets'].shape)
        
    return embedding


def test_deep_pr_classification(graph,delimiter):
    
    if ".mat" in graph:
        example_net = load_hinmine_object(graph,delimiter) ## add support for weight
        embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=True)
    else:        
        embedding = decompose_test(graph,"---")
    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    ##### learn #####
    
    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    batch = 0        
    threshold = 0.
    models_results = defaultdict(list)
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1        
        print("Fold: {}".format(batch))
                       
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]

        ## for m in models...

        model = baseline_dense_model(train_X, train_Y)
        preds = model.predict(test_X)
        preds[preds>=threshold] = 1
        preds[preds<threshold] =  0
        sc_micro = f1_score(test_Y, preds, average='micro')
        sc_macro = f1_score(test_Y, preds, average='macro')
        models_results[ids].append((sc_micro,sc_macro))
                    
    for k,v in models_results.items():

        micros = []
        macros = []
        for x,y in v:
            micros.append(x)
            macros.append(y)

        print("Model: {} micro: {} macro: {}".format(k,np.mean(micros),np.mean(macros)))
    
    print("Finished test - deep learning..")
            

def test_graphlet_classification(graph, delimiter):

    if ".mat" in graph:
        example_net = load_hinmine_object(graph, delimiter) ## add support for weight
        embedding = hinmine_embedding_gp(example_net, verbose=True, use_decomposition=False, from_mat=True)
    else:        
            example_net = load_hinmine_object(graph,"---") ## add support for weight
    
            ## split and re-weight
            print("Beginning decomposition..")

            decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)
    
            ## embedding
            print("Starting embedding..")
            embedding = hinmine_embedding_gp(decomposed,verbose=True)

            
    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    
    results = []

    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)

    batch = 0

    threshold = embedding['decision_threshold']

    scores_micro = []
    scores_macro = []
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1        
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = v.fit(train_X,train_Y).predict_proba(test_X)
        model_preds[model_preds > threshold] = 1
        model_preds[model_preds <= threshold] = 0
        sc_micro = f1_score(test_Y, model_preds, average='micro')
        sc_macro = f1_score(test_Y, model_preds, average='macro')
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - graphlet-based classification basic")
    

def test_classification(graph,delimiter):
    
    ## direct decomposition
    
    if ".mat" in graph:
        example_net = load_hinmine_object(graph,delimiter) ## add support for weight
        embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=True)
    else:        
        embedding = decompose_test(graph,"---")

    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))


    ## 10 splits 50% train
    
    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    
    results = []

    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)

    batch = 0

    threshold = embedding['decision_threshold']
    
    sel = preprocessing.StandardScaler()

    scores_micro = []
    scores_macro = []
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1
        transformer = sel.fit(embedding['data'][train_index])
        
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = v.fit(train_X,train_Y).predict_proba(test_X)
        model_preds[model_preds > threshold] = 1
        model_preds[model_preds <= threshold] = 0
        sc_micro = f1_score(test_Y, model_preds, average='micro')
        sc_macro = f1_score(test_Y, model_preds, average='macro')
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - classification basic")


def test_automl(graph, delimiter):

    import autosklearn.classification
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    classifiers = {'autoML' :autosklearn.classification.AutoSklearnClassifier(per_run_time_limit=15, time_left_for_this_task=1200)}

    ## result container
    embedding = decompose_test(graph,delimiter)
    
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

def test_rnn(graph, delimiter):

    from sklearn.model_selection import StratifiedKFold
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    if ".mat" in graph:
        example_net = load_hinmine_object(graph,delimiter) ## add support for weight
        embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=True)
    else:        
        embedding = decompose_test(graph,"---")
        
    cvscores = []

    X = embedding['data']
    Y = embedding['targets']
    print(X.shape,Y.shape)
    kf = KFold(n_splits=10,random_state=None, shuffle=False)
    for train, test in kf.split(X):
        # create model
        model = Sequential()
        model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(40,activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(Y.shape[1], activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam')
        # Fit the model
        model.fit(X[train], Y[train], epochs=300, batch_size=40, verbose=1)
        # evaluate the model
        preds = model.predict(X[test])
        preds[preds>=0.5] = 1
        preds[preds<0.5] =  0
        cvscores.append(f1_score(Y[test], preds, average='weighted'))
        
    print("Mean F1: {} and std: {}".format(np.mean(cvscores),np.std(cvscores)))
    
def test_embedding_raw():

    ## test simple embedding
    simple_net = load_hinmine_object("../data/example_weighted.txt", targets=False) ## embed only
    embedding = hinmine_embedding_pr(simple_net,use_decomposition=False, parallel=8,verbose=True)

def test_embedding_prediction():
    ## do embedding + prediction
    pass

def test_label_propagation():

    example_net = load_hinmine_object("../data/imdb_gml.gml","---") ## add support for weight
    ## split and re-weight
    print("Beginning decomposition..")   
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None)
    print("Beginning label propagation..")
    pmat = run_label_propagation(decomposed,weights="balanced")
    print(pmat)


def test_writing(fname,delim,outname):

    example_net = load_hinmine_object(fname,delim) ## add support for weight
    embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,return_type="file",outfile=outname)
    
    pass


def parse_mat(fname, delim):

    ## direct decomposition
    example_net = load_hinmine_object(fname,delim) ## add support for weight
    embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=True)

    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

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
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB


    from sklearn.cross_validation import StratifiedShuffleSplit
    import scipy.io as spi

    ## 10 splits 50% train
    
    rs = StratifiedShuffleSplit(embedding['targets'], 1, test_size=0.5,random_state=42)
            
    results = []

    v = LogisticRegression(penalty='l2')
    v = OneVsRestClassifier(v)

    scores = []
    batch = 0
    
    for train_index, test_index in rs:

        batch += 1
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = v.fit(train_X,train_Y).predict(train_X)
        sc = f1_score(train_Y, model_preds, average='samples')
        scores.append(sc)
            
    results.append(("LR",np.mean(scores)))
        
    results= sorted(results, key=lambda tup: tup[1])
    for x in results:
        cls, score = x
        print("Classifier: {} performed with score of {}".format(cls,score))

    print("Finished test - classification basic")
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_classification")
    parser.add_argument("--test_automl")
    parser.add_argument("--decompose_test")
    parser.add_argument("--test_label_prop")
    parser.add_argument("--graph")
    parser.add_argument("--delimiter")
    parser.add_argument("--test_rnn")
    parser.add_argument("--frommat")
    parser.add_argument("--test_write")
    parser.add_argument("--test_graphlet")
    parser.add_argument("--e2edl")
    
    args = parser.parse_args()

    if args.test_label_prop:
        test_label_propagation()
            
    if args.decompose_test:
        decompose_test(args.graph,args.delimiter)
        
    if args.test_classification:
        test_classification(args.graph,args.delimiter)

    if args.test_automl:
        test_automl(args.graph,args.delimiter)
        
    if args.test_rnn:
        test_rnn(args.graph, args.delimiter)

    if args.frommat:
        parse_mat(args.graph, " ")
        
    if args.test_write:
        test_writing(args.graph, " ","test.emb")

    if args.test_graphlet:
        test_graphlet_classification(args.graph,args.delimiter)

    if args.e2edl:
        test_deep_pr_classification(args.graph,args.delimiter)
        
