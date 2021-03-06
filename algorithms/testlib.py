## this tests the libHIN

from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import hinmine_embedding_pr, hinmine_embedding_gp,hinmine_deep_gp,hinmine_embedding_n2v,hinmine_laplacian
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

def test_n2v_embedding(graph,delimiter):


    if ".mat" in graph:
        example_net = load_hinmine_object(graph, "---") ## add support for weight
        embedding = hinmine_embedding_n2v(example_net, verbose=True, use_decomposition=False, from_mat=True)
    else:        
        example_net = load_hinmine_object(graph,"---") ## add support for weight    
        ## split and re-weight
        print("Beginning decomposition..")

        decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)
        ## embedding
        print("Starting embedding..")
        embedding = hinmine_embedding_n2v(decomposed,verbose=True)

        print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

        rs = ShuffleSplit(10, test_size=0.5,random_state=42)
        results = []
        v = LogisticRegression(penalty="l2")
        v = OneVsRestClassifier(v)
        batch = 0
        scores_micro = []
        scores_macro = []
        threshold = 0.5
    
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

        print("Finished test - n2v classification basic")    
        
def test_laplacian_embedding_ff(graph,delimiter):

    if ".mat" in graph:
        example_net = load_hinmine_object(graph, "---") ## add support for weight
        embedding = hinmine_laplacian(example_net, verbose=True, use_decomposition=False, from_mat=True)
    else:        
        example_net = load_hinmine_object(graph,"---") ## add support for weight

        ## spt and re-weight
        print("Beginning decomposition..")
        decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)

        ## embedding
        print("Starting embedding..")
        embedding = hinmine_laplacian(decomposed,verbose=True)

    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    results = []
    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)
    batch = 0
    scores_micro = []
    scores_macro = []
    threshold = 0.5
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1        
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = baseline_dense_model(train_X, train_Y,test_X,vtag=0)
        model_preds[model_preds > threshold] = 1
        model_preds[model_preds <= threshold] = 0
        sc_micro = f1_score(test_Y, model_preds, average='micro')
        sc_macro = f1_score(test_Y, model_preds, average='macro')
        print(sc_micro,sc_macro)
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - laplacian classification basic")    


        
def test_laplacian_embedding(graph,delimiter):


    if ".mat" in graph:
        example_net = load_hinmine_object(graph, "---") ## add support for weight
        embedding = hinmine_laplacian(example_net, verbose=True, use_decomposition=False, from_mat=True)
    else:        
        example_net = load_hinmine_object(graph,"---") ## add support for weight    
        ## spt and re-weight
        print("Beginning decomposition..")

        decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)    
        ## embedding
        print("Starting embedding..")
        embedding = hinmine_laplacian(decomposed,verbose=True)

    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    results = []
    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)
    batch = 0
    scores_micro = []
    scores_macro = []
    threshold = 0.5
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1        
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = convolutional_model(train_X, train_Y,test_X,vtag=0)
        model_preds[model_preds > threshold] = 1
        model_preds[model_preds <= threshold] = 0
        sc_micro = f1_score(test_Y, model_preds, average='micro')
        sc_macro = f1_score(test_Y, model_preds, average='macro')
        print(sc_micro,sc_macro)
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - laplacian classification basic")    
        

        
def test_weighted_embedding(graph,delimiter):

    print("Weighted embedding test - weighted")
    example_net = load_hinmine_object(graph,"---",weight_tag="weight") ## add support for weight
    print("embedding in progress..")
    embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=False)
    

def test_deep_pr_classification(graph,delimiter):
    
    if ".mat" in graph:
        example_net = load_hinmine_object(graph,delimiter) ## add support for weight
        embedding = hinmine_embedding_pr(example_net, parallel=True,verbose=True,use_decomposition=False,from_mat=True)
    else:        
        embedding = decompose_test(graph,"---")
    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    ##### learn #####
    
    rs = ShuffleSplit(3, test_size=0.5,random_state=42)
    batch = 0        
    threshold = 0.5
    models_results = []

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embedding['data'] = scaler.fit_transform(embedding['data'])
    
    for train_index, test_index in rs.split(embedding['targets']):
        
        batch += 1
        print("Fold: {}".format(batch))
                       
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]

        ## for m in models...

        preds = convolutional_model(train_X, train_Y,test_X,vtag=0)
#        preds = baseline_dense_model(train_X, train_Y,test_X,vtag=0)
        preds[preds>=threshold] = 13
        preds[preds<threshold] =  0
        sc_micro = f1_score(test_Y, preds, average='micro')
        sc_macro = f1_score(test_Y, preds, average='macro')
        print("This fold's scores; micro: {}, macro: {}".format(sc_micro,sc_macro))
        models_results.append((sc_micro,sc_macro))

    micros = []
    macros = []
    for v in models_results:
        if v[1] > 0:
            micros.append(v[0])
            macros.append(v[1])
        
    print("Model: {} micro: {} macro: {}".format("base",np.mean(micros),np.mean(macros)))
    
    print("Finished test - deep learning..")


def test_deep_graphlet_classification(graph, delimiter):

    if ".mat" in graph:
        example_net = load_hinmine_object(graph, delimiter) ## add support for weight
        embedding = hinmine_deep_gp(example_net, verbose=True, use_decomposition=False, from_mat=True)
    else:        
            example_net = load_hinmine_object(graph,"---") ## add support for weight    
            ## split and re-weight
            print("Beginning decomposition..")

            decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)    
            ## embedding
            print("Starting embedding..")
            embedding = hinmine_deep_gp(decomposed,verbose=True)

            
    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))

    rs = ShuffleSplit(10, test_size=0.5,random_state=42)    
    results = []
    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)
    batch = 0
    scores_micro = []
    scores_macro = []
    threshold = 0.5
    
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

    print("Finished test - deep graphlet-based classification basic")
    

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
        embedding = hinmine_embedding_pr(example_net,
                                         parallel=True,
                                         verbose=True,
                                         use_decomposition=False,
                                         from_mat=True,
                                         feature_permutator_first="0001",
                                         deep_embedding=True)
    else:        
        example_net = load_hinmine_object(graph,"---") ## add support for weight
    
        ## split and re-weight
        print("Beginning decomposition..")

        decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)
    
        ## embedding
        print("Starting embedding..")
        embedding = hinmine_embedding_pr(decomposed,
                                         parallel=True,
                                         verbose=True,
                                         use_decomposition=True,
                                         from_mat=False,
                                         feature_permutator_first="0001",
                                         deep_embedding=True)

    print("Trainset dimension {}, testset dimension {}.".format(embedding['data'].shape,embedding['targets'].shape))


    ## 10 splits 50% train
    
    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    
    results = []

    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)

    batch = 0

    threshold = 0.5
    
    #sel = preprocessing.StandardScaler()

    scores_micro = []
    scores_macro = []
    #embedding['data'] = sel.fit_transform(embedding['data'])
    
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
        print(sc_micro,sc_macro)
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - classification basic")




def test_label_propagation():

    example_net = load_hinmine_object("../data/imdb_gml.gml","---") ## add support for weight
    ## split and re-weight
    print("Beginning decomposition..")   
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None)
    print("Beginning label propagation..")
    pmat = run_label_propagation(decomposed,weights="balanced")
    print(pmat)

    

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
    parser.add_argument("--test_weighted_embedding")
    parser.add_argument("--test_deepgp")
    parser.add_argument("--test_n2v")
    parser.add_argument("--test_laplacian")
    parser.add_argument("--test_laplacian_ff")
    
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

    if args.test_weighted_embedding:
        test_weighted_embedding(args.graph,args.delimiter)

    if args.test_deepgp:
        test_deep_graphlet_classification(args.graph,args.delimiter)

    if args.test_n2v:
        test_n2v_embedding(args.graph,args.delimiter)

    if args.test_laplacian:
        test_laplacian_embedding(args.graph,args.delimiter)

    if args.test_laplacian_ff:
        test_laplacian_embedding_ff(args.graph,args.delimiter)
