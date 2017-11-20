## compare all methods
import argparse
from gem.utils import graph_util
from libHIN.IO import load_hinmine_object, generate_cv_folds  ## gml_parser
from libHIN.embeddings import *
from libHIN.decomposition import * ## basic embedding
import networkx as nx
from gem.embedding.gf       import GraphFactorization
from gem.embedding.hope     import HOPE
#from gem.embedding.lap      import LaplacianEigenmaps
from gem.embedding.lle      import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne     import SDNE
from time import time


def decompose_and_test(fname):
    example_net = load_hinmine_object(fname,"---") ## add support for weight    
    ## split and re-weight
    print("Beginning decomposition..")   
    decomposed = hinmine_decompose(example_net,heuristic="idf", cycle=None, parallel=True)

    G = nx.from_scipy_sparse_matrix(example_net.decomposed['decomposition'],edge_attribute="weight")
    G = G.to_directed()
    
    models = []
    models.append(GraphFactorization(2, 100000, 1*10**-4, 1.0))
    models.append(HOPE(4, 0.01))
# #    models.append(LaplacianEigenmaps(2))
#     models.append(LocallyLinearEmbedding(2))
#     models.append(node2vec(2, 1, 80, 10, 10, 1, 1))
#     models.append(SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,n_units=[50, 15,], rho=0.3, n_iter=50, xeta=0.01,n_batch=500,modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5']))

    models.append(hinmine_embedding(method="pagerank",augmentation="community"))

    for method in models:
        t1 = time()

        # Learn embedding - accepts a networkx graph or file with edge list
        X, t = method.learn_embedding(graph=G,edge_f=None, is_weighted=True, no_python=True)
        
        print ("time elapsed {}".format((time() - t1)))
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_all")
    parser.add_argument("--graph")
    args = parser.parse_args()

    if args.test_all:
        decompose_and_test(args.graph)

    
