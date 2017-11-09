## this is the interface to the orca algorithm

import networkx as nx
import numpy as np
import os
from subprocess import call
from collections import defaultdict

def count_graphlets_orca(graph,binary):

    ## get the graph..
    G = nx.from_scipy_sparse_matrix(graph)
    for e in G.edges():
        if e[0] == e[1]:
            G.remove_edge(e[0],e[0])
    
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    tmp_graph = "tmp/tmpgraph.in"
    out_graph = "tmp/tmpgraph.out"

    number_of_nodes = len(G.nodes())
    number_of_edges = len(G.edges())

    ## n e + for loop..
    f= open(tmp_graph,"w+")
    f.write(str(number_of_nodes)+" "+str(number_of_edges)+"\n")
    for e1,e2 in G.edges():
        f.write(str(e1)+" "+str(e2)+"\n")
    f.close()

    print("Starting graphlet counts..")
    call([binary, "5", tmp_graph, out_graph])
    matf = np.loadtxt(out_graph,delimiter=" ")
    call(["rm","-rf","tmp"])
    print("Finished graphlet counting:",matf.shape)
    return matf


if __name__ == "__main__":

    pass
