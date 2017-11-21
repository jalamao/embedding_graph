## this is a simple interface, which runs a node2vec instance in order to obtain an embedding.
import networkx as nx
import numpy as np
from subprocess import call
import os

def get_n2v_embedding(graph, binary):

    ## construct the embedding and return the binary..
    #./node2vec -i:graph/karate.edgelist -o:emb/karate.emb -l:3 -d:24 -p:0.3 -dr -v

    ## get the graph..
    G = nx.from_scipy_sparse_matrix(graph,edge_attribute='weight')
    for e in G.edges():
        if e[0] == e[1]:
            G.remove_edge(e[0],e[0])
    
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    tmp_graph = "tmp/tmpgraph.edges"
    out_graph = "tmp/tmpgraph.emb"

    number_of_nodes = len(G.nodes())
    number_of_edges = len(G.edges())

    print("Graph has {} edges and {} nodes.".format(number_of_edges,number_of_nodes))
    ## n e + for loop..
    f= open(tmp_graph,"w+")
    #f.write(str(number_of_nodes)+" "+str(number_of_edges)+"\n")
    for e in G.edges(data=True):
        f.write(str(e[0])+" "+str(e[1])+" "+str(e[2]['weight'])+"\n")
    f.close()

    print("Starting graphlet counts..")
    call([binary, "-i:"+tmp_graph, "-o:"+out_graph, "-l:3","-d:128","-p:0.3","-dr","-v"])
    matf = np.loadtxt(out_graph,delimiter=" ", skiprows=1)
    call(["rm","-rf","tmp"])
    print("Finished n2v:",matf.shape)
    return matf
