from HINMINE.library_cli import *
from HINMINE.lib.HIN import HeterogeneousInformationNetwork
import networkx as nx
import gzip as gz

def read_rfa():
    G = nx.Graph()
    with open("../data/rfa_all.NL-SEPARATED.txt") as vote:

        ## init parts
        currentN0 = ""
        currentN1 = ""
        etype = ""
        cn1 = 0
        cn2 = 0
        for line in vote:
            parts = line.split(":")
            if len(parts) == 1:
                currentN0 = ""
                currentN1 = ""
                etype = ""
            else:
                if parts[0] == "SRC":
                    cn1+=1
                    currentN0 = parts[1]
                elif parts[0] == "TGT":
                    cn2+=1
                    currentN1 = parts[1]              
                elif parts[0] == "VOT":
                    G.add_node(cn1,type=etype,label="test",labels="test_test2")
                    G.add_node(cn2,type="being_voted",label="test2")
                    G.add_edge(cn2,cn2,type="being_voted")
        return G


if __name__ == "__main__":

    voting_graph = read_rfa()    
    # print(nx.info(voting_graph))
    tmp_path = "tmp.gml"
    nx.write_gml(voting_graph,tmp_path)
    example_graph = load_gml("tmp.gml","_")
    decomposed = hinmine_decompose(example_graph,heuristic="idf")
