from HINMINE.library_cli import *
import networkx as nx
import gzip as gz

def read_rfa():
    G = nx.MultiDiGraph()
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
                    G.add_node(currentN1,type="person", labels="candidate")
                    G.add_node(currentN0,labels="voter",type='person')
                    G.add_edge(currentN1,currentN0,type="votes_for")
        return G

if __name__ == "__main__":

    ## read the graph
    voting_graph = read_rfa()

    converted = nx.convert_node_labels_to_integers(voting_graph,first_label=0)        
    #nx.write_edgelist(converted, "../data/el.txt")    
    cycle = {'node_list': [u'person'], 'edge_list': [u'votes_for']}
    tmp_path = "tmp.gml"
    nx.write_gml(converted,tmp_path)

    ## some random testing right there
    example_graph = load_gml("tmp.gml"," ")
    decomposed = hinmine_decompose(example_graph,heuristic="idf",cycle=cycle)
    print(decomposed.__dict__.keys())
