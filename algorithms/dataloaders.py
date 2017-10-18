##########################
import networkx as nx
import numpy as np

def read_rfa(fname):
    G = nx.MultiDiGraph()
    with open(fname) as vote:

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
                    G.add_edge(currentN0,currentN1,type="voted_by")
    return G

def read_bitcoin(fname):

    G = nx.MultiDiGraph()
    with open(fname) as fn:
        for line in fn:
            source,target,weight,timestamp = line.strip().split(",")
            label = ""
            if int(weight) >= 0:
                label = "+"
            else:
                label = "-"
            G.add_node(source, type="agent",labels="w1")
            G.add_node(target, type="agent",labels="opposite")
            G.add_edge(source,target,type="sended_money_to",weight=np.absolute(int(weight)/10))

    return G

def read_web(fname):

    G = nx.MultiDiGraph()
    with open(fname) as fn:
        for line in fn:
            try:
                source, target = line.strip().split()
                G.add_edge(source,target,weight=1)
            except:
                pass

    return G
            
if __name__ == "__main__":

    ## read the graph1
    #voting_graph = read_rfa("../data/smaller.txt")
    rx = read_bitcoin("../data/soc-sign-bitcoinotc.csv")
    print(nx.info(rx))
