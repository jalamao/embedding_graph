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


def read_ecommerce(data,labels):

    from collections import defaultdict

    nodes = defaultdict(list)
    labels_list = []

    with open(labels) as labs:
        for line in labs:
            labels_list.append(line.strip())

    with open(data) as maindata:

        for ind, line in enumerate(maindata):            
            parts = line.strip().split(" ")
            try:
                
                gender = labels_list[ind]
                id_ind=parts[0].split(",")[0]
                items = parts[-1].split(",")[1]
                id_individual = parts[-1].split(",")[0]
                individual_items = items.split(";")

                first_order = [item.split("/")[0] for item in individual_items]
                second_order = [item.split("/")[1] for item in individual_items]
                third_order = [item.split("/")[2] for item in individual_items]
                fourth_order = [item.split("/")[3] for item in individual_items]

                if ind not in nodes:
                    nodes[ind] = {'gender' : "", 'items' : {'A' : [], 'B' : [], 'C' : [], 'D' : []}}
                nodes[ind]['gender'] = gender
                for x in first_order:
                    nodes[ind]['items']['A'].append(x)
                for x in second_order:
                    nodes[ind]['items']['B'].append(x)
                for x in third_order:
                    nodes[ind]['items']['C'].append(x)
                for x in fourth_order:
                    nodes[ind]['items']['D'].append(x)
                
            except:
                pass

    ## obravnavaj kot bipartitnigraf brez autociklov, pa bo.
            
    G = nx.MultiDiGraph()
    for n1,data1 in nodes.items():
        G.add_node(n1,type='purchase',labels=data1['gender'], name=str(n1))
        for item in data1['items']['A']:
            G.add_node(item,type='A_level_item')
            G.add_edge(n1,item,type='purchased_by')
            G.add_edge(item,n1,type='purchased')
        for item in data1['items']['B']:
            G.add_node(item,type='B_level_item')
            G.add_edge(n1,item,type='purchased_by')
            G.add_edge(item,n1,type='purchased')
        for item in data1['items']['C']:
            G.add_node(item,type='C_level_item')
            G.add_edge(n1,item,type='purchased_by')
            G.add_edge(item,n1,type='purchased')
        for item in data1['items']['D']:
            G.add_node(item,type='D_level_item')
            G.add_edge(n1,item,type='purchased_by')
            G.add_edge(item,n1,type='purchased')

    return G
    
if __name__ == "__main__":

    graph = read_ecommerce("../data/ecommerce/dataset.txt","../data/ecommerce/labels.txt")
    nx.write_gml(graph, "../data/ecommerce.gml")
    ## read the graph1
    #voting_graph = read_rfa("../data/smaller.txt")
    #rx = read_bitcoin("../data/soc-sign-bitcoinotc.csv")
    #print(nx.info(rx))
