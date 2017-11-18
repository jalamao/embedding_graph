## this scripts constructs the tissue graph

from collections import defaultdict
import os
import networkx as nx

tissue_folder_networks = "../data/tissues/bio-tissue-networks"
tissue_folder_labels = "../data/tissues/bio-tissue-labels"

G = nx.MultiDiGraph()
for folder in os.listdir(tissue_folder_networks):
    fpath = tissue_folder_networks+"/"+folder
    tname = folder.split(".")[0]
    with open(fpath) as fp:
        for line in fp:
            e1, e2 = line.strip().split()
            G.add_node(e1,type='gene')
            G.add_node(e2,type='gene')
            G.add_edge(e1,e2,type='ground_truth',weight=1)

print(nx.info(G))

tnames = defaultdict(list)
labels_final = defaultdict(list)
for folder in os.listdir(tissue_folder_labels):
    fpath = tissue_folder_labels+"/"+folder
    tissue = folder.split(".")[0].split("_")
    label = tissue[-1]
    tname = "_".join(tissue[0:-1])
    with open(fpath) as pf:
        for line in pf:
            parts = line.strip().split()
            if parts[0] != "#":
                n = parts[0]
                l = parts[1]
                labels_final[n].append(label+"_"+l)


for n in G.nodes(data=True):
    n[1]['labels']="---".join(labels_final[n[0]])

nx.write_gml(G, "../data/tissues.gml")
