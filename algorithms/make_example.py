import networkx as nx
from HINMINE.library_cli import *

a = nx.MultiDiGraph()

a.add_node(1,type='tip2')
a.add_node(2,type='tip2')
a.add_node(3,type='tip1',labels="t")
a.add_node(4,type='tip2')
a.add_node(5,type='tip1',labels="t")

a.add_edge(1,3,type='povezava1')
a.add_edge(2,3,type='povezava1')
a.add_edge(1,4,type='povezava1')
a.add_edge(3,1,type='povezava1')

nx.write_gml(a,"tmp.gml")
## some random testing right there
example_graph = load_gml("tmp.gml"," ")
cycle = {'node_list': [u'tip1', u'tip2', u'tip1'], 'edge_list': [u'povezava1', u'povezava1']}
decomposed = hinmine_decompose(example_graph,heuristic="idf",cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)
print(propositionalized)
