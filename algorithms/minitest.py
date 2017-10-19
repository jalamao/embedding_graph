## this is a basic test of hinmine capabilities

from HINMINE.library_cli import *
import numpy as np

## load test gml and read it further on..
heuristic = "tf"
#cycle = {'node_list': [u'movie', u'person',u'movie'], 'edge_list': [u'features',u'acts_in']}
cycle = ['movie_____features_____person_____acts_in_____movie']
example_net = load_gml("../data/imdb_gml.gml","---")
decomposed = hinmine_decompose(example_net,heuristic,cycle=cycle)
propositionalized = hinmine_propositionalize(decomposed)
data = propositionalized

train = data['train_features']['data']
target = data['train_features']['target']

## CV classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

## multi target example
forest = RandomForestClassifier(n_estimators=100, random_state=1)
dummy_clf = DummyClassifier(strategy='most_frequent',random_state=0)

print(train,target)

mclass_cls = OneVsRestClassifier(forest)
mclass_cls_dummy = OneVsRestClassifier(dummy_clf)

scores = cross_val_score(mclass_cls, train, target, cv=10)
scores_dummy = cross_val_score(mclass_cls_dummy, train, target, cv=10)
print(np.mean(scores)," % F1 macro.")
print(np.mean(scores_dummy)," % F1 macro.")
        
print("Finished test 2 - classification")
#print(np.count_nonzero(data['data'])) ## to vrne 0, kar ni isto kot CF-workflow
