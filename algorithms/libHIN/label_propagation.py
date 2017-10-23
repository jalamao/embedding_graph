## label propagation algorithms:
import networkx as nx
import numpy as np
import scipy.sparse as sp
from .dataStructures import HeterogeneousInformationNetwork

def label_propagation_normalization(matrix):
    matrix = matrix.tocsr()
    try:
        matrix.setdiag(0)
    except TypeError:
        matrix.setdiag(np.zeros(matrix.shape[0]))
    d = matrix.sum(axis=1).getA1()
    nzs = np.where(d > 0)
    d[nzs] = 1 / np.sqrt(d[nzs])
    dm = sp.diags(d, 0).tocsc()
    return dm.dot(matrix).dot(dm)


## dodaj numba compiler tule
def label_propagation(graph_matrix, class_matrix, alpha, epsilon=1e-12, max_steps=10000):
    # This method assumes the label-propagation normalization and a symmetric matrix with no rank sinks.
    steps = 0
    diff = np.inf
    current_labels = class_matrix
    while diff > epsilon and steps < max_steps:
        steps += 1
        new_labels = alpha * graph_matrix.dot(current_labels) + (1 - alpha) * class_matrix
        diff = np.linalg.norm(new_labels - current_labels) / np.linalg.norm(new_labels)
        current_labels = new_labels
    return current_labels


## tu mu dodaj set index-ov, ki jih ima za train/test
def run_label_propagation(hin, weights=None, alpha=0.85, semibalanced=None):
    assert isinstance(hin, HeterogeneousInformationNetwork)
    matrix = label_propagation_normalization(hin.decomposed['decomposition'])
    hin.create_label_matrix(weights=weights)    
#    hin.label_matrix[0:50] = 0
    propagated_matrix = label_propagation(matrix, hin.label_matrix, alpha)
    return propagated_matrix
