## inspired by Rossi et al., deep feature construction for graphs

import numpy as np
import itertools
from sklearn.metrics import mutual_info_score as kl

# def kl(a, b):
#     a = np.asarray(a, dtype=np.float)
#     b = np.asarray(b, dtype=np.float)
#     return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def hadamand_sum(v1,v2):
    return np.add(v1,v2)

def pairwise_kl(feature,targets):

    total = 0    
    for x in range(targets.shape[1]):
        try:
            ttar = targets[:,x]            
            total+= kl(feature,ttar)
        except:
            pass

    return total

def total_kl(features,targets):
    
    total_kl = 0
    for k in range(features.shape[1]):
        pkl = pairwise_kl(features[:,k],targets)
        if pkl > total_kl:
            total_kl = pkl
            
    return total_kl
            
def combine_features(fdata,tar,beta):

    dims = fdata.shape[1]
    ncycle = 0
    lmin = -1000000
    operators_layer = []
    for comb in itertools.combinations(range(dims), 2):
        ncycle +=1
        if ncycle > 30:
            break

        new_feature = hadamand_sum(fdata[:,comb[0]],fdata[:,comb[1]])
        kl_tar = pairwise_kl(new_feature,tar)
        kl_intra = beta*pairwise_kl(new_feature,fdata)
        kl_total = kl_tar - kl_intra
        
        if kl_total > lmin:
            operators_layer.append(comb)
            lmin=kl_total ## update local max and add a feature
            new_feature = new_feature.reshape((len(new_feature),1))
            fdata = np.append(fdata, new_feature, axis=1)

    return (fdata,operators_layer)

def deep_embedding_gp(gcounts,targets,nlayers=2,beta=0.02):

    operator_trace = []
    for j in range(nlayers):
        print("Currently constructing the depth of {}".format(j))
        gcounts, ops = combine_features(gcounts,targets,beta)
        operator_trace.append(ops)
    print(operator_trace)
    
    return gcounts
