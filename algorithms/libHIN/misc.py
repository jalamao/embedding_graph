## this file contains evaluation metrics, commonly used for multiclass classification

import numpy as np

def find_cv_converged_split(Dx,Dy,nfolds):

    incorrect_split = True
    nrow, ncol = Dx.shape
    percentage = 1-1/nfolds
    f1 = int(nrow*percentage)
    converged_folds = 0
    while incorrect_split:        

        ## construct a random split
        ## check if it is valid
        ## if yes, then emit
        
        idx = np.random.randint(nrow, size=f1)        
        inverse = np.ones(nrow, np.bool)
        inverse[idx] = 0        
        train_targets = Dy[idx]
        test_targets = Dy[inverse]        
        train_data = Dx[idx]
        test_data = Dx[inverse]
        completness = np.where(~train_targets.any(axis=1))[0]
        if len(completness) == 0:
            converged_folds += 1
            yield (train_data,test_data,train_targets,test_targets)
