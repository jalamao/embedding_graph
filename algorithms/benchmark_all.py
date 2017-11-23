## benchmarks from the paper

def test_classification(graphfile):
    
    ## direct decomposition
    

    import scipy.io
    mat = scipy.io.loadmat(graphfile)
    labels= mat['group']
    core_network= mat['network']

    ## train the embedding here..

    
    
    ## 10 splits 50% train
    
    rs = ShuffleSplit(10, test_size=0.5,random_state=42)
    
    results = []

    v = LogisticRegression(penalty="l2")
    v = OneVsRestClassifier(v)

    batch = 0
    threshold = 0.5    
    scores_micro = []
    scores_macro = []
    
    for train_index, test_index in targets):
        
        batch += 1
        
        print("Fold: {}".format(batch))
        train_X = embedding['data'][train_index]
        train_Y = embedding['targets'][train_index]
        test_X = embedding['data'][test_index]
        test_Y = embedding['targets'][test_index]
        model_preds = v.fit(train_X,train_Y).predict_proba(test_X)
        model_preds[model_preds > threshold] = 1
        model_preds[model_preds <= threshold] = 0
        sc_micro = f1_score(test_Y, model_preds, average='micro')
        sc_macro = f1_score(test_Y, model_preds, average='macro')
        print(sc_micro,sc_macro)
        scores_micro.append(sc_micro)
        scores_macro.append(sc_macro)
        
    results.append(("LR, t:{}".format(str(threshold)),np.mean(scores_micro),np.mean(scores_macro)))

    results = sorted(results, key=lambda tup: tup[2])
    
    for x in results:
        cls, score_mi, score_ma = x
        print("Classifier: {} performed with micro F1 score {} and macro F1 score {}".format(cls,score_mi,score_ma))

    print("Finished test - classification basic")
