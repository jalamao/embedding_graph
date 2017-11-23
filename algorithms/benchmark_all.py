## benchmarks from the paper

from deepR import *
import scipy.io
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def classification_benchmark(graphfile, outfile):

    dname = graphfile.split(".")[2].split("/")[-1]
    
    ## direct decomposition    
    mat = scipy.io.loadmat(graphfile)
    labels= mat['group']
    core_network= mat['network']

    ## train the embedding here..
    embedders = [
        DeepR("default"),
        DeepR("no_community"),
        DeepR("no_deep"),
        DeepR("no_deep_no_community")
    ]

    classifiers = {'LR' : LogisticRegression(penalty="l2"),'SVM' :svm.SVC(probability=True,C=10,kernel="rbf")}
    
    results = []
    for k,v in classifiers.items():
        for emb in embedders:
            embedding, type_of_embedding = emb.learn_embedding(core_network,labels)            
            rs = ShuffleSplit(10, test_size=0.5,random_state=42)
            v = OneVsRestClassifier(v)
            batch = 0
            threshold = 0.5    
            scores_micro = []
            scores_macro = []
        
            for train_index, test_index in rs.split(labels):
                batch += 1        
                train_X = embedding['data'][train_index]
                train_Y = embedding['targets'][train_index]
                test_X = embedding['data'][test_index]
                test_Y = embedding['targets'][test_index]
                model_preds = v.fit(train_X,train_Y).predict_proba(test_X)
                model_preds[model_preds >= threshold] = 1
                model_preds[model_preds < threshold] = 0
                sc_micro = f1_score(test_Y, model_preds, average='micro')
                sc_macro = f1_score(test_Y, model_preds, average='macro')
                scores_micro.append(sc_micro)
                scores_macro.append(sc_macro)
        
            rtup = (k,
                    type_of_embedding,
                    np.around(np.mean(scores_micro),decimals=4),
                    np.around(np.std(scores_micro),decimals=4),
                    np.around(np.mean(scores_macro),decimals=4),
                    np.around(np.std(scores_macro),decimals=4),
                    dname)
            
            results.append(rtup)

    results = sorted(results, key=lambda tup: tup[5])
    
    for x in results:
        cls,emb ,score_mi,std_mi ,score_ma, std_ma, dataname = x
        outstring = ("{} {} {} ({}) {} ({}) {}\n".format(cls,emb,score_mi,std_mi,score_ma,std_ma,dataname))
        with open(outfile, "a") as myfile:
            myfile.write(outstring)

if __name__ == "__main__":

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph")
    parser.add_argument("--outfile")
    args = parser.parse_args()
    classification_benchmark(args.graph,args.outfile)
