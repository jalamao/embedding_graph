## this decomposes a multiplex into singleplex network..

from libHIN.IO import load_hinmine_object ## gml_parser
from libHIN.decomposition import * ## basic embedding
import scipy.io as sio

def decompose_test(fname,out,name):
    example_net = load_hinmine_object(fname,"---") ## add support for weight
    
    ## split and re-weight
    print("Beginning decomposition..")
    # c2 = ["movie_____features_____person_____acts_in_____movie"]
    heuristic_list = ["idf","tf","chi","ig","gr","delta","rf","okapi"]
    for h in heuristic_list:
        dout= hinmine_decompose(example_net,heuristic=h, cycle=None, parallel=True)
        net = dout.decomposed['decomposition']
        labels = dout.label_matrix
        sio.savemat(out+name+"_"+h+".mat", {'group':labels,'network':net})
    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gml")
    parser.add_argument("--outfolder")
    parser.add_argument("--name")
    args = parser.parse_args()
    decompose_test(args.gml,args.outfolder,args.name)
