import numpy as np
from kernels import *
from dataset_preprocessing import *
from data_io import *
from constants import *
from ssk_ap import *
import time
if __name__ == "__main__":

 
    untrimmed_docs = load_raw_data()
    
    trimmed_docs = preprocess(untrimmed_docs, normalize=True)
    
    t = trimmed_docs["train"]["acq"][1]
    
    # save_data(trimmed_docs)
    
    # do = load_data("train", "acq")
    
    # print(do[1])
    
    s = trimmed_docs[TRAIN][ACQ][0]
    t = trimmed_docs[TRAIN][ACQ][1]
    
    base = get_features([s,t], 40, 3)
    
    start = time.time()
    kern1 = ssk_ap(s,t,base,0.2,3)
    end = time.time()
    print("val: " + str(kern1) + ". Time: " + str(end-start))
    base = get_features([s], 40, 3)
    
    start = time.time()
    d1 = ssk_ap(s,s,base,0.2,3)
    end = time.time()
    print("val: " + str(kern1) + ". Time: " + str(end-start))
    base = get_features([t], 40, 3)
    
    start = time.time()
    d2 = ssk_ap(t,t,base,0.2,3)
    end = time.time()
    print("val: " + str(kern1) + ". Time: " + str(end-start))
    alignment = kern1/(np.power(d1*d2, 0.5))

    print(alignment)


    print("val: " + str(kern1) + ". Time: " + str(end-start))
    
    ssk = SSK(3, 0.2, 10, s, t)
    start = time.time()
    kern2 = ssk.k(s,s,3)
    end = time.time()
    print("val: " + str(kern2) + ". Time: " + str(end-start))
    
    
    # gram_matrices = {}
    
    # for cat in CATEGORIES :
    #     l = trimmed_docs["train"][cat]
    
    #     gram_matrices[cat] = compute_matrix(l, kernel="ngk", n=3)