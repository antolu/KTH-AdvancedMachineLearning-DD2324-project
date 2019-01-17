import dataset_preprocessing
import data_io
#from SSK_Kernel import SSK
#import kernels
import constants
import numpy as np
from copy import deepcopy
import ssk
from multiprocessing import Pool, cpu_count, Process, Array, sharedctypes, active_children
import ctypes
import os
import time

def preprocessData():
    raw = dataset_preprocessing.load_raw_data()
    processed = dataset_preprocessing.preprocess(raw)
    return processed

def parallel(n1,n2,l,n, shared_array, doclist1, doclist2):
    #print("in parallel")
    #print(doclist1[i])
    #print(doclist2[j])
    #val = ssk.SSK(str(doclist1[i]), str(doclist2[j]), l,n)
    #Normalize
    #res = val / np.power((ssk.SSK(str(doclist1[i]), str(doclist1[i]), l, n) * ssk.SSK(str(doclist2[j]), str(doclist2[j]), l, n)),0.5)
    #shared_array[i][j] = res
    #shared_array[j][i] = res
    #print(res)
    print(os.getpid())
    val = ssk.SSK(str(doclist1[n1]), str(doclist2[n2]), l,n)

    shared_array[n1][n2] = val
    shared_array[n2][n1] = val
    print(val)
    print("Finished: " + str(os.getpid))


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
if __name__ == "__main__":

    """ If you haven't loaded the data uncomment the two lines below and comment everything below out."""
    data = preprocessData()
    data_io.save_data(data)

    #print("main")
    all_docs_train = []
    classes = []
    data = deepcopy(data_io.load_data("train", "acq")[:5])
    all_docs_train += data
    classes += ["acq"]*len(data)

    data = deepcopy(data_io.load_data("train", "earn")[:5])
    all_docs_train += data
    classes += ["earn"]*len(data)
    
    data = deepcopy(data_io.load_data("train", "crude")[:5])
    all_docs_train += data
    classes += ["crude"]*len(data)
    
    data = deepcopy(data_io.load_data("train", "corn")[:5])
    all_docs_train += data
    classes += ["corn"]*len(data)

    """all_docs_test = []
    data = deepcopy(data_io.load_data("test", "acq")[:5])
    all_docs_test += data
    data = deepcopy(data_io.load_data("test", "earn")[:5])
    all_docs_test += data
    
    data = deepcopy(data_io.load_data("test", "crude")[:5])
    all_docs_test += data
    
    data = deepcopy(data_io.load_data("test", "corn")[:5])
    all_docs_test += data"""

    mergedDocs = []
    mergedDocs.append(all_docs_train)
    mergedDocs.append(all_docs_train)
    #mergedDocs.append(all_docs_train)
    #mergedDocs.append(all_docs_train)
    np.save('classes.npy', classes)

    doclist1 = mergedDocs[0]
    doclist2 = mergedDocs[1]

    N1 = len(doclist1)
    N2 = len(doclist2)
    #shared_array = np.zeros((N1, N2))

    shared_array = Array(ctypes.c_double, (N1,N2))

    result = np.ctypeslib.as_ctypes(np.zeros((N1, N2)))
    shared_array = sharedctypes.RawArray(result._type_, result)

    n = 2
    l = 0.2
    indecies = []
    for n1 in range(N1) :
        for n2 in range(N2) :
            if n2 < n1:
                continue
            indecies.append((n1,n2, l,n,shared_array, doclist1, doclist2,))
    print(indecies)
    
    
    ps = []
    for num in indecies:
        ps.append(Process(target=parallel, args=num))
        
    #for i in chunks(ps,14):
    for i in chunks(ps, cpu_count()-1):
        for p in i:
            p.start()
        while True:
            time.sleep(1)
            if not active_children():
                break

    """for n1 in range(N1) :
        for n2 in range(N2) :
            val = ssk.SSK(str(doclist1[N1]), str(doclist2[j]), l,n)
            #Normalize
            res = val / np.power((ssk.SSK(str(doclist1[N1]), str(doclist1[N1]), l, n) * ssk.SSK(str(doclist2[N2]), str(doclist2[N2]), l, n)),0.5)
            shared_array[N1][N2] = res
            shared_array[N2][N1] = res"""

    for n1 in range(N1) : 
        for n2 in range(N2) :
            if n2 <= n1 :
                continue
            shared_array[n1][n2] = shared_array[n1][n2] / np.sqrt(shared_array[n1][n1] * shared_array[n2][n2])
            shared_array[n2][n1] = shared_array[n2][n1] / np.sqrt(shared_array[n1][n1] * shared_array[n2][n2])

    # Normalize diagonal
    for n in range(N1) : 
         shared_array[n][n] =  shared_array[n][n] /  shared_array[n][n]


    res = np.ctypeslib.as_array(shared_array)
    np.save("kernel_matrix.npy", res)

    #return shared_array

    #ssk.SSK(doc1,doc2,lambda,n)