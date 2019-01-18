import dataset_preprocessing
# import data_io
#from SSK_Kernel import SSK
import kernels
from constants import *
import numpy as np
# from copy import deepcopy
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

    print(n1, n2)

    print(os.getpid())
    val = ssk.SSK(str(doclist1[n1]), str(doclist2[n2]), l,n)

    shared_array[n1][n2] = val
    # shared_array[n2][n1] = val
    print(val)
    print("Finished: " + str(os.getpid))


def parallel(n1,n2,l,n, shared_array, train, test):

    print(n1, n2)

    val = ssk.SSK(str(train[n1]), str(test[n2]), l,n)

    shared_array[n1][n2] = val


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":

    """ If you haven't loaded the data uncomment the two lines below and comment everything below out."""
    data = preprocessData()
    # data_io.save_data(data)

    for set_ in DATA_SPLIT: 
        for cat in CATEGORIES : 
            data[set_][cat].sort(key=len)

    #print("main")
    # all_docs_train = []
    # classes = []
    # data = deepcopy(data_io.load_data("train", "acq")[:5])
    # all_docs_train += data
    # classes += ["acq"]*len(data)

    # data = deepcopy(data_io.load_data("train", "earn")[:5])
    # all_docs_train += data
    # classes += ["earn"]*len(data)
    
    # data = deepcopy(data_io.load_data("train", "crude")[:5])
    # all_docs_train += data
    # classes += ["crude"]*len(data)
    
    # data = deepcopy(data_io.load_data("train", "corn")[:5])
    # all_docs_train += data
    # classes += ["corn"]*len(data)

    """all_docs_test = []
    data = deepcopy(data_io.load_data("test", "acq")[:5])
    all_docs_test += data
    data = deepcopy(data_io.load_data("test", "earn")[:5])
    all_docs_test += data
    
    data = deepcopy(data_io.load_data("test", "crude")[:5])
    all_docs_test += data
    
    data = deepcopy(data_io.load_data("test", "corn")[:5])
    all_docs_test += data"""

    train_docs = list()
    test_docs = list()

    NUM_DOCS = 10

    classes = []
    for cat in CATEGORIES : 
        for i in range(NUM_DOCS) : 
            train_docs.append(data[TRAIN][cat][i])
        classes += [cat] * NUM_DOCS

    for cat in CATEGORIES : 
        for i in range(10) : 
            test_docs.append(data[TEST][cat][i])

    np.save('classes.npy', classes)

    N1 = len(train_docs)
    N2 = len(test_docs)

    n = 5
    l_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]

    for l in l_list :

        train_result = np.ctypeslib.as_ctypes(np.zeros((N1, N1)))
        train_shared_array = sharedctypes.RawArray(train_result._type_, train_result)

        traintest_result = np.ctypeslib.as_ctypes(np.zeros((N1, N2)))
        traintest_shared_array = sharedctypes.RawArray(traintest_result._type_, traintest_result)

        testtest_result = np.ctypeslib.as_ctypes(np.zeros((N2, N2)))
        testtest_shared_array = sharedctypes.RawArray(testtest_result._type_, testtest_result)

        ps = []

        # Train-train
        traintrain = []
        for n1 in range(N1) :
            for n2 in range(N1) :
                if n2 < n1:
                    continue
                traintrain.append((n1,n2, l,n,train_shared_array, train_docs, train_docs,))
        
        for num in traintrain:
            ps.append(Process(target=parallel, args=num))

        # Test-train
        traintest = []
        for n1 in range(N1) :
            for n2 in range(N2) :
                traintest.append((n1,n2, l,n,traintest_shared_array, train_docs, test_docs,))
            
        for num in traintest:
            ps.append(Process(target=parallel, args=num))

        # Testtest
        testtest = []
        for n1 in range(N1) :
            testtest.append((n1,n1, l,n,testtest_shared_array, test_docs, test_docs,))
            
        for num in testtest:
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

        # Recover normal numpy arrays
        train_res = np.ctypeslib.as_array(train_shared_array)
        traintest_res = np.ctypeslib.as_array(traintest_shared_array)
        testtest_res = np.ctypeslib.as_array(testtest_shared_array)

        # Make symmetric
        train_res = kernels.make_symmetric(train_res)

        # Normalize
        train_res = kernels.normalize_symmetric(train_res)
        traintest_res = kernels.normalize_asymmetric(traintest_res, train_res, testtest_res)

        # Save matrices
        np.save("train-train_l" + str(l), train_res)
        np.save("train-test_l" + str(l), traintest_res)

        #return shared_array

        #ssk.SSK(doc1,doc2,lambda,n)