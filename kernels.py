import numpy as np
import math as m
from collections import Counter
from scipy.sparse import csr_matrix
import re
from multiprocessing import Pool, cpu_count, Process, Array, sharedctypes
from SSK_Kernel import SSK
import pickle
import ctypes

#regex = re.compile(r"*\s")

def ngk(doc1, doc2, n) : 
    """
    n-grams algorithm

    Computes the n-grams features of given documents

    Parameters
    ----------
    doc1 : str
        The first document.
    doc2 : str
        The second document.
    n : The n in n-grams

    Returns
    -------
    A kernel entry
    """

    # Remove whitespace in string
    # doc1 = regex.sub("", doc1)
    # doc2 = regex.sub("", doc2)

    # Extract all the n-grams
    ngrams1 = list()
    for i in range(len(doc1)-n) :
        ngrams1.append(doc1[i:i+n])
    ngrams2 = list()
    for i in range(len(doc2)-n) :
        ngrams2.append(doc2[i:i+n])

    # Count n-grams
    ngram1_count = Counter(ngrams1)
    ngram2_count = Counter(ngrams2)

    # Find shared ngrams
    shared_ngrams = set(ngram1_count.keys()).intersection(set(ngram2_count.keys()))

    # Count occurrence of shared n-grams
    shared_ngrams_sum = sum([ngram1_count[ngram] * ngram2_count[ngram] for ngram in shared_ngrams])

    norm1 = np.linalg.norm(np.asarray(list(ngram1_count)))
    norm2 = np.linalg.norm(np.asarray(list(ngram2_count)))

    similarity = shared_ngrams_sum / m.sqrt(norm1 * norm2)

    return similarity

def compute_matrix(documents, kernel='ngk', n=2) :
    """
    Computes the kernel matrix of a list of documents.

    Parameters
    ----------
    documents : list() [string]
        A list of documents to train the kernel on.
    kernel : ngk, bwk, ssk
        The kernel function to be used as similarity measure for the kernel matrix.
    n : int
        The n used in each kernel. n-grams, ssk with n-characters etc.
    """

    # Check input arguments
    if kernel == "ngk" :
        kernel = ngk
    elif kernel == "ssk" :
        kernel = ngk
    elif kernel == "bwk" :
        kernel = ngk
    else : 
        raise Exception("Kernel " + kernel + " is not a valid kernel")

    if n < 1 :
        raise Exception("n: " + str(n) + " is not a valid value for n.")


    # Compute the actual matrix (symmetric)
    N = len(documents)

    kernel_matrix = np.zeros((N, N))
    for n1 in range(N):
        for n2 in range(N) :
            if n2 < n1 :
                continue
            val = kernel(documents[n1], documents[n2], n)

            kernel_matrix[n1, n2] = val
            kernel_matrix[n2, n1] = val

    return kernel_matrix

def call_kernel(i, j, n, documents, shared_array):
    #print("hellll")
    kernel = SSK(2, 0.2, 0.2, "h", "h")
    #print("hello")
    val = kernel.k(documents[i][:100], documents[j][:100], n)
    shared_array[i][j] = val
    shared_array[j][i] = val
    #print(shared_array.value)
    #print(val)
    return val


def parallel_matrix_compute(documents, kernel = ngk, n = 2):
    #kernel = SSK(2, 0.2, 0.2,"test", "test")
    
    
    
    
    
    print("parallel")
    N = len(documents)
    global kernel_matrix
    kernel_matrix = np.zeros((N, N))

    shared_array = Array(ctypes.c_double, (N,N))

    result = np.ctypeslib.as_ctypes(np.zeros((N, N)))
    shared_array = sharedctypes.RawArray(result._type_, result)






    print(shared_array)
    doc_len = len(documents)

    indecies = []
    for i in range(doc_len):
        for j in range(doc_len):
            indecies.append((i,j,n, documents,shared_array, ))
    print("for loop passed")

    """pool = Pool(cpu_count()-1)
    print((pool.map(call_kernel, indecies)))
    pool.close()
    pool.join()"""

    #lock = Lock()

    for num in indecies:
        p = Process(target=call_kernel, args=num)
        p.start()
    p.join()

    #f=open('ssk_matrix','w')
    #pickle.dump(kernel_matrix, f)
    #f.close()
    res = np.ctypeslib.as_array(shared_array)
    print(res)
    np.save('kernel_matrix.npy', res)

    



