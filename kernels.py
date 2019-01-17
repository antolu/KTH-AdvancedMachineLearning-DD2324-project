import numpy as np
import math as m
from collections import Counter
from scipy.sparse import csr_matrix
import re
import ssk

# regex = re.compile(r"*\s")

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
        kernel = ssk.SSK
    elif kernel == "bwk" :
        kernel = ngk
    else : 
        raise Exception("Kernel " + kernel + " is not a valid kernel")

    if n < 1 :
        raise Exception("n: " + str(n) + " is not a valid value for n.")


    # Compute the actual matrix (symmetric)
    N = len(documents)

    kernel_matrix = np.zeros((N, N))

    for n1 in range(N) :
        for n2 in range(N) :
            if n2 < n1 :
                continue
            val = kernel(documents[n1], documents[n2], n)

            kernel_matrix[n1, n2] = val
            kernel_matrix[n2, n1] = val

    for n1 in range(N) : 
        for n2 in range(N) :
            if n2 <= n1 :
                continue
            kernel_matrix[n1, n2] = kernel_matrix[n1, n2] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])
            kernel_matrix[n2, n1] = kernel_matrix[n2, n1] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])

    # Normalize diagonal
    for n in range(N) : 
         kernel_matrix[n, n] =  kernel_matrix[n, n] /  kernel_matrix[n, n]

    return kernel_matrix

def compute(documents, l, n=2) :

    # Compute the actual matrix (symmetric)
    N = len(documents)

    kernel_matrix = np.zeros((N, N))

    for n1 in range(N) :
        for n2 in range(N) :
            if n2 < n1 :
                continue
            print(n1, " ", n2)
            print(documents[n1])
            print(documents[n2])

            val = ssk.SSK(str(documents[n1]), str(documents[n2]), l, n)

            kernel_matrix[n1, n2] = val
            kernel_matrix[n2, n1] = val

    for n1 in range(N) : 
        for n2 in range(N) :
            if n2 <= n1 :
                continue
            kernel_matrix[n1, n2] = kernel_matrix[n1, n2] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])
            kernel_matrix[n2, n1] = kernel_matrix[n2, n1] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])

    # Normalize diagonal
    for n in range(N) : 
         kernel_matrix[n, n] =  kernel_matrix[n, n] /  kernel_matrix[n, n]

    return kernel_matrix

def compute_assymmetric(train, test, traintrain, l, n=2) :

    # Compute the actual matrix (symmetric)
    N1 = len(train)
    N2 = len(test)

    kernel_matrix = np.zeros((N1, N2))

    testtest = list()
    for i in range(N2) :
        testtest
    np.save("testtest",testtest)

    for n1 in range(N1) :
        for n2 in range(N2) :
            # if n2 < n1 :
            #     continue
            print(n1, " ", n2)
            print(train[n1])
            print(test[n2])

            val = ssk.SSK(str(train[n1]), str(test[n2]), l, n)

            kernel_matrix[n1, n2] = val / m.sqrt(traintrain[n1, n1] * testtest[n2, n2])

    return kernel_matrix
