import numpy as np
import math as m
from collections import Counter
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from ssk_ap import SSK_AP, get_features, count_occurrences
from SSK_Kernel import SSK
import re
from WK import WK

regex = re.compile(r"\s")

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

    norm1 = np.linalg.norm(np.asarray(list(ngram1_count.values())))
    norm2 = np.linalg.norm(np.asarray(list(ngram2_count.values())))

    similarity = shared_ngrams_sum / m.sqrt(norm1 * norm2)

    return similarity

def wk(doc1, doc2, n) :
    """
    bag-of-words algorithm

    Computes the bag-of-words features of given documents

    Parameters
    ----------
    doc1 : str
        The first document.
    doc2 : str
        The second document.
    n : not used

    Returns
    -------
    A kernel entry
    """

    # Extract all the n-grams
    words1 = word_tokenize(doc1)
    words2 = word_tokenize(doc2)

    # Count n-grams
    word1_count = Counter(words1)
    word2_count = Counter(words2)

    # Find shared ngrams
    shared_words = set(word1_count.keys()).intersection(set(word2_count.keys()))

    # Count occurrence of shared n-grams
    shared_words_sum = sum([word1_count[word] + word2_count[word] for word in shared_words])

    norm1 = np.linalg.norm(np.asarray(list(word1_count.values())))
    norm2 = np.linalg.norm(np.asarray(list(word2_count.values())))

    similarity = shared_words_sum / m.sqrt(norm1 * norm2)

    return similarity

def compute_matrix(documents, kernel="ngk", n=2, x=100, l=0.2) :
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
    x : int
        Number of n-grams in ssk_ap
    l : double
        The decay factor
    """

    # Check input arguments
    if kernel == "ngk" :
        kernel = ngk
    elif kernel == "ssk" :
        ssk = SSK(n, l, x, "", "")
        kernel = ssk.k
    elif kernel == "wk" :
        bow = WK(documents)
        kernel = bow.kernel_function
    elif kernel == "ssk_ap" :
        base = get_features(documents, x, n)
        ssk_ap = SSK_AP(base, l)
        kernel = ssk_ap.ssk_ap
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

    # Normalize
    for n1 in range(N) :
        for n2 in range(N) :
            if n2 <= n1 :
                continue
            kernel_matrix[n1, n2] = kernel_matrix[n1, n2] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])
            kernel_matrix[n2, n1] = kernel_matrix[n2, n1] / m.sqrt(kernel_matrix[n1, n1] * kernel_matrix[n2, n2])

    # Normalize diagonal
    for n in range(N) :
        kernel_matrix[n, n] = 1.0

    return kernel_matrix


def compute_nonsym_matrix(doclist1, doclist2, kernel="ngk", n=2, x=100, l=0.2) :
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
    x : int
        Number of n-grams in ssk_ap
    l : double
        The decay factor
    """

    # Check input arguments
    if kernel == "ngk" :
        kernel = ngk
    elif kernel == "ssk" :
        ssk = SSK(n, l, x, "", "")
        kernel = ssk.k
    elif kernel == "wk" :
        bow = WK(doclist1)      # Assuming doclist1 is train data
        kernel = bow.kernel_function
    elif kernel == "ssk_ap" :
        base = get_features(documents, x, n)
        ssk_ap = SSK_AP(base, l)
        kernel = ssk_ap.ssk_ap
    else :
        raise Exception("Kernel " + kernel + " is not a valid kernel")

    if n < 1 :
        raise Exception("n: " + str(n) + " is not a valid value for n.")


    # Compute the actual matrix (asymmetric)
    N1 = len(doclist1)
    N2 = len(doclist2)
    kernel_matrix = np.zeros((N1, N2))

    for n1 in range(N1) :
        for n2 in range(N2) :

            val = kernel(doclist1[n1], doclist2[n2], n)
            #Normalize
            kernel_matrix[n1, n2] = val / m.sqrt(kernel(doclist1[n1], doclist1[n1], n) * kernel(doclist2[n2], doclist2[n2], n))


    return kernel_matrix
