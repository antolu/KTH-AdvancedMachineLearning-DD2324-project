import numpy as np
import math as m
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

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

    vectorizer = TfidfVectorizer(
                                input='content', 
                                norm="l2", tokenizer=None, 
                                preprocessor=None, 
                                analyzer='char', 
                                ngram_range=(n, n)
                                )

    s = vectorizer.fit_transform([doc1, doc2])

    # Value of the kernel
    val = s[0].dot(s[1].T)

    # Return normalized kernel entry
    return val[0,0]

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

    for n1 in range(N) :
        for n2 in range(N) :
            if n2 < n1 :
                continue
            val = kernel(documents[n1], documents[n2], n)

            kernel_matrix[n1, n2] = val
            kernel_matrix[n2, n1] = val

    return kernel_matrix

