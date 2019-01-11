import numpy as np
import math as m
from collections import Counter
from scipy.sparse import csr_matrix
from SSK_Kernel import *
# import itertools

# LETTERS = "abcdefghijklmnopqrstuvwxyz"

# def get_combinations(n) :
#     """
#     Returns all contiguous combinations of strings with
#     length n.
#     """

#     s = itertools.product(LETTERS, repeat=n)

#     l = list()

#     for comb in s :
#         l.append("".join(comb))

#     return l

def get_features(docs, x, n) :
    """
    Calculates an x long base of n-grams for the docs 

    (Finds the most common shared n-grams)

    Parameters
    ----------
    docs : list(string)
        List of documents
    x : int
        x most common n-grams
    n : int
        Length of n-grams
    """

    cnt = Counter()
    
    for doc in docs :
        cnt += count_occurrences(doc, n)

    # print(cnt)
    final_counts = dict(cnt)

    most_common = cnt.most_common(x)

    most_common_ngrams = [ngram[0] for ngram in most_common]

    return most_common_ngrams


def count_occurrences(doc, n) : 
    """
    Counts occurrences of n-grams in the given document

    Parameters
    ----------
    doc : string
        The document
    n : int
        The n in n-grams

    Returns
    -------
    A counter of how many times an ngram occur in the doc
    """

    # Get n-grams
    ngrams = list()
    for i in range(len(doc)-n) :
        ngrams.append(doc[i:i+n])

    # Count occurrences
    cnt = Counter(ngrams)

    return cnt
    

def ssk_ap(doc1, doc2, base, l, n, C) :
    """
    An approximation of the SSK kernel

    Parameters
    ----------
    doc1 : string
        The first document
    doc2 : string
        The second document
    base : list(string)
        A list of n-grams that is the orthogonal base for the kernel
    l : int
        The decay value (lambda)
    n : int
        The n in SSK (should be the same length as the n-grams)
    C : double
        Normalizing constant (?)

    Returns
    -------
    A kernel entry for doc1, doc2
    """

    kernel = 0

    ssk = SSK(n, l, 10, doc1, doc2)

    for s in base :
        kernel += ssk.k(doc1, s, n) * ssk.k(doc2, s, n)

    return kernel

