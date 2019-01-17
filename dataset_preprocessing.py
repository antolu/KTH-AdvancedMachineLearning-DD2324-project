import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize
import re
from constants import *

stop_words = set(stopwords.words("english"))
stop_words.update([".", ",", ">"])
reg = re.compile(r"&lt;")
regg = re.compile(r"[\'\`\"/\,\-\(\)\.]")

def trim(s) :
    """
    Removes stopwords and punctuation from a string

    Parameters
    ----------
    s : string
        The string to remove stop words from

    Returns
    -------
    A trimmed string
    """

    # Remove punctuation and html entities
    s = reg.sub("", s)
    s = regg.sub("", s)
    s = s.lower()

    # Remove title
    s = s.split("\n", 1)[1]

    tokens = word_tokenize(s)

    filtered_sentence = [w for w in tokens if not w in stop_words]

    ret_str = " ".join(filtered_sentence)

    return ret_str

def preprocess(docs, normalize=False) :
    """
    Preprocesses a dataset of documents

    Parameters
    ----------
    docs : dict
        Dictionary of documents, accessed like docs["train"]["corn"][i]

    Returns
    -------
    A dictionary with trimmed documents
    """

    trimmed = {}

    for set_ in DATA_SPLIT :
        trimmed[set_] = {}

        for cat in CATEGORIES :
            l = list()

            untrimmed = docs[set_][cat]

            for d in untrimmed :
                s = trim(d)
                if normalize :
                    if len(s) < NORM_DOC_LENGTH :
                        raise Exception("Document is too short: " + set_ + " " + cat + ". Length: " + str(len(s)))
                    s = s[:NORM_DOC_LENGTH]
                l.append(s)

            trimmed[set_][cat] = l

    return trimmed

def load_raw_data() : 
    """
    Read reuters dataset into memory

    Returns
    -------
    A dictionary of all documents (in raw form)
    Indexed by return_var{"train":{"earn":list(docs)}}
    """
    datasets = {}

    for set_ in DATA_SPLIT :
        datasets[set_] = {}

        for cat in CATEGORIES :
            l = list()
            docs = [reuters.raw(ID) for ID in reuters.fileids(cat) if set_ in ID]

            for i in range(len(docs)) :
                if len(docs[i]) < MIN_DOC_LENGTH :
                    continue
                l.append(docs[i])
                if len(l) >= SPLIT_SIZES[set_][cat] :
                    break

            datasets[set_][cat] = l

    return datasets