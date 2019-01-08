import nltk
from nltk.corpus import stopwords, reuters
from nltk.tokenize import word_tokenize
import re

stop_words = set(stopwords.words("english"))
stop_words.update([".", ",", ">"])
reg = re.compile(r"&lt;")
regg = re.compile(r"'")

EARN = "earn"
ACQ = "acq"
CRUDE = "crude"
CORN = "corn"
TRAIN = "train"
TEST = "test"

CATEGORIES = [EARN, ACQ, CRUDE, CORN]
DATA_SPLIT = [TRAIN, TEST]

SPLIT_SIZES = {
    TRAIN:{
        EARN:153, 
        ACQ:114,
        CRUDE:76,
        CORN:38
    },
    TEST:{
        EARN:40, 
        ACQ:25,
        CRUDE:15,
        CORN:10
    }
}

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

    st = ""

    for tkn in filtered_sentence :
        st += tkn + " "

    return st

def preprocess(docs) :
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
                l.append(trim(d))

            trimmed[set_][cat] = l

    return trimmed

def load_data() : 
    """
    Read reuters dataset into memory
    """
    datasets = {}

    for set_ in DATA_SPLIT :
        datasets[set_] = {}

        for cat in CATEGORIES :
            l = list()
            docs = [reuters.raw(ID) for ID in reuters.fileids(cat) if set_ in ID]

            for i in range(SPLIT_SIZES[set_][cat]) :
                l.append(docs[i])

            datasets[set_][cat] = l

    return datasets