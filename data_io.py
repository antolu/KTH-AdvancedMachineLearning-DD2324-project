import pickle

data_path = "data/"

EARN = "earn"
ACQ = "acq"
CRUDE = "crude"
CORN = "corn"
TRAIN = "train"
TEST = "test"
CATEGORIES = [EARN, ACQ, CRUDE, CORN]
DATA_SPLIT = [TRAIN, TEST]

def save_data(docs) :
    """
    Saves documents from dictionary form to pickle format.

    Saves data in format "train_acq.pickle"

    Parameters
    ----------
    docs : dict
        Dictionary of documents, accessed like docs["train"]["corn"][i]
    """

    for set_ in docs.keys() :
        for cat in docs[set_].keys() :
            data = docs[set_][cat]
            with open(data_path + set_ + "_" + cat + ".pickle", "wb") as handle :
                pickle.dump(data, handle)

def load_data(set_, category) :
    """
    Loads dataset from pickle files into memory.

    Parameters
    ----------
    set_ : string
        "train" or "test"
    category : string
        "earn", "acq", "crude" or "corn"

    Returns
    ------
    A list of strings (the docs) of the given set and category
    """

    if set_ not in DATA_SPLIT :
        raise Exception(set_ + " is not a valid set.")
    if category not in CATEGORIES :
        raise Exception(category + " is not a valid category.")

    with open(data_path + set_ + "_" + category + ".pickle", "rb") as handle :
        docs = pickle.load(handle, encoding="utf8")

    return docs