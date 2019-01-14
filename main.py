import dataset_preprocessing
import data_io
from SSK_Kernel import SSK
import kernels
import constants
import numpy as np
from copy import deepcopy


def preprocessData():
    raw = dataset_preprocessing.load_raw_data()
    processed = dataset_preprocessing.preprocess(raw)
    return processed


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

    """all_docs_train = []
    data = deepcopy(data_io.load_data("train", "acq")[:5])
    all_docs_train += data

    data = deepcopy(data_io.load_data("train", "earn")[:5])
    all_docs_train += data
    
    data = deepcopy(data_io.load_data("train", "crude")[:5])
    all_docs_train += data
    
    data = deepcopy(data_io.load_data("train", "corn")[:5])
    all_docs_train += data"""

    mergedDocs = []
    mergedDocs.append(["science is organized knowledge"])
    mergedDocs.append(["wisdom is organized life"])
    #mergedDocs.append(all_docs_train)
    #mergedDocs.append(all_docs_train)
    np.save('classes.npy', classes)
    kernels.parallel_matrix_compute(mergedDocs)