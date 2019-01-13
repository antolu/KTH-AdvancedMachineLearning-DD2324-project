import dataset_preprocessing
import data_io
from SSK_Kernel import SSK
import kernels
import constants
import random
import numpy as np

def preprocessData():
    raw = dataset_preprocessing.load_raw_data()
    processed = dataset_preprocessing.preprocess(raw)
    return processed


if __name__ == "__main__":

    """ If you haven't loaded the data uncomment the two lines below and comment everything below out."""
    #data = preprocessData()
    #data_io.save_data(data)

    #print("main")
    all_docs = []
    classes = []

    data = data_io.load_data("test", "corn")
    all_docs += data
    classes += ["corn"]*len(data)

    data = data_io.load_data("test", "earn")
    all_docs += data
    classes += ["earn"]*len(data)

    data = data_io.load_data("test", "crude")
    all_docs += data
    classes += ["crude"]*len(data)

    data = data_io.load_data("test", "acq")
    all_docs += data
    classes += ["acq"]*len(data)

    c = list(zip(all_docs,classes))

    random.shuffle(c)

    all_docs, classes = zip(*c)

    np.save('classes.npy', classes)
    
    kernels.parallel_matrix_compute(all_docs)