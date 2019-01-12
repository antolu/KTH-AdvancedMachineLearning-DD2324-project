import dataset_preprocessing
import data_io
from SSK_Kernel import SSK
import kernels
import constants


def preprocessData():
    raw = dataset_preprocessing.load_raw_data()
    processed = dataset_preprocessing.preprocess(raw)
    return processed


if __name__ == "__main__":

    """ If you haven't loaded the data uncomment the two lines below and comment everything below out."""
    #data = preprocessData()
    #data_io.save_data(data)

    #print("main")
    data = data_io.load_data("train", "corn")

    kernels.parallel_matrix_compute(data)
