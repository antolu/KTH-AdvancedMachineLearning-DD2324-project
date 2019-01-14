import numpy as np
from kernels import *
from dataset_preprocessing import *
from data_io import *
from constants import *
from ssk_ap import *
import time

untrimmed_docs = load_raw_data()

trimmed_docs = preprocess(untrimmed_docs, normalize=False)

t = trimmed_docs["train"]["acq"][1]

# save_data(trimmed_docs)

# do = load_data("train", "acq")

# print(do[1])

s = trimmed_docs[TRAIN][ACQ][0]
t = trimmed_docs[TRAIN][ACQ][1]

base = get_features([s,t], 100, 3)

# start = time.time()
# kern1 = ssk_ap(s,t,base,0.2,3)
# end = time.time()
# print("val: " + str(kern1) + ". Time: " + str(end-start))

# start = time.time()
# norm11 = ssk_ap(s,s,base,0.2,3)
# end = time.time()
# print("val: " + str(norm11) + ". Time: " + str(end-start))

# start = time.time()
# norm12 = ssk_ap(t,t,base,0.2,3)
# end = time.time()
# print("val: " + str(norm12) + ". Time: " + str(end-start))

# print("Normalized: " + str(kern1/m.sqrt(norm11*norm12)))

# ssk = SSK(3, 0.2, 10, s, t)
# start = time.time()
# kern2 = ssk.k(s,t,3)
# end = time.time()
# print("val: " + str(kern2) + ". Time: " + str(end-start))

# ssk = SSK(3, 0.2, 10, s, t)
# start = time.time()
# norm21 = ssk.k(s,s,3)
# end = time.time()
# print("val: " + str(norm21) + ". Time: " + str(end-start))

# ssk = SSK(3, 0.2, 10, s, t)
# start = time.time()
# norm22 = ssk.k(t,t,3)
# end = time.time()
# print("val: " + str(norm22) + ". Time: " + str(end-start))

# print(" Normalized:" + str(kern2/m.sqrt(norm21*norm22)))


# gram_matrices = {}

# for cat in CATEGORIES :
#     l = trimmed_docs["train"][cat]

#     gram_matrices[cat] = compute_matrix(l, kernel="ngk", n=3)
