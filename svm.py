from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
#import svmlight
import sklearn.metrics as skmet
import numpy as np
from constants import CATEGORIES


class TextClassifier:

    def __init__(self, train_m, test_m, y_train, y_test):
        self.train_matrix = train_m
        self.test_matrix = test_m
        self.train_labels = y_train
        self.test_labels = y_test


    def transform_labels(self):
        '''
        Not currently used
        '''
        le = preprocessing.LabelEncoder()
        le.fit(test_labels)
        le.transform(train_labels)
        le.transform(test_labels)


    def SVM(self, ker_type):
        if ((ker_type == "ssk") or (ker_type == "ngk")):
            clf = SVC(kernel = "precomputed")
        elif (ker_type == "wk"):
            clf = SVC(kernel="precomputed")
        else:
            print("No valid kernel defined.")


        clf.fit(self.train_matrix, self.train_labels)
        y_pred = clf.predict(self.test_matrix)
        accuracy = clf.score(self.test_matrix, self.test_labels)

        precision,recall,f1score,supp = skmet.precision_recall_fscore_support(y_true=self.test_labels, y_pred=y_pred, labels=CATEGORIES)

        return precision,recall,f1score


    def ANN(self):
        clf = MLPClassifier(solver="lbfgs")

        clf.fit(self.train_matrix, self.train_labels)
        y_pred = clf.predict(self.test_matrix)

        precision, recall, f1score, supp = skmet.precision_recall_fscore_support(y_true=self.test_labels, y_pred=y_pred,
                                                                                 labels=CATEGORIES)

        return precision, recall, f1score


'''
Loads relevant test/train-kernel matrices and their respective class labels
'''

test = np.load("results/sskap_testmatrix.npy")
train = np.load("results/sskap_trainmatrix.npy")
test_labels_byte = np.load("results/sskap_testclasses.npy")
train_labels_byte = np.load("results/sskap_trainclasses.npy")
train_labels = []
test_labels = []

for item in train_labels_byte:
    train_labels.append(item.decode('UTF-8'))

for item in test_labels_byte:
    test_labels.append(item.decode('UTF-8'))



classifier = TextClassifier(train, test, train_labels, test_labels)

prec,rec,f1 = classifier.SVM("ssk")
#prec,rec,f1 = classifier.ANN()
print("Category: ", CATEGORIES)
print("F1: ",f1)
print("Prec: ",prec)
print("Recall: ",rec)
