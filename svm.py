from sklearn.svm import SVC,LinearSVC
from sklearn import preprocessing
#import svmlight
import sklearn.metrics as skmet
import numpy as np



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

        #self.transform_labels()
        print(self.test_matrix.shape)
        clf.fit(self.train_matrix, self.train_labels)
        y_pred = clf.predict(self.test_matrix)

        print(y_pred)

        precision,recall,f1,supp = skmet.precision_recall_fscore_support(y_true=self.test_labels, y_pred=y_pred, average=None)

        return f1,precision,recall



'''
Loads relevant test/train-kernel matrices and their respective class labels
'''

# test = np.load("results/ngk_testmatrix.npy")
# train = np.load("results/ngk_trainmatrix.npy")
# test_labels = np.load("results/ngk_testclasses.npy")
# train_labels = np.load("results/ngk_trainclasses.npy")

test = np.load("results/ngktest6.npy")
train = np.load("results/ngktrain6.npy")
test_labels = np.load("results/ngktestlabels6.npy")
train_labels = np.load("results/ngktrainlabels6.npy")

#print(test.shape, train.shape, test_labels.shape, train_labels.shape)

classifier = TextClassifier(train, test, train_labels, test_labels)

f1_out,prec_out,recall_out = classifier.SVM(ker_type="wk")


print(f1_out)
print(prec_out)
print(recall_out)
