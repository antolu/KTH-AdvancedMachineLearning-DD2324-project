import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from data_io import load_data


class WK:

    def __init__(self, all_documents):
        """
        Parameters
        ----------
        all_documents : list
            List of all documents
        """
        
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False)
        df = self.vectorizer.fit_transform(all_documents).toarray()

        # Normalize to 1 if word is in document
        df[df > 0] = 1

        self.df = np.sum(df, axis=0)
        self.n = len(all_documents)

    def kernel_function(self, doc1, doc2, unusedvariable):

        tf1 = self.vectorizer.transform([doc1]).toarray().flatten()
        tf2 = self.vectorizer.transform([doc2]).toarray().flatten()

        v1 = np.log(1 + tf1) * np.log(self.n / self.df)
        v2 = np.log(1 + tf2) * np.log(self.n / self.df)

        return np.dot(v1, v2)
