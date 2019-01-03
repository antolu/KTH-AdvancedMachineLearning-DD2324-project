""" Reimplementation of the SSK kernel described in Text Classification using String Kernels by Lodhi et al."""

class SSK():
    """ SSK class to handle all properites of the SSK kernel"""

    def __init__(self, n, l, features, s, t):
        self.n = n
        self.l = l
        self.features = features
        self.s = s
        self.t = t










if __name__ == "__main__":
    s = "hej"
    t = "he"
    
    ssk = SSK(3, 0.3, 10, s, t)