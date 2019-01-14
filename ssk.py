""" Reimplementation of the SSK kernel described in Text Classification using String Kernels by Lodhi et al."""
import numpy as np
class SSK():
    """ SSK class to handle all properites of the SSK kernel"""

    def __init__(self, n, l, features, s, t):
        self.n = n
        self.l = l
        self.features = features
        self.s = s
        self.t = t
    


    
    def kb(self, s, t, n):
        # print(s + " " + t)
        x = s[-1]
        #del s[-1]
        u_len = 0
        if n == 0:
            return 1
        if n > min(len(s), len(t)):
            return 0
        
        if x == t[-1]:
            #del t[-1]
            kb_res = self.l * (self.kb(s, t[:-1], n) + self.l * self.kp(s[:-1],t[:-1],n-1) )
            return kb_res
        
        
        else:
            for letter_index in range(len(t)-1, -1, -1):
                if t[letter_index] == x:
                    u_len = letter_index
                    print(u_len)
                    # print(s + t)
                    break
            #del t[u_len:]
            kb_res = np.power(self.l,u_len+1)*self.kb(s, t[:u_len-1],n)
            return kb_res

                


    def kp(self, s, t, n):
        if n == 0:
            return 1
        if n > min(len(s), len(t)):
            return 0
        else:
            x = s[-1]
            #del s[-1]
            kp_res = self.l*self.kp(s[:-1],t,n)
            kb_res = self.kb(s, t, n)
            return kp_res+kb_res
    
    #implements k(s,x) as per definition 2
    def k(self, s,t, n):
        
        if n > min(len(s), len(t)):
            return 0
        else:
            kp_sum = 0
            x = s[-1]
            #del s[-1]
            k_res = self.k(s[:-1],t,n)
            for letter_index in range(len(t)):
                if t[letter_index] == x:
                    #def 2 says t[1:j-1] where j = t[j] = x
                    kp_sum += self.kp(s[:-1],t[:letter_index], n-1)*np.power(self.l,2)
            return k_res + kp_sum
                    


if __name__ == "__main__":
    # s = "cat"
    # t = "cat"
    s = "science is organized knowledge";
    t = "wisdom is organized life";
    
    ssk = SSK(3, 0.2, 10, s, t)
    print(ssk.k(s,t,2))