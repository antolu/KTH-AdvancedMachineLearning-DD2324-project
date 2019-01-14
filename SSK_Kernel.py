""" Reimplementation of the SSK kernel described in Text Classification using String Kernels by Lodhi et al."""
import numpy as np
import time
import sys
sys.setrecursionlimit(5000)
class SSK():
    """ SSK class to handle all properites of the SSK kernel"""

    def __init__(self, n, l, features, s, t):
        self.n = n
        self.l = l
        self.features = features
        self.s = s
        self.t = t
    


    
    def kb(self, s, t, n):
        try:
            x = s[-1]
            #del s[-1]
            u_len = -2
            if n == 0:
                #print("\n\n\n\n ------------------------\n n = 0 \n ------------------ \n\n\n\n ")
                return 1
            if n > min(len(s), len(t)):
                #print("\n\n\n\n ------------------------\n n bigger than s and t \n ------------------ \n\n\n\n ")
                return 0
            
            if x == t[-1]:
                #del t[-1]
                kb_res = self.l * (self.kb(s, t[:-1], n) + self.l * self.kp(s[:-1],t[:-1],n-1) )
                return kb_res
            
            
            else:
                #print("\n\n\n\n ------------------------\n inside first else \n ------------------ \n\n\n\n ")
                for letter_index in range(len(t)-1, -1, -2):
                    if t[letter_index] == x:
                        u_len = letter_index
                        break

                if u_len == -2:
                    #print("\n\n\n\n ------------------------\n u_len = -1 \n ------------------ \n\n\n\n ")
                    return 0
                """if u_len == 0:
                    t_tmp = []
                else:
                    t_tmp = t[:u_len+1]2"""
                lenU = len(t[u_len+1:])
                #print('\n\n\n\n ------------------------\n u_len = ‰s \n ------------------ \n\n\n\n '% u_len+1)
                kb_res = np.power(self.l,lenU)*self.kb(s, t[:u_len+1],n)
                return kb_res
        except RecursionError as re:
            print('Sorry but this maze solver was not able to finish '
                'analyzing the maze: {}'.format(re.args[0]))
            print("s is = " +s)
            print("t is = " +t)

                


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
                    


#if __name__ == "__main__":
def go(s, t, n): 
    #s = "i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer"
    #t = "i stockholm hej jag heter gustav kjellberg jag kommer fran goteborg men bor i stockholm hej jag heter gustav kjellberg jag kommer"
    
    #ssk = SSK(3, 0.2, 10, s, t)
    time1 = time.time()
    #print(ssk.k(s,t,n))
    time2 = time.time()
    print ('%s function took %0.3f ms' % ("Compiling", (time2-time1)*1000.0))