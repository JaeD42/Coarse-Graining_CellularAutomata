import numpy as np
import random
from CA_Classes.CAClass import CA

class CA1D(CA):
    """
    Base Class for 1D CA
    """

    def __init__(self,size,time=-1,useh=False):
        super(CA1D, self).__init__(useh=useh)

        self.size=size
        self.time=time if time!=-1 else size
        self.num_states = 2**self.size


    def update(self,arr):
        return arr

    def arr_to_num(self, arr):
        num = 0
        for val in arr:
            num*=2
            num+=val
        return num

    def num_to_arr(self, num):
        arr = np.zeros((self.size,))
        for i in range(self.size):
            arr[-1-i]=num%2
            num=int(num/2)
        return arr

    def arr_to_nums(self,arr):
        res = arr[::self.size]
        for i in range(1,self.size):
            res*=2
            res+=arr[i::self.size]
        return res

    def arr_to_numsb(self,arr):
        return np.array([self.arr2num(arr[x:x+self.size]) for x in range(0,arr.shape[0],self.size)])

    def nums_to_arr(self,nums):
        res = np.zeros(shape=(nums.shape[0]*self.size,))
        for x in range(0,nums.shape[0]):
                res[x*self.size:x*self.size+self.size]=self.num2arr(nums[x])
        return res

    def get_random_arr(self,num_reps):
        return np.random.randint(0,2,size=(num_reps*self.size,))

    def get_substituted_arr(self,arr,num,vals):
        for i in range(num):
            arr[(2+i*5)*self.size:(3+i*5)*self.size]=self.num2arr(vals[i])
        return arr

    def get_plot_mat(self,height,width,start=None):
        if start is None:
            start = self.get_random_arr(int(width/self.size))
        arr = [start]
        cur=start
        for i in range(height-1):
            cur = self.update(cur)
            arr.append(cur)

        return arr



class ElementaryAutomaton(CA1D):
    """
    Class for ECA,
    """

    def __init__(self,rule,size,time=-1,useh=False):
        super(ElementaryAutomaton, self).__init__(size,time=time,useh=useh)
        self.rule=rule

    def get_equiv_rules(self):
        """
        calculates rule numbers of equivalent rules
        """

        bin_rep = []
        rule=self.rule
        def flip(arr):
            arr[4],arr[1]=arr[1],arr[4] ##100,001=001,100
            arr[6],arr[3]=arr[3],arr[6] ##110,011
            return arr

        def comp(arr):
            return [1-arr[len(arr)-1-i] for i in range(len(arr))]

        def arr_to_num(arr):
            num=0
            for val in arr[::-1]:
                num*=2
                num+=val
            return num


        def num_to_arr(rule):
            bin_rep = []
            rule=int(rule)
            ##### 000           001
            #####  bin_rep[0]   bin_rep[1]
            for i in range(8):
                bin_rep.append(rule%2)
                rule=int(rule/2)
            return bin_rep



        b = num_to_arr(rule)
        b2 = flip(b[:])
        b3 = comp(b[:])
        b4 = flip(b3[:])

        return [arr_to_num(i) for i in (b,b2,b3,b4)]




    def update(self,arr):
        """
        Update function, quickest variant we were able to program
        Use bitwise ands to calc rule
        """
        #self.update_calls+=1
        mid_res = 2*arr[:]
        mid_res[1:]+=1*arr[:-1]
        mid_res[0]+=arr[-1]
        mid_res[:-1]+=4*arr[1:]
        mid_res[-1]+=4*arr[0]
        mid_res = np.array(np.exp2(mid_res),dtype=np.int16)
        res = (mid_res&self.rule)!=0
        return res.astype(int)
