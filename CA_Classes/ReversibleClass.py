import numpy as np
import random
from CA_Classes.CAClass import CA

class SecondOrderAutomaton(CA):
    """
    SecondORderAutomaton class, takes any other Automaton an turns it into Second Order One
    """

    def __init__(self,CA,useh=False):
        super(SecondOrderAutomaton, self).__init__(useh=useh)
        self.CA=CA
        self.size = self.CA.size
        self.time = self.CA.time
        self.num_states = self.CA.num_states**2


    def update(self,arrs):
        res = np.zeros_like(arrs)
        arrs = arrs.astype(np.bool)
        xor_vals = self.CA.update(arrs[0]).astype(np.bool)
        res_0 = np.bitwise_xor(xor_vals,arrs[1]).astype(np.int)
        res[1]=arrs[0]
        res[0]=res_0
        return res


    def num_to_arr(self, num):
        a,b = divmod(num, self.CA.num_states)
        a_arr = self.CA.num_to_arr(a)
        b_arr = self.CA.num_to_arr(a)
        return np.array([a,b])

    def arr_to_num(self,arr):
        a = self.CA.arr_to_num(arr[0])
        b = self.CA.arr_to_num(arr[1])
        return a*self.CA.num_states+b

    def arr_to_nums(self,arr):
        a = self.CA.arr_to_nums(arr[0])
        b = self.CA.arr_to_nums(arr[1])
        return a*self.CA.num_states+b

    def nums_to_arr(self,nums):
        a,b = np.divmod(nums, self.CA.num_states)
        #print(a,b)
        a_arr = self.CA.nums_to_arr(a)
        #print(a_arr)
        b_arr = self.CA.nums_to_arr(b)
        #print(b_arr)
        return np.array([a_arr,b_arr])

    def get_random_arr(self, num_reps):
        return np.array([self.CA.get_random_arr(num_reps) for i in range(2)])



    def get_substituted_arr(self,arr,num,vals):
        avals,bvals = np.divmod(vals,self.CA.num_states)
        arr[0]=self.CA.get_substituted_arr(arr[0],num,avals)
        arr[1]=self.CA.get_substituted_arr(arr[1],num,bvals)
        return arr


    def get_plot_mat(self,height,width,start=None):
        if type(start)!=np.array:
            start = self.get_random_arr(int(width/self.size))
        arr = [start[1],start[0]]
        cur=start
        for i in range(height-1):
            cur = self.update(cur)
            arr.append(cur[0])
        return arr
