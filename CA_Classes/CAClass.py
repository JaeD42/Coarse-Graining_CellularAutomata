import numpy as np
import random
class CA:
    """
    Base Class of CA, mostly simply defines functions that need to exist
    """

    def __init__(self, useh=False):

        self.num_states=0
        self.update_calls=0
        self.dnum = {}
        self.darr = {}

        self.arr2num=self.arr_to_num
        self.num2arr=self.num_to_arr

        if useh:
            self.arr2num=self.arr_to_numh
            self.num2arr=self.num_to_arrh


    def get_random_arr(self,num_reps):
        pass

    def get_substituted_arr(self,arr,num,vals):
        pass

    def update(self,arr):
        return arr

    def run_arr(self,arr):
        for t in range(self.time):
            arr=self.update(arr)
        return self.arr_to_nums(arr)

    def arr_to_num(self,arr):
        pass

    def arr_to_nums(self,arr):
        pass

    def nums_to_arr(self,nums):
        pass

    def num_to_arr(self,num):
        pass

    def num_to_arrh(self,num):
        """
        we often convert supercells back to normal arrays.
        This saves such a calculation and reuses computation.
        Was tested to be significantly faster if all superstates are used
        multiple times. Can lead to slower performance in, for example, Fast_DFS
        Can be activeated/deactivated with useh=True/False
        """
        if num in self.dnum:
            return np.copy(self.dnum[num])
        else:
            arr = self.num_to_arr(num)
            self.dnum[num]=arr
            return np.copy(arr)

    def arr_to_numh(self,arr):
        """
        Same as num_to_arrh but other direction
        """
        s = arr.tostring()
        if s not in self.darr:
            self.darr[s] = self.arr_to_num(arr)
        return self.darr[s]

    def check_correct(self):
        for i in range(self.num_states):
            assert(i==self.arr2num(self.num2arr(i)))


    def check_diff_equiv_set(self,equiv_set,num):
        p1 = random.choices(equiv_set,k=num)
        p2 = random.choices(equiv_set,k=num)
        pairs = [(p1[i],p2[i]) for i in range(num)]

        return self.check_diff(pairs,num)

    def check_diff_pair_list(self,pair_list,num):
        pairs = random.choices(pair_list,k=num)
        return self.check_diff(pairs,num)


    def check_diff(self,pairs,num):
        """
        Calculates what differences can arise if pairs are substituted
        """
        arr = self.get_random_arr(5*num)
        arrC = np.copy(arr)

        arr = self.get_substituted_arr(arr, num, [pairs[i][0] for i in range(num)])
        arrC = self.get_substituted_arr(arrC, num, [pairs[i][1] for i in range(num)])

        narr = self.run_arr(arr)
        narr2 = self.run_arr(arrC)

        bools = narr != narr2
        return [(i,j) if i<j else (j,i) for (i,j) in list(zip(list(narr[bools]),list(narr2[bools]))) ]
