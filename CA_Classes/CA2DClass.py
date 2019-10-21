import numpy as np
import random
from CA_Classes.CAClass import CA
#from keras.models import Model
#from keras.layers import *
#from keras.optimizers import *

class CA2D(CA):
    """
    Base Class for 2D CA
    """

    def __init__(self,size,time=-1,useh=False,torus=True):
        super(CA2D, self).__init__(useh=useh)
        self.torus=torus
        self.size=size
        self.time=time if time!=-1 else size
        self.num_states=2**(self.size*self.size)
        self.n=3


    def wrap_around(self,arr):
        """
        creates a bigger arr in order to simluate torus
        """
        shape = arr.shape
        arr = arr.copy()
        new_arr = np.zeros((shape[0]+2,shape[1]+2))
        new_arr[1:-1,1:-1]=arr
        new_arr[:,0]=new_arr[:,-2]
        new_arr[:,-1]=new_arr[:,1]
        new_arr[0,:]=new_arr[-2,:]
        new_arr[-1,:]=new_arr[1,:]

        return new_arr

    def update(self,arr):
        if self.torus:
            return self.update_conway(self.wrap_around(arr))
        return self.update_conway(arr)


    def update_conway(self,arr):
        return arr[1:-1,1:-1]

    def arr_to_num(self,arr):
        c = 0
        size = arr.shape[0]
        for i in range(size):
            for j in range(size):
                c=c*2
                c+=arr[i,j]

        return int(c)

    def num_to_arr(self,num):
        arr = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(self.size):
                arr[-i-1,-j-1]=num%2
                num=num//2
        return arr

    def arr_to_nums(self,arr):
        res = np.zeros_like(arr[::self.size,::self.size])
        for i in range(self.size):
            for j in range(self.size):
                res*=2
                res+=arr[i::self.size,j::self.size]
        return res

    def nums_to_arr(self,nums):
        res = np.zeros(shape=(nums.shape[0]*self.size,nums.shape[1]*self.size))
        for x in range(0,nums.shape[0]):
            for y in range(0,nums.shape[1]):
                res[x*self.size:x*self.size+self.size,y*self.size:y*self.size+self.size]=self.num2arr(nums[x,y])
        return res

    def get_random_arr(self,num_reps):

        return np.random.randint(0,2,size=(self.n*self.size*num_reps,self.n*self.size*num_reps))

    def get_substituted_arr(self,arr,num,vals):

        for i in range(num):
            for j in range(num):
                arr[(int(self.n/2)+i*self.n)*self.size:(int(self.n/2)+1+i*self.n)*self.size,(int(self.n/2)+j*self.n)*self.size:(int(self.n/2)+1+j*self.n)*self.size]=self.num2arr(vals[i])
        return arr


    def get_anim_arr(self,time=100,size=100,start_arr=None):
        arr=start_arr
        if start_arr is None:
            arr = np.random.randint(0,2,size=(size,size))

        arrs = [arr]
        for i in range(time):
            arrs.append(self.update(arrs[-1]))
        return arrs





"""
@deprecated
class DeepCA2D(CA2D):
    #Creates a CA based on a DL network, not used

    def __init__(self,model,size,time=-1,num_cols=2,useh=True):
        super(DeepCA2D, self).__init__(size,time=time,useh=useh)



        self.model=model
        self.num_cols=num_cols

    def change_model(self,model):
        self.model=model

    def update_conway(self,arr):
        next_arr = self.model.predict(np.array([[arr]]))
        return (self.num_cols*next_arr).astype(int)[0,0]

    @staticmethod
    def get_random_CA(n=3,num_filters=16):


        inputNow = Input(shape=(1,None,None,))
        x = Conv2D(num_filters,(n,n),activation='relu',data_format="channels_first",padding="same",input_shape=(1,None,None))(inputNow)
        y = Conv2D(1,(1,1),activation='sigmoid',data_format="channels_first")(x)
        model = Model(inputs=inputNow,outputs=y)
        model.compile(loss='mse',optimizer='adam')

        return DeepCA2D(model,n)

    def show(self):
        arr = np.random.randint(2,size=(200,200))
        arrs = [arr]

        for i in range(100):
            arr = self.update_conway(arr)
            arrs.append(arr)
            #print(arr[:10,:10])

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure()

        ims = []
        for i in range(100):

            im = plt.imshow(arrs[i], animated=True, interpolation='nearest')
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=1000)

        return ani

    def get_langton(self):
        arr = self.nums_to_arr(np.array([list(range(2**(self.size**2)))]))
        arr = self.update_conway(arr)
        vals = arr[1,1::3]
        return (sum(vals)/len(vals))
#        print(arr.shape)

    def get_code(self):
        arr = self.nums_to_arr(np.array([list(range(2**(self.size**2)))]))
        arr = self.update_conway(arr)
        vals = arr[1,1::3]
        return (sum(vals)/len(vals))
#        print(arr.shape)
"""

class TotalCA2D(CA2D):
    """
    Outer Totalistice 2D CA, can be instantiated by
    by_bs_string, using "B/S" notation
    by_num, converting B/S notation into an integer
    """

    def __init__(self,sbrule,size,time=-1,useh=True):

        super(TotalCA2D, self).__init__(size,time=time,useh=useh)
        self.rule=sbrule
        self.s = sbrule[0]
        self.b = sbrule[1]
        self.d8_dict={}

    def update_conway(self,arr):
        res = np.zeros_like(arr)
        res[1:-1,1:-1] = arr[0:-2,0:-2]+arr[0:-2,1:-1]+arr[0:-2,2:]+ \
                arr[1:-1,0:-2]+arr[1:-1,2:]+ \
                arr[2:,0:-2]+arr[2:,1:-1]+arr[2:,2:]
        next_arr = np.zeros_like(arr)

        for val in self.s:
            next_arr[res==val]=arr[res==val]
        for val in self.b:
            next_arr[(res==val) * (arr==0)]=1

        return next_arr[1:-1,1:-1]


    def rot_arr(self,arr):
        return arr.transpose()[::-1]

    def mir_arr(self,arr):
        return arr[::-1]

    def get_d8(self,num):
        """
        Get nums that are D8 equivalent
        """
        if num not in self.d8_dict:

            arr = self.num2arr(num)
            nums = [num]
            for i in range(3):
                arr = self.rot_arr(arr)
                nums.append(self.arr2num(arr))

            arr = self.rot_arr(arr)
            arr = self.mir_arr(arr)

            nums.append(self.arr2num(arr))
            for i in range(3):
                arr = self.rot_arr(arr)
                nums.append(self.arr2num(arr))
            self.d8_dict[num]=nums
        return self.d8_dict[num]

    @staticmethod
    def by_bs_string(s,size=2,time=-1,useh=True):
        arr = s.split("/")
        s=[]
        b=[]
        if arr[0]=="S":
            arr[0],arr[1]=arr[1],arr[0]
        for c in arr[0][1:]:
            b.append(int(c))
        for c in arr[1][1:]:
            s.append(int(c))
        return TotalCA2D([s,b], size=size,time=time,useh=useh)

    @staticmethod
    def by_num(rule_num,size,time=-1,useh=True):
        s = []
        b = []
        for i in range(9):
            if rule_num%2:
                s.append(i)
            rule_num = int(rule_num/2)

        for i in range(9):
            if rule_num%2:
                b.append(i)
            rule_num = int(rule_num/2)

        return TotalCA2D([s,b], size, time=time,useh=useh)

    def get_langton(self):
        """
        get langton lambda value
        """
        from scipy.special import comb
        pos = 0.0
        for num in self.s:
            pos+=comb(8,num)

        for num in self.b:
            pos+=comb(8,num)

        return pos/(2**9)



class TotalCA2D_Neumann(CA2D):
    """
    Similar as above, but using Neumann neighborhood
    """

    def __init__(self,sbrule,size,time=-1,useh=True):

        super(TotalCA2D_Neumann, self).__init__(size,time=time,useh=useh)
        self.rule=sbrule
        self.s = sbrule[0]
        self.b = sbrule[1]
        self.d8_dict={}

    def update_conway(self,arr):
        res = np.zeros_like(arr)
        res[1:-1,1:-1] = arr[0:-2,0:-2]+arr[0:-2,1:-1]+arr[0:-2,2:]+ \
                arr[1:-1,0:-2]+arr[1:-1,2:]+ \
                arr[2:,0:-2]+arr[2:,1:-1]+arr[2:,2:]

        res[1:-1,1:-1] = arr[0:-2,1:-1]+ \
                arr[1:-1,0:-2]+arr[1:-1,2:]+ \
                arr[2:,1:-1]


        next_arr = np.zeros_like(arr)

        for val in self.s:
            next_arr[res==val]=arr[res==val]
        for val in self.b:
            next_arr[(res==val) * (arr==0)]=1

        return next_arr[1:-1,1:-1]


    def rot_arr(self,arr):
        return arr.transpose()[::-1]

    def mir_arr(self,arr):
        return arr[::-1]

    def get_d8(self,num):
        if num not in self.d8_dict:

            arr = self.num2arr(num)
            nums = [num]
            for i in range(3):
                arr = self.rot_arr(arr)
                nums.append(self.arr2num(arr))

            arr = self.rot_arr(arr)
            arr = self.mir_arr(arr)

            nums.append(self.arr2num(arr))
            for i in range(3):
                arr = self.rot_arr(arr)
                nums.append(self.arr2num(arr))
            self.d8_dict[num]=nums
        return self.d8_dict[num]

    @staticmethod
    def by_bs_string(s,size=2,time=-1,useh=True):
        arr = s.split("/")
        s=[]
        b=[]
        if arr[0]=="S":
            arr[0],arr[1]=arr[1],arr[0]
        for c in arr[0][1:]:
            b.append(int(c))
        for c in arr[1][1:]:
            s.append(int(c))
        return TotalCA2D_Neumann([s,b], size=size,time=time,useh=useh)

    @staticmethod
    def by_num(rule_num,size,time=-1,useh=True):
        s = []
        b = []
        for i in range(5):
            if rule_num%2:
                s.append(i)
            rule_num = int(rule_num/2)

        for i in range(5):
            if rule_num%2:
                b.append(i)
            rule_num = int(rule_num/2)

        return TotalCA2D_Neumann([s,b], size, time=time,useh=useh)

    def get_langton(self):
        from scipy.special import comb
        pos = 0.0
        for num in self.s:
            pos+=comb(4,num)

        for num in self.b:
            pos+=comb(4,num)

        return pos/(2**5)


class Wireworld(TotalCA2D):
    """
    Wireworld Automaton
    """

    def __init__(self,size=2,time=-1,useh=True):
        super(Wireworld, self).__init__([[],[]],size,time=time,useh=useh)
        self.d8_dict={}
        self.num_states=4**(size*size)

    def update_conway(self, arr):
        res = np.zeros_like(arr)

        arrHead = (arr==1).astype(int)
        res[1:-1,1:-1] = arrHead[0:-2,0:-2]+arrHead[0:-2,1:-1]+arrHead[0:-2,2:]+ \
                arrHead[1:-1,0:-2]+arrHead[1:-1,2:]+ \
                arrHead[2:,0:-2]+arrHead[2:,1:-1]+arrHead[2:,2:]
        #print(res)
        next_arr = np.zeros_like(arr)

        #conductors
        next_arr[arr!=0]=3

        #head->tail
        next_arr[arr==1]=2

        #tail->conductor
        next_arr[arr==2]=3

        #conductor->head
        next_arr[(arr==3)*(res==1)]=1
        next_arr[(arr==3)*(res==2)]=1

        return next_arr[1:-1,1:-1]


    def arr_to_num(self,arr):
        c = 0
        size = arr.shape[0]
        for i in range(size):
            for j in range(size):
                c=c*4
                c+=arr[i,j]

        return int(c)

    def num_to_arr(self,num):
        arr = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(self.size):
                arr[-i-1,-j-1]=num%4
                num=num//4
        return arr


    def get_random_arr(self,num_reps):
        return np.random.randint(0,4,size=(self.n*self.size*num_reps,self.n*self.size*num_reps))


class Wireworld_Cool(TotalCA2D):
    """
    Combination of wireworld with other automata, by
    growing the wires using any 2D automaton
    Mostly for fun
    """

    def __init__(self,size=2,time=-1,useh=True, other_ca=None):
        super(Wireworld_Cool, self).__init__([[],[]],size,time=time,useh=useh)
        self.d8_dict={}
        self.num_states=4**(size*size)
        self.other_ca = other_ca

    def update_conway(self, arr):
        res = np.zeros_like(arr)
        if self.other_ca is not None:
            new_bg = self.other_ca.update((arr!=0).astype(int))
        else:
            new_bg = (arr!=0).astype(int)
        arrHead = (arr==1).astype(int)
        res[1:-1,1:-1] = arrHead[0:-2,0:-2]+arrHead[0:-2,1:-1]+arrHead[0:-2,2:]+ \
                arrHead[1:-1,0:-2]+arrHead[1:-1,2:]+ \
                arrHead[2:,0:-2]+arrHead[2:,1:-1]+arrHead[2:,2:]
        #print(res)
        next_arr = np.zeros_like(arr)

        #conductors
        next_arr[new_bg!=0]=3

        #head->tail
        next_arr[arr==1]=2

        #tail->conductor
        #next_arr[arr==2]=3

        #conductor->head
        next_arr[(arr==3)*(res==1)]=1
        next_arr[(arr==3)*(res==2)]=1

        return next_arr[1:-1,1:-1]


    def arr_to_num(self,arr):
        c = 0
        size = arr.shape[0]
        for i in range(size):
            for j in range(size):
                c=c*4
                c+=arr[i,j]

        return int(c)

    def num_to_arr(self,num):
        arr = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(self.size):
                arr[-i-1,-j-1]=num%4
                num=num//4
        return arr


    def get_random_arr(self,num_reps):
        return np.random.randint(0,4,size=(self.n*self.size*num_reps,self.n*self.size*num_reps))
