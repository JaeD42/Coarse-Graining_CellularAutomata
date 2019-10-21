import random
class DisjointSet:
    """
    Disjoint Set Class
    """

    def __init__(self):
        self.parents = {}
        self.rank = {}
        self.size = {}
        self.same = {}
        self.bigger = set()
        self.max_size=0

    def get_rand_repr(self,num):
        reprs = list(self.same[self.find(num)])
        return random.choice(reprs)

    def find(self,x):
        if not x in self.parents:
            self.parents[x]=x
            self.rank[x]=0
            #self.size[x]=1
            self.same[x]=set([x])
        if self.parents[x]!=x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def get_set(self,x):
        return self.same[self.find(x)]

    def get_random_set(self):
        return self.same[random.sample(self.bigger,1)[0]]

    def check_same(self,x,y):
        xRoot = self.find(x)
        yRoot = self.find(y)

        return xRoot==yRoot

    def union(self,x,y):
        xRoot = self.find(x)
        yRoot = self.find(y)

        if xRoot==yRoot:
            return False

        if self.rank[xRoot]<self.rank[yRoot]:
            xRoot,yRoot = yRoot,xRoot

        self.parents[yRoot]=xRoot
        self.same[xRoot].update(self.same[yRoot])
        self.bigger.add(xRoot)
        self.bigger.discard(yRoot)
        del self.same[yRoot]


        if self.rank[xRoot]==self.rank[yRoot]:
            self.rank[xRoot]+=1
        self.max_size = max(len(self.same[xRoot]),self.max_size)
        return True

    def check(self,nums):
        all = set(range(nums))
        c = 0
        sizes = []
        while all:
            curr = all.pop()
            re = self.get_set(curr)
            all = all-re
            c+=1
            sizes.append(len(re))
        sizes = [i for i in sizes if i>1]
        return c,sorted(sizes)


    def __str__(self,nums=16):
        s=""
        for val in self.bigger:
            s+=str(self.same[val])
            s+="\n"
        return s

    def __len__(self):
        pass

    def get_num_classes(self,max_num):
        classes = set()

        for key in self.parents.keys():
            max_num-=1
            if self.find(key) not in classes:
                classes.add(self.find(key))
        return max_num+len(classes)

    def __getitem__(self,val):
        return self.find(val)
