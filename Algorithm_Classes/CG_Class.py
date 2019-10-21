from Algorithm_Classes.DisjointSet import DisjointSet
from graph_tool.all import *
import numpy as np
import random
from collections import deque
from tqdm import tqdm

class CG_Algorithm:

    def __init__(self,CA):
        self.CA=CA
        self.tqdm = lambda x:x

    def run(self):
        pass

    def get_other_d8(self,node):
        i,j = node
        val_is = self.CA.get_d8(i)
        val_js = self.CA.get_d8(j)
        res = []
        for i in range(len(val_is)):
            res.append((min(val_is[i],val_js[i]),max(val_is[i],val_js[i])))
        return res




class Check_AB(CG_Algorithm):
    """Most basic Algorithm, checks if it is possible to CG two states by
    checking all enforced equivalencies and collapsing them. Does this heuristically
    """

    def __init__(self,CA,a,b,num_checks=5,max_runs=100):
        super(Check_AB,self).__init__(CA)
        self.disset=DisjointSet()
        self.disset.union(a, b)
        self.num_checks=num_checks
        self.max_runs = max_runs



    def run(self):
        max_runs = self.max_runs
        while self.disset.max_size<self.CA.num_states and max_runs>0:
            cur = list(self.disset.get_random_set())
            diffs = self.CA.check_diff_equiv_set(cur,self.num_checks)
            for (i,j) in diffs:
                self.disset.union(i,j)
            max_runs-=1
        return self.disset.max_size

class ForbiddenStates(CG_Algorithm):
    """
    Heuristic Implementation of the forbidden states algorithm. Starts with some node
    and checks if it leads to a trivial coarse graining.
    Saves these nodes as forbidden states and uses that information to sace runtime.
    """

    def __init__(self,CA,num_checks=5,max_runs=20,show_progress=False,used8=False):
        super(ForbiddenStates,self).__init__(CA)
        self.num_checks = num_checks
        self.max_runs=max_runs
        self.impossibles = set()
        n=self.CA.num_states
        self.max_nodes = (n*(n-1))/2
        self.used8=used8

        if show_progress:
            self.tqdm=tqdm


    def check_set(self,disset):
        for (x,y) in self.impossibles:
            if disset.check_same(x,y):
                return True

    def run(self):
        l = list(range(self.CA.num_states))
        random.shuffle(l)
        for i in self.tqdm(l):
            #print(len(self.impossibles))
            for j in range(i+1,self.CA.num_states):
                #print(len(self.impossibles))
                if (i,j) not in self.impossibles:
                    if self.single_run(i,j):
                        continue


    def single_run(self,a,b):
        disset = DisjointSet()
        disset.union(a,b)
        max_runs = self.max_runs
        prev_size=0
        check_count=0
        while disset.max_size<self.CA.num_states and max_runs>0:
            #print(disset.max_size,self.CA.num_states)
            cur = list(disset.get_random_set())
            diffs = self.CA.check_diff_equiv_set(cur,self.num_checks)
            for (i,j) in diffs:
                disset.union(i,j)
            if disset.max_size == prev_size:
                max_runs-=1
                continue
            else:
                prev_size = disset.max_size
                max_runs=self.max_runs
            if check_count==0:
                if self.check_set(disset):
                    if self.used8:
                        for val in self.get_other_d8((a,b)):
                            self.impossibles.add(val)
                    self.impossibles.add((a,b))
                    return True
                check_count=10
            else:
                check_count-=1
        if disset.max_size==self.CA.num_states:
            if self.used8:
                for val in self.get_other_d8((a,b)):
                    self.impossibles.add(val)
            self.impossibles.add((a,b))
            return True
        #print(disset.max_size)
        #if disset.max_size>=500:
        #    print(list(sorted(list(set(range(512))-disset.get_random_set()))))
        return False

    def any_cg(self):
        return len(self.impossibles)!=self.max_nodes



class Base_Graph(CG_Algorithm):
    """
    Base Class for the following Algorithms
    Those are based on the Enforcement Graph, and all Graph Algorithms
    Can easily be visualised.
    """

    def __init__(self, CA, num_tests=5):
        super(Base_Graph, self).__init__(CA)
        self.num_tests = num_tests
        self.g = Graph(directed=True)
        self.node_dict={}
        self.rev_dict=None


    def get_node_num(self,node):
        """
        Takes a node and returns the index in the graph
        """
        if node not in self.node_dict:
            self.node_dict[node]=len(self.node_dict.keys())
        return self.node_dict[node]

    def get_neighbors_of_vert(self, pos):
        return self.CA.check_diff_pair_list([pos], self.num_tests)

    def get_neighbors_of_comp(self, comp):
        return self.CA.check_diff_pair_list(comp, self.num_tests)


    def create_rev_dict(self):
        """
        Reverses the node dict
        """
        self.rev_dict={self.node_dict[key]:key for key in self.node_dict.keys()}
        return self.rev_dict


    def arr_to_class(self,arr,disset):
        narr = np.array(arr)
        flat = narr.flatten().astype(int)
        carr = np.array([disset.find(i) for i in flat])
        return np.reshape(carr,narr.shape)

    def comp_to_set(self,comp):
        self.create_rev_dict()
        s=set()
        ds = DisjointSet()
        count=self.CA.num_states

        for c in comp:
            tup = self.rev_dict[c]
            if not ds.check_same(tup[0],tup[1]):
                count-=1
                ds.union(tup[0],tup[1])
        return ds,count

    def comp_arr_to_set(self,comp):
        l=[]
        for i in range(len(comp)):
            if comp[i]:
                l.append(i)
        return self.comp_to_set(l)

    def comp_to_num_classes(self, comp):
        self.create_rev_dict()
        s=set()
        ds = DisjointSet()
        #for i in range(self.CA.num_states):
        #    ds.find(i)
        count=self.CA.num_states
        #print(comp)
        for c in comp:
            tup = self.rev_dict[c]
            if not ds.check_same(tup[0],tup[1]):
                count-=1
                ds.union(tup[0],tup[1])

        return count

    def comp_ind_to_num_classes(self,comp):
        l=[]
        for i in range(len(comp)):
            if comp[i]:
                l.append(i)
        return self.comp_to_num_classes(l)

    def save_graph_plot(self,path,show_name_full=False,show_name_scc=False,only_scc=False):
        """
        saves images of the graph and the scc to the disk
        """

        comps = np.array(list(label_components(self.g)[0]))
        if not only_scc:
            names = self.g.new_vertex_property("string")
            color = self.g.new_vertex_property("int")
            for k in self.node_dict.keys():
                names[self.node_dict[k]]=str(k)
                color[self.node_dict[k]]=comps[self.node_dict[k]]

            pos = sfdp_layout(self.g)
            if show_name_full:
                graph_draw(self.g,pos=pos,vertex_fill_color=color,vertex_text=names,output=path+"full.pdf")
            else:
                graph_draw(self.g,pos=pos,vertex_fill_color=color,output=path+"full.pdf")


        g2,comps,s_comps = self.create_scc_graph()

        names = g2.new_vertex_property("string")
        color = g2.new_vertex_property("int")
        shapes = g2.new_vertex_property("string")
        #print(comps)
        for v in g2.vertices():
            color[v]=s_comps[int(v)]
            names[v]=self.comp_ind_to_num_classes(comps==s_comps[int(v)])
            shapes[v]="circle"
            if v.out_degree()==0:
                shapes[v]="square"


        pos = sfdp_layout(g2)
        if show_name_scc:
            graph_draw(g2,pos=pos,vertex_fill_color=color,vertex_text=names,vertex_shape=shapes,output=path+"scc.pdf")
        else:
            graph_draw(g2,pos=pos,vertex_fill_color=color,vertex_shape=shapes,output=path+"scc.pdf")


    def save_graph(self,path):
        """
        saves the graph to disk in order to load again
        """
        names = self.g.new_vertex_property("string")
        rdict = self.create_rev_dict()
        for i in self.g.vertices():
            names[i]=str(rdict[i])
        self.g.vertex_properties["name"] = names
        self.g.save(path)

    def load_graph(self,path):
        """
        loads graph from disk
        """
        g = load_graph(path)
        names = g.vertex_properties["name"]
        for i in g.vertices():
            self.node_dict[tuple(map(int, names[i][1:-1].split(',')))]=i
        self.g=g

    def display_graph(self,show_names=False,max_iter=1):
        """
        Displays the graph using sfdp layout, set max_iter to 0 in order to let
        sfdp run till convergence
        """
        #remove_parallel_edges(self.g)
        names = self.g.new_vertex_property("string")
        color = self.g.new_vertex_property("int")
        comps = label_components(self.g)[0]
        for k in self.node_dict.keys():
            names[self.node_dict[k]]=str(k)
            color[self.node_dict[k]]=comps[self.node_dict[k]]

        pos = sfdp_layout(self.g,max_iter=max_iter)
        #pos = fruchterman_reingold_layout(self.g)
        #pos = arf_layout(self.g)
        #pos = radial_tree_layout(self.g,0)
        if show_names:
            graph_draw(self.g,pos=pos,vertex_fill_color=color,vertex_text=names)#,output="rule60_big11_nice.pdf")
        else:
            graph_draw(self.g,pos=pos,vertex_fill_color=color)
        return comps

    def display_scc(self,show_names=False,max_iter=1):
        """
        Displays the scc graph using sfdp layout, set max_iter to 0 in order to let
        sfdp run till convergence
        """
        comps = np.array(list(label_components(self.g)[0]))
        g2,comps,s_comps = self.create_scc_graph()

        names = g2.new_vertex_property("string")
        color = g2.new_vertex_property("int")
        #print(comps)
        for k in range(max(comps)+1):
            color[k]=s_comps[k]
            names[k]=self.comp_ind_to_num_classes(comps==s_comps[k])


        pos = sfdp_layout(g2,max_iter=max_iter)

        graph_draw(g2,pos=pos,vertex_fill_color=color,vertex_text=names)


    def create_scc_graph(self):
        """
        creates and returns a condensation if the graph
        """
        comps = label_components(self.g)[0]

        res = condensation_graph(self.g, comps,self_loops=False)
        scc_graph, bb, vcount, ecount, avp, aep = res

        return scc_graph,np.array(list(comps)),np.array(list(bb))

    def any_cg(self):
        sccg,comps,sccomps = self.create_scc_graph()
        for vert in sccg.vertices():
            if vert.out_degree()==0:
                if self.comp_ind_to_num_classes(comps==sccomps[int(vert)])>1:
                    return True
        return False

class Full_AB(Base_Graph):
    """

    """

    def __init__(self,CA,a,b):
        super(Full_AB,self).__init__(CA)
        self.disset = DisjointSet()
        self.start = (min(a,b),max(a,b))


    def get_neighbors(self,a,b):
        new_nodes=set()
        for i in range(self.CA.num_states):
            for j in range(self.CA.num_states):
                arr_a = self.CA.nums_to_arr(np.array([i,a,j]))
                new_nums_a = self.CA.run_arr(arr_a)
                arr_b = self.CA.nums_to_arr(np.array([i,b,j]))
                new_nums_b = self.CA.run_arr(arr_b)

                for i in range(3):
                    if new_nums_a[i]!=new_nums_b[i]:
                        x = new_nums_b[i]
                        y = new_nums_a[i]
                        if new_nums_a[i]<new_nums_b[i]:
                            x,y = new_nums_a[i],new_nums_b[i]

                        new_nodes.add((int(x),int(y)))
        return new_nodes




    def run(self):
        q = deque()
        q.append(self.start)
        while q:
            cur = q.pop()
            #print(cur)
            if not self.disset.check_same(cur[0],cur[1]):
                self.disset.union(cur[0],cur[1])
                new = self.get_neighbors(cur[0],cur[1])
                for n in new:
                    q.append(n)

        return self.disset




#DEPRECATED
class Complete_Undirected_Graph(Base_Graph):
    def __init__(self,CA):
        super(Complete_Undirected_Graph,self).__init__(CA)
        self.g.set_directed(False)
        self.visited = set()

    def run(self):

        cur_set = set()
        nodes = self.CA.num_states*(self.CA.num_states-1)/2
        cur_num = (0,1)
        while len(self.visited)< nodes:
            while cur_num in self.visited:
                if cur_num[1]==self.CA.num_states-1:
                    cur_num = (cur_num[0]+1,cur_num[0]+2)
                else:
                    cur_num = (cur_num[0],cur_num[1]+1)
            if cur_num[0]>=self.CA.num_states-1:
                break

            cur_set.add(cur_num)

            cur_id = self.get_node_num(cur_num)
            cur_length = 0
            merger=False
            while len(cur_set)>cur_length and not merger:
                cur_length=len(cur_set)
                for i in range(10):
                    candidates = self.get_neighbors_of_comp(list(cur_set))
                    for c in candidates:
                        if c in cur_set:
                            continue
                        elif c in self.visited:
                            merger = True
                            self.visited.add(c)
                        self.g.add_edge(cur_id,self.get_node_num(c))
                        cur_set.add(c)
            self.visited.add(cur_num)


class Cheap_Graph_Calc(Base_Graph):
    """
    Calculates an approximation of the complete Graph
    num_test describes how big the used pattern is,
    note that this increases complexity in the square on 2D

    num_runs describes how many iterations are calculates
    """


    def __init__(self,CA,num_runs=10,num_tests=10,show_progress=False,used8=False):
        super(Cheap_Graph_Calc,self).__init__(CA)
        self.update_map = {}
        self.num_states = self.CA.num_states
        stat=self.num_states
        self.num_verts = int(stat/2*(stat-1))
        self.g.add_vertex(self.num_verts)
        self.num_tests=num_tests
        self.num_runs=num_runs
        self.used8=used8
        self.tqdm=lambda x:x
        self.edge_set=set()
        if show_progress:
            self.tqdm=tqdm


    def pair_calc(self,i,j):
        for runs in range(self.num_runs):
            numa = self.get_node_num((i,j))
            diff_list = self.CA.check_diff_pair_list([(i,j)],num=self.num_tests)
            if not self.used8:
                for diff in diff_list:
                    n = self.get_node_num(diff)
                    if n!=numa:
                        self.g.add_edge(numa,n)
            else:
                val_is = self.CA.get_d8(i)
                val_js = self.CA.get_d8(j)
                for diff in diff_list:
                    diff_is = self.CA.get_d8(diff[0])
                    diff_js = self.CA.get_d8(diff[1])

                    for ind in range(8):
                        start =self.get_node_num((min(val_is[ind],val_js[ind]),max(val_is[ind],val_js[ind])))
                        end = self.get_node_num((min(diff_is[ind],diff_js[ind]),max(diff_is[ind],diff_js[ind])))

                        if start!=end and (start,end) not in self.edge_set:
                            self.g.add_edge(start,end)
                            self.edge_set.add((start,end))
                        else:
                            break

    def run(self):
        #print(self.num_verts)
        for i in self.tqdm(range(self.num_states)):
            for j in range(i+1,self.num_states):
                self.pair_calc(i,j)
        remove_parallel_edges(self.g)







class Complete_Graph_Calc(Base_Graph):
    """ Very runtime intensive but maximally correct algorithm. Tries to calculate
    as much as possible of the Enforcement Graph. One can then check SCC on this graph structure
    to find CGs. Note that this is *very* infeasible for even slightly bigger neighborhood sizes
    as the graph size increases sqaured exponential in N size.
    """
    """ Will only work for 1D !!!!!"""

    def __init__(self,CA,show_progress=False):
        super(Complete_Graph_Calc,self).__init__(CA)
        self.update_map = {}
        stat = self.CA.num_states
        self.g.add_vertex(int(stat/2*(stat-1)))

        if show_progress:
            self.tqdm=tqdm


    def calc_all_updates(self):
        states = self.CA.num_states
        for a in self.tqdm(range(states)):
            for b in range(states):
                for c in range(max(a,b),states):
                    arr = self.CA.nums_to_arr(np.array([a,b,c]))
                    new_nums = self.CA.run_arr(arr)
                    #print(new_nums)
                    #new_nums = self.CA.arr_to_nums(arr)
                    self.update_map[(a,b,c)]=new_nums
                    self.update_map[(b,c,a)]=(new_nums[1],new_nums[2],new_nums[0])
                    self.update_map[(c,a,b)]=(new_nums[2],new_nums[0],new_nums[1])


    def add_diffs(self,val1,val2,goal_inds):
        res1 = self.update_map[val1]
        res2 = self.update_map[val2]
        for i in range(3):
            if res1[i]!=res2[i]:
                a = res2[i]
                b = res1[i]
                if res1[i]<res2[i]:
                    a,b = res1[i],res2[i]
                goal_inds.add(self.get_node_num((int(a),int(b))))

    def add_edges_fast(self,a,b):
        #print(a,b)
        start_node = (a,b)
        start_ind = self.get_node_num(start_node)

        goal_inds = set()

        for num1 in range(self.CA.num_states):
            for num2 in range(self.CA.num_states):
                self.add_diffs((a,num1,num2),(b,num1,num2),goal_inds)
                #self.add_diffs((num1,a,num2),(num1,b,num2),goal_inds)
                #self.add_diffs((num1,num2,a),(num1,num2,b),goal_inds)

        #if a==12 and b==13:
        #    print(goal_inds)

        for end in goal_inds:
            if end!=start_ind:
                self.g.add_edge(start_ind,end)

    def create_graph(self):
        for a in self.tqdm(range(self.CA.num_states)):
            for b in range(a+1,self.CA.num_states):
                self.add_edges_fast(a,b)



    def run(self):
        self.calc_all_updates()
        self.create_graph()

class Fast_DFS(Base_Graph):
    """Based on the observation that Enforcement Graph
    Contains all information and any CG needs to be a subset of Graph
    with no outgoing edges. From this observation we can conclude that any
    CG is a union of strongly connected components. If one can find the lowest
    of these one can immediately describe all CGs

    num_tests describes how big a pattern is used
    max_depth gives a cutoff after which finding a cycle happens far more quickly
    max_runs describes how often we restart after a cycle is found


    """

    def __init__(self,CA,num_tests=2,max_runs=10,max_depth=20):
        super(Fast_DFS, self).__init__(CA)
        self.disset = DisjointSet()
        self.visited = set()
        self.max_depth=max_depth
        self.max_runs=max_runs
        self.num_tests=num_tests
        self.path = []

    def run(self,start=None):
        if start==None:
            #TODO is wrong, can collapse, fix!
            start = (1,0)
            while start[0]>=start[1]:
                start = (random.randint(0,self.CA.num_states-1),random.randint(0,self.CA.num_states-1))
        cur = start

        old_cur = cur
        cur_changed = self.max_runs
        while cur_changed:
            old_cur=cur
            cur = self.dfs(cur)
            if self.disset.check_same(cur,old_cur):
                cur_changed-=1
            else:
                cur_changed=self.max_runs

    def merge(self,path):
        for i in range(len(path)-1):
            self.disset.union(path[i], path[i+1])

    def add_edges(self,path):
        for ind,node in enumerate(path):
            if node not in self.node_dict:
                self.node_dict[node]=len(self.node_dict.keys())
            if ind==0:
                continue
            self.g.add_edge(self.node_dict[path[ind-1]],self.node_dict[node])

    def add_edges_beta(self,path):
        """
        Can be used additionally to add_edges in order to add all nodes that have not
        been expanded
        """
        for node,c in path:
            if node not in self.node_dict:
                self.node_dict[node]=len(self.node_dict.keys())

            for node2 in c:
                if node2 not in self.node_dict:
                    self.node_dict[node2]=len(self.node_dict.keys())

                self.g.add_edge(self.node_dict[node],self.node_dict[node2])



    def dfs(self,start):
        """
        Single iteration of dfs on graph. Start at a vertex, find non visited neighbors
        """

        path=[]
        path_beta = []
        curset = set()
        cur = start
        max_depth=self.max_depth

        while self.disset.find(cur) not in curset:
            path.append(cur)
            curset.add(self.disset.find(cur))
            self.visited.add(cur)
            candidates = self.get_neighbors_of_comp(list(self.disset.get_set(cur)))
            if len(candidates)==0:
                break
            path_beta.append((cur,candidates))
            cur = candidates[0]
            if max_depth:
                for c in candidates:
                    if c not in self.visited:
                        cur=c
                        break
                max_depth-=1
            else:
                for c in candidates:
                    if self.disset.find(c) in curset:
                        cur=c
                        break


        cur_comp = self.disset.find(cur)
        for ind,c in enumerate(path):
            if cur_comp==self.disset.find(c):
                self.merge(path[ind:])
                break
        path.append(cur)
        self.add_edges(path)
        #self.add_edges_beta(path_beta)
        #print(path)
        self.path.extend(path)
        return cur



#DEPRECATED
class Fancy_Heuristic_Graph(Base_Graph):
    """
    Alternative to Forbidden States, but also creates a graph, not used.
    """

    def __init__(self,CA,num_checks=5,max_runs=20,show_progress=False,used8=False):
        super(Fancy_Heuristic_Graph,self).__init__(CA)
        self.num_checks = num_checks
        self.max_runs=max_runs
        self.impossibles = set()
        n=self.CA.num_states
        self.max_nodes = (n*(n-1))/2
        self.used8=used8

        if show_progress:
            self.tqdm=tqdm


    def check_set(self,disset):
        for (x,y) in self.impossibles:
            if disset.check_same(x,y):
                return (x,y)
        return None

    def run(self):
        l = list(range(self.CA.num_states))
        random.shuffle(l)
        for i in self.tqdm(l):
            #print(len(self.impossibles))
            for j in range(i+1,self.CA.num_states):
                #print(len(self.impossibles))
                if (i,j) not in self.impossibles:
                    if self.single_run(i,j):
                        continue

    def any_cg(self):
        return len(self.impossibles)!=self.max_nodes

    def single_run(self,a,b):
        disset = DisjointSet()
        disset.union(a,b)
        max_runs = self.max_runs
        prev_size=0
        check_count=0
        start_num = self.get_node_num((a,b))
        while disset.max_size<self.CA.num_states and max_runs>0:
            #print(disset.max_size,self.CA.num_states)
            cur = list(disset.get_random_set())
            diffs = self.CA.check_diff_equiv_set(cur,self.num_checks)
            for (i,j) in diffs:
                unioned = disset.union(i,j)
                if unioned:
                    end_num = self.get_node_num((i,j))
                    #if end_num!=start_num:
                    #    self.g.add_edge(start_num,end_num)
            if disset.max_size == prev_size:
                max_runs-=1
                continue
            else:
                prev_size = disset.max_size
                max_runs=self.max_runs
            if check_count==0:
                combined = self.check_set(disset)
                if combined is not None:
                    comb_num = self.get_node_num(combined)

                    if self.used8:
                        for val in self.get_other_d8((a,b)):
                            num = self.get_node_num(val)
                            self.impossibles.add(val)
                            self.g.add_edge(num,comb_num)

                    num = self.get_node_num((a,b))
                    self.g.add_edge(num,comb_num)
                    self.impossibles.add((a,b))
                    return True
                check_count=10
            else:
                check_count-=1
        if disset.max_size==self.CA.num_states:
            if self.used8:
                for val in self.get_other_d8((a,b)):
                    self.impossibles.add(val)
            self.impossibles.add((a,b))
            return True
        #print(disset.max_size)
        #if disset.max_size>=500:
        #    print(list(sorted(list(set(range(512))-disset.get_random_set()))))
        return False
