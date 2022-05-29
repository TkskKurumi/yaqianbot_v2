
import random
import numpy as np
_inf = float('inf')


def quick_sort(ls, l=None, r=None, no_recursion=False):
    if(l is None):
        l = 0
    if(r is None):
        r = len(ls)-1
    _l = l
    _r = r
    ref = ls[l]

    def sw(a, b):
        tmp = ls[a]
        ls[a] = ls[b]
        ls[b] = tmp
    while(l != r):
        while(l != r and ls[r] >= ref):
            r -= 1
        sw(l, r)
        while(l != r and ls[l] <= ref):
            l += 1
        sw(l, r)
    if(not no_recursion):
        lr = l-1
        rl = l+1
        if(_l < lr):
            quick_sort(ls, _l, lr)
        if(rl < _r):
            quick_sort(ls, rl, _r)
    return l


def quick_rank(ls, rank):
    l = 0
    r = len(ls)-1
    while(True):
        mid = quick_sort(ls, l, r, no_recursion=True)
        if(mid == rank):
            return ls[mid]
        if(rank <= mid):
            r = mid
        else:
            l = mid+1


class point:
    def __init__(self, arr):
        self.arr = arr
        self.hash = None
        self.id = None
        self.nd = len(arr)

    def dist(self, other):
        ret = 0
        # for idx,i in enumerate(self.arr):
        for idx in range(self.nd):
            ret += (self.arr[idx]-other.arr[idx])**2
        return ret

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return self.arr.__repr__()

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.arr)

    def __mul__(self, other):
        if(isinstance(other, point)):
            ret = 0
            for idx, i in enumerate(self.arr):
                ret += i*other.arr[idx]
            return ret
        elif(isinstance(other, int) or isinstance(other, float)):
            arr = [i*other for idx, i in enumerate(self.arr)]
            return point(arr)
        return NotImplemented

    def __add__(self, other):
        arr = [i+other.arr[idx] for idx, i in enumerate(self.arr)]
        return point(arr)

    def __sub__(self, other):
        arr = [i-other.arr[idx] for idx, i in enumerate(self.arr)]
        return point(arr)

    def __truediv__(self, other):
        if(isinstance(other, int) or isinstance(other, float)):
            arr = [i/other for idx, i in enumerate(self.arr)]
            return point(arr)
        else:
            return NotImplemented

    def distO(self):
        return sum([i*i for i in self.arr])**0.5

    def dist_line(self, A, B):
        AB = B-A
        AP = self-A
        dot = AB*AP
        dAP = AP.distO()
        if(AB.distO() < 1e-8):
            return dAP
        #assert dAP>=0,str(dAP)
        #assert AB.distO()>0,str(AB.distO())
        dAH = dot/AB.distO()  # AH
        tmp = dAP*dAP-dAH*dAH
        assert tmp > -1e-8
        if(tmp < 0):
            return 0
        return abs(tmp**0.5)

    def dist_bisector(self, A, B):
        C = (A+B)/2
        PC = C-self
        AB = B-A
        dot = AB*PC
        return abs(dot/AB.distO())


def variance(ls):
    avg = sum(ls)/len(ls)
    ret = 0
    for i in ls:
        ret += (i-avg)**2
    return ret


class kdt:
    def __init__(self):
        self.node_points = []
        self.left_child = []
        self.right_child = []
        self.axis = []
        self.value = []
        self.root = None

    def _new_index(self):
        self.node_points.append(None)
        self.left_child.append(None)
        self.right_child.append(None)
        self.axis.append(None)
        self.value.append(None)
        if(self.root is None):
            self.root = len(self.node_points)-1
        self.size = len(self.node_points)
        return self.size-1

    def initiate_statics(self):
        self._cnt_call_ann_top = 0
        self._cnt_call_ann_recursive = 0
        self._cnt_calc_dist = 0
        self._sum_leaf = 0
        self._cnt_recall = 0
        self._sum_leaf_depth = 0

    def print_performance(self):
        print("call ann", self._cnt_call_ann_recursive/self._cnt_call_ann_top)
        print("calc dist", self._cnt_calc_dist/self._cnt_call_ann_top)
        print("recall", self._cnt_recall/self._cnt_call_ann_top)

    def build(self, points, stop_num=None, depth=0, stop_depth=20):
        # print(len(points),depth)
        if(depth == 0):

            self.initiate_statics()
            # points = list(set(points))
            if(stop_num is None):
                # stop_num=len(points)**0.25
                stop_num = 1
            #print('\n'.join(sorted([str(tuple(i.arr)) for i in points])),len(points))
            for idx, i in enumerate(points):
                if(not isinstance(i, point)):
                    points[idx] = point(i)
            for id, i in enumerate(points):
                i.id = id
        u = self._new_index()
        if(len(points) <= stop_num or stop_depth <= depth):
            self.node_points[u] = points
            self._sum_leaf += 1
            self._sum_leaf_depth += depth
            return u

        nd = len(points[0].arr)
        vars = []
        for i in range(nd):
            ls = [p.arr[i] for p in points]
            var = variance(ls)
            vars.append((var, i))
        _, axis = max(vars)
        ls = [p.arr[axis] for p in points]
        rank = (len(ls)-1)//2
        value = quick_rank(ls, rank)
        if(value >= max(ls)):
            value = sum(ls)/len(ls)
        self.axis[u] = axis
        self.value[u] = value
        lpoints = list()
        rpoints = list()
        for p in points:
            if(p.arr[axis] <= value):
                lpoints.append(p)
            else:
                rpoints.append(p)

        self.left_child[u] = self.build(
            lpoints, stop_num=stop_num, depth=depth+1, stop_depth=stop_depth)
        self.right_child[u] = self.build(
            rpoints, stop_num=stop_num, depth=depth+1, stop_depth=stop_depth)
        return u

    def ann1(self, p, u=None, cut_dist=_inf, with_dist=False, recall=False):

        if(u is None):
            self._cnt_call_ann_top += 1
            ret, retd = self.ann1(p, u=self.root, with_dist=True,recall=recall)
            assert ret is not None
            if(with_dist):
                return ret, retd
            else:
                return ret
        self._cnt_call_ann_recursive += 1
        # leaf node
        if(self.node_points[u] is not None):
            ret = None
            retd = None
            for i in self.node_points[u]:
                if(ret is None):
                    ret = i
                    self._cnt_calc_dist += 1
                    retd = p.dist(ret)
                    continue
                self._cnt_calc_dist += 1
                id = p.dist(i)
                if(id < retd):
                    ret = i
                    retd = id
            # assert (ret is not None), 'ret is None,%s' % self.node_points[u]
            return ret, retd
        # print(self.axis[u],self.value[u],self.node_points[u],u)
        axis = self.axis[u]
        value = self.value[u]
        if(p.arr[axis] <= value):
            ret, retd = self.ann1(p, self.left_child[u], cut_dist=cut_dist,recall=recall)
            mn = min(retd, cut_dist)
            if(recall and mn > abs(p.arr[axis]-value)):
                self._cnt_recall += 1
                ret1, ret1d = self.ann1(
                    p, self.right_child[u], cut_dist=mn,recall=recall)

                if(ret1d < retd):
                    # assert (ret1 is not None)
                    return ret1, ret1d
                else:
                    # assert (ret is not None)
                    return ret, retd
            else:
                return ret, retd
        else:
            ret, retd = self.ann1(p, self.right_child[u], cut_dist=cut_dist,recall=recall)
            # assert (ret is not None), "%s,%s" % (
            #     self.right_child[u], self.node_points[self.right_child[u]])
            mn = min(retd, cut_dist)
            if(recall and mn > abs(p.arr[axis]-value)):
                self._cnt_recall += 1
                ret1, ret1d = self.ann1(
                    p, self.left_child[u], cut_dist=mn,recall=recall)

                # ret1d=ret1.dist(p)
                if(ret1d < retd):
                    assert (ret1 is not None)
                    return ret1, ret1d
                else:
                    assert (ret is not None)
                    return ret, retd
            else:
                return ret, retd


if(__name__ == '__main__'):
    import random

    def rand_nd(n):
        return tuple([random.random()*255 for i in range(n)])
        # return tuple([random.choice([1,2,3]) for i in range(n)])
    nd = 3
    num = 300
    points = [point(rand_nd(nd)) for i in range(num)]
    pointss = [point(rand_nd(nd)) for i in range(num)]
    
    def find_nearest_basic():
        sumd = 0
        for p in pointss:
            ret=None
            retd = 0
            for i in points:
                id = i.dist(p)
                if(ret is None):
                    ret = i
                    retd = id
                if(id < retd):
                    retd = id
                    ret = i
            sumd += retd
        return sumd
    K = kdt()
    K.build(points, stop_num=num**0.5)

    def find_nearest_kdt():
        sumd=0
        for p in pointss:
            
            ret, retd = K.ann1(p, with_dist=1, recall=True)
            sumd += retd
        return sumd 
    from timeit import timeit
    print(timeit(stmt=find_nearest_basic, number=10))
    print(timeit(stmt=find_nearest_kdt, number=10))
    K.print_performance()
    print(find_nearest_basic())
    print(find_nearest_kdt())
