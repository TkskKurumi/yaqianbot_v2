import inspect
from mimetypes import init


def _max(a, b):
    return max(a, b)


def _zeros(idx):
    return 0


def _mid():
    frame = inspect.currentframe().f_back
    # print(frame.f_locals)
    start = frame.f_locals["start"]
    end = frame.f_locals["end"]
    return (start+end) >> 1


ops = {"max": _max}
# init = {"max":lambda idx:(0, idx)}


class SegTree:
    def merge(self):
        for k, v in self.ops.items():
            self.v[k] = v(self.le.v[k], self.ri.v[k])
    def merge2(self, a, b):
        ret = dict()
        for k, v in a.items():
            ret[k] = self.ops[k](v, b[k])
        return ret
    def __init__(self, start, end, merge_operator=ops, initializer=_zeros):
        self.start = start
        self.end = end
        self.mid = (start+end)>>1
        self.v = dict()
        self.ops = merge_operator
        if(start < end):
            mid = (start+end) >> 1
            self.le = SegTree(start, mid, merge_operator=merge_operator, initializer=initializer)
            self.ri = SegTree(mid+1, end, merge_operator=merge_operator, initializer=initializer)
            self.merge()
        else:
            if(callable(initializer)):
                for k in ops:
                    self.v[k] = initializer(start)
            elif(isinstance(initializer, list)):
                for k in ops:
                    self.v[k] = initializer[start]
            else:
                raise TypeError("Initializer should be Callable(index) or List[index], unexpected %s"%type(initializer))
            # print(start, self.v, initializer)
    def query(self, start, end):
        # assert self.start<=end and start<=self.end
        if(start<=self.start and self.end<=end):
            return self.v
        else:
            mid = self.mid
            if(start<=mid and mid+1<=end):
                return self.merge2(self.le.query(start, end), self.ri.query(start, end))
            elif(start>mid):
                return self.ri.query(start, end)
            elif(end<=mid):
                return self.le.query(start, end)
            else:
                raise Exception("Internal Error")
            
    @property
    def is_leaf(self):
        return self.start == self.end
    # @property
    # def mid(self):
    #     return (self.start+self.end)>>1
    def __setitem__(self, idx, v):
        if(self.is_leaf):
            for k in self.v:
                self.v[k] = v
        else:
            if(idx<=self.mid):
                self.le[idx]=v
            self.merge()
if(False and __name__=="__main__"):
    ls = [1,4,2,8,5,7]
    st = 0
    ed = len(ls)-1
    tree = SegTree(0, ed, initializer=ls)
    for i in range(len(ls)):
        for j in range(i, len(ls)):
            print(i, j , ls[i:j+1], tree.query(i, j))
if(__name__=="__main__"):
    import random
    from timeit import timeit
    length = 1000
    query = 10000
    def naive():
        ls = list(range(length))
        random.shuffle(ls)
        for i in range(query):
            u = random.randrange(length)
            v = random.randrange(length)
            u, v=min(u,v),max(u,v)
            q = max(ls[u:v+1])
    def seg():
        ls = list(range(length))
        random.shuffle(ls)
        tree = SegTree(0, length-1, initializer=ls)
        for i in range(query):
            u = random.randrange(length)
            v = random.randrange(length)
            u, v=min(u,v),max(u,v)
            q = tree.query(u, v)
    print(timeit(stmt=naive, number=1))
    print(timeit(stmt=seg, number=1))

