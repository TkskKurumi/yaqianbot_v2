
def GREATER(a, b): return a > b


class heapget:
    def __init__(self, heap, type):
        self.heap = heap
        if(type == "fa"):
            self.fidx = lambda idx: (idx-1) >> 1
        elif(type == "le"):
            self.fidx = lambda idx: idx*2+1
        else:
            self.fidx = lambda idx: idx*2+2

    def __call__(self, idx):
        return self.fidx(idx)

    def __getitem__(self, idx):
        return self.heap[self.fidx(idx)]


class heap(list):
    def __init__(self, contents = None, fcmp=GREATER):
        # super().__init__(*args, **kwargs)
        self.fa = heapget(self, "fa")
        self.le = heapget(self, "le")
        self.ri = heapget(self, "ri")
        self.fcmp = fcmp
        if(contents is not None):
            for i in contents:
                self.push(i)
    def swap(self, i, j):
        tmp = self[i]
        self[i] = self[j]
        self[j] = tmp

    def push(self, element):
        fcmp = self.fcmp
        idx = len(self)
        self.append(element)
        while(idx != 0):
            if(fcmp(element, self.fa[idx])):
                self.swap(idx, self.fa(idx))
                idx = self.fa(idx)
            else:
                break
        return None

    def topK(self, k):
        if(k > len(self)):
            raise IndexError(
                "cannot get top-%d from heap of length %d" % (k, len(self)))
        ret = []
        for i in range(k):
            ret.append(self.pop())
        for i in ret:
            self.push(i)
        return ret
    def pop(self):
        if(not self):
            raise IndexError("pop from empty heap")
        fcmp = self.fcmp
        ret = self[0]
        self.swap(0, len(self)-1)
        super().pop()
        idx = 0
        while(self.le(idx) < len(self)):
            greater = self.le[idx]
            greater_idx = self.le(idx)
            if(self.ri(idx) < len(self)):
                if(fcmp(self.ri[idx], self.le[idx])):
                    greater = self.ri[idx]
                    greater_idx = self.ri(idx)
            if(fcmp(greater, self[idx])):
                self.swap(idx, greater_idx)
                idx = greater_idx
            else:
                break
        return ret


if(__name__ == "__main__"):
    a = heap("abcdfgjasdoijo", fcmp = lambda a, b:a<b)
    print(a.topK(3))