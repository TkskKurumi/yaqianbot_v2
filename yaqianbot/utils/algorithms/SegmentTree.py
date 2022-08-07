inf = float("inf")


class Segment:
    def __init__(self, lo, up, initial=None):
        self.lo = lo
        self.up = up
        self.maximum = 0
        self.sum = 0
        if(lo < up):
            mid = (lo+up) >> 1
            self.le = Segment(lo,  mid)
            self.ri = Segment(mid+1, up)
            self.sum = self.le.sum+self.ri.sum
            self.maximum = max(self.le.maximum, self.ri.maximum)
        elif(initial):
            self.sum = self.maximum = initial[lo]
        else:
            self.sum = self.maximum = 0
    def set(self, idx, value):
        if(self.lo <= idx and idx <= self.up):
            pass
        else:
            return
        if(self.lo < self.up):
            mid = (self.lo+self.up) >> 1
            if(idx <= mid):
                self.le.set(idx, value)
            else:
                self.ri.set(idx, value)
            self.sum = self.le.sum+self.ri.sum
            self.maximum = max(self.le.maximum, self.ri.maximum)
        else:
            self.sum = self.maximum = value

    def add(self, idx, cnt):
        if(self.lo <= idx and idx <= self.up):
            pass
        else:
            return
        if(self.lo < self.up):
            mid = (self.lo+self.up) >> 1
            if(idx <= mid):
                self.le.add(idx, cnt)
            else:
                self.ri.add(idx, cnt)
            self.maximum = max(self.le.maximum, self.ri.maximum)
            self.sum = self.le.sum + self.ri.sum
        else:
            self.sum += cnt
            self.maximum = self.sum

    def query(self, lo, up):
        if(lo > self.up):
            return 0
        if(up < self.lo):
            return 0
        if(lo <= self.lo and self.up <= up):
            return self.sum
        if(self.lo < self.up):
            ret = self.le.query(lo, up)+self.ri.query(lo, up)
            return ret

    def query_max(self, lo, up):
        if(lo > self.up):
            return -inf
        if(up < self.lo):
            return -inf
        if(lo <= self.lo and self.up <= up):
            return self.maximum
        if(self.lo < self.up):
            ret = max(self.le.query_max(lo, up), self.ri.query_max(lo, up))
            return ret
        assert False, "Internal Error"
    