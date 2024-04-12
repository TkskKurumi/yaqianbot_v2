from threading import Lock
from functools import wraps
import time
import heapq

class _Profiler:
    items = {}
    @classmethod
    def open(cls, func, *args, **kwargs):
        i = id(func)
        if(i in cls.items):
            return cls.items[i]
        else:
            ret = cls(func, *args, **kwargs)
            cls.items[i] = ret
            return ret
    def __init__(self, foo, eps=0.1):
        self.foo = foo
        self.eps = eps
        self.durations = []
        self.lck = Lock()
    def __call__(self, *args, **kwargs):
        start = time.time()
        try:
            ret = self.foo(*args, **kwargs)
        except Exception as e:
            raise e
        end = time.time()
        elapsed = end-start
        if(elapsed>self.eps):
            itm = (end, elapsed)
            with self.lck:
                heapq.heappush(self.durations, itm)
            with self.lck:
                clr = self.clear()
            if(clr):
                print(self)
        return ret
    def clear(self, t=None):
        while(self.durations):
            if(t):
                if(self.durations[0][0]<t-600):
                    heapq.heappop(self.durations)
                    continue
            if(len(self.durations)>64):
                heapq.heappop(self.durations)
                continue
            return self.durations
    def __str__(self):
        times = len(self.durations)
        tot = sum(i[1] for i in self.durations)
        if(times):
            return "<Profile %s %.3fsec/it>"%(self.foo, tot/times)
        else:
            return "<Profile %s>"%(self.foo)
def Profile(func):
    pf = _Profiler.open(func)
    @wraps(func)
    def inner(*args, **kwargs):
        return pf(*args, **kwargs)
    return inner
