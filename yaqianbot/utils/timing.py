from collections import defaultdict
from .candy import locked
from threading import Lock
from time import time as nowtime
import heapq, time

class Limit:
    def __init__(self, period, times):
        self.times = times
        self.period = period
        self.ts = []
        self.lck = Lock()
    def _clean(self):
        tm = nowtime()-self.period
        while(self.ts and self.ts[0]<tm):
            self.ts.pop(0)

    def __call__(self):
        with self.lck:
            self._clean()
            if(len(self.ts)>=self.times):
                tm = nowtime()
                until = self.ts[-self.times]+self.period
                assert until>=tm
                print("Too fast, throttling %.3f seconds"%(until-tm))
                time.sleep(until-tm)
            self.ts.append(nowtime())
        





class speedo:
    def __init__(self, period_t=2, period_n=5):
        self.ts = []
        self.lock = Lock()
        self.t = period_t
        self.n = period_n

    def clear(self):
        t = nowtime()
        with locked(self.lock):
            while(len(self.ts) > self.n and (self.ts[0] < t-self.t)):
                heapq.heappop(self.ts)

    def count(self):
        with locked(self.lock):
            t = nowtime()
            heapq.heappush(self.ts, t)
        self.clear()

    def result(self):
        self.clear()
        return len(self.ts)/(nowtime()-self.ts[0])


class timer:
    def __init__(self):
        self.runcount = defaultdict(int)
        self.runtime = defaultdict(int)
        self.lock = Lock()

    def count(self, taskname, runtime):
        """
        Count the task runtime.
        """
        with locked(self.lock):
            self.runcount[taskname] += 1
            self.runtime[taskname] += runtime

    def __getitem__(self, taskname):
        if(self.runcount[taskname] == 0):
            return 0
        return self.runtime[taskname]/self.runcount[taskname]

    def __str__(self):
        ret = []

        def prt(*args, sep=" ", end="\n"):
            ret.append(sep.join([str(_) for _ in args]))
            ret.append(end)
        st = []
        for taskname in self.runtime:
            itm = self[taskname], taskname
            st.append(itm)
        for per, taskname in sorted(st, reverse=True):
            prt('run "%s" %d times in %.2f seconds, %.2f per run' %
                (taskname, self.runcount[taskname], self.runtime[taskname], per))
        return "".join(ret)

    def __repr__(self):
        return self.__str__


if(__name__ == "__main__"):
    import time
    import random
    spd = speedo()
    for i in range(6):
        time.sleep(0.33)
        spd.count()
    print(spd.result())
