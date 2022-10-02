from ..lvldb import TypedLevelDB
from time import time
opened_mgr = dict()


class ActionPointMgr:
    @classmethod
    def open(cls, pth, *args, **kwargs):
        if(pth in opened_mgr):
            ret = opened_mgr[pth]
            ret.reinit(*args, **kwargs)
            return ret
        else:
            ret = cls(pth, *args, **kwargs)
            opened_mgr[pth] = ret
            return ret

    def reinit(self, max_point=None, per_hour=None):
        if(max_point is not None):
            self.max_point = max_point
        if(per_hour is not None):
            self.per_hour = per_hour

    def __init__(self, pth, max_point=100, per_hour=10):
        self.db = TypedLevelDB(pth)
        self.max_point = max_point
        self.per_hour = per_hour

    def bonus(self, uid, n):
        return self.cost(uid, -n)

    def cost(self, uid, n):
        entry = self.get_entry(uid)
        entry["ap"] -= n
        self.db[uid] = entry
        return entry["ap"]

    def time_targeting(self, uid, n):
        ap = self.get_ap(uid)
        if(n < ap):
            return 0
        return (n-ap)/self.per_hour*3600

    def time_targeting_str(self, uid, n):
        ret = []
        tsec = self.time_targeting(uid, n)
        sec = tsec % 60
        ret.append("%.1f秒" % sec)
        tmin = int(tsec)//60
        min = tmin % 60
        if(min):
            ret.append("%d分" % min)
        thour = tmin//60
        if(thour):
            ret.append("%d小时" % thour)
        return "".join(ret[::-1])+"(每小时恢复%.1f)" % self.per_hour

    def afford(self, uid, n):
        ap = self.get_ap(uid)
        return ap >= n

    def __getitem__(self, uid):
        return self.get_ap(uid)

    def get_entry(self, uid):
        if(uid in self.db):
            entry = self.db[uid]
        else:
            entry = {"id": uid, "ap": self.max_point, "time": time()}
        return entry

    def get_ap(self, uid):
        entry = self.get_entry(uid)
        ap = entry["ap"]
        tm = entry["time"]
        t_delta = time()-tm
        increase = t_delta/3600*self.per_hour
        if(ap < self.max_point):
            ap = min(ap+increase, self.max_point)
        else:
            # remain greater than max, for bonus
            pass
        entry["ap"] = ap
        entry["time"] = time()
        self.db[uid] = entry
        return entry["ap"]
