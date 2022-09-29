from threading import Lock
from .candy import lockedmethod, locked

class LockPool:
    def __init__(self):
        self.lck = Lock()
        self.locks = dict()
    @lockedmethod
    def __call__(self, id):
        if(id in self.locks):
            return self.locks[id]
        ret = Lock()
        self.locks[id] = ret
        return ret
    @lockedmethod
    def __getitem__(self, id):
        if(id in self.locks):
            return self.locks[id]
        ret = Lock()
        self.locks[id] = ret
        return ret
    def locked(self, id):
        lck = self[id]
        # print(self.locks)
        return locked(lck)