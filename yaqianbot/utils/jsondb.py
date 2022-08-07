from collections import defaultdict
from .myhash import base32
from os import path
from .candy import locked
import json
import os
from threading import Lock
from glob import glob

def loadjson(pth):
    with open(pth, "r") as f:
        ret = json.load(f)
    return ret


def ensure_dir(pth):
    dir = path.dirname(pth)
    if(not path.exists(dir)):
        os.makedirs(dir)


def savejson(pth, j):
    ensure_dir(pth)
    with open(pth, "w") as f:
        json.dump(j, f)
    return j


class jsondb:
    def __init__(self, pth, method=lambda x: base32(x, length=3)):
        self.pth = pth
        self.method = method
        self.d = defaultdict(dict)
        self.loaded = set()
        self.lock = Lock()

    def _get_pth(self, key):
        hashed = self.method(key)
        pth = path.join(self.pth, hashed+".json")
        return pth
    def _load_all(self):
        for i in glob(path.join(self.pth, "*.json")):
            self._load(pth = i)
    def _asdict(self):
        self._load_all()
        ret = dict()
        for i, j in self.d.items():
            ret.update(j)
        return ret
    def _iter(self):
        return self._asdict().__iter__()
    def __iter__(self):
        with locked(self.lock):
            ret = self._iter()
        return ret
    def _load(self, key=None, pth=None):
        if(pth is None):
            if(key is not None):
                hashed = self.method(key)
                pth = self._get_pth(key)
            else:
                raise Exception("Either key or pth should be specified")
        else:
            bn = path.basename(pth)
            bn, ext = path.splitext(bn)
            hashed = bn
        if(hashed in self.loaded):
            return self.d[hashed]
        if(path.exists(pth)):
            j = loadjson(pth)
            self.d[hashed] = j
        else:
            j = {}
        self.loaded.add(hashed)
        return j
    def _save(self, key):
        self._load(key)
        savejson(self._get_pth(key), self.d[self.method(key)])
    def save(self, key):
        with locked(self.lock):
            ret = self._save(key)
        return ret
    def _setitem(self, key, value):
        self._load(key)
        self.d[self.method(key)][key] = value
        self._save(key)
    
    def __setitem__(self, key, value):
        with locked(self.lock):
            ret = self._setitem(key, value)
        return ret

    def _getitem(self, key):
        self._load(key)
        return self.d[self.method(key)][key]

    def __getitem__(self, key):
        with locked(self.lock):
            ret = self._getitem(key)
        return ret

    def _contains(self, key):
        self._load(key)
        return key in self.d[self.method(key)]

    def __contains__(self, key):
        with locked(self.lock):
            ret = self._contains(key)
        return ret

    def get(self, key, default=None):
        if(key in self):
            return self[key]
        else:
            return default
    def _items(self):
        return self._asdict().items()
    def items(self):
        with locked(self.lock):
            ret = self._items()
        return ret
    def _pop(self, key):
        ret = self._getitem(key)
        hashed = self.method(key)
        self.d[hashed].pop(key)
        self._save(key)
        return ret
    def pop(self, key):
        with locked(self.lock):
            ret = self._pop(key)
        return ret
if(__name__ == "__main__"):
    import time
    dic = jsondb("/tmp/tmpjsondb")
    dic["foo"]="bar"
    print(dic._asdict())
    print(list(dic))