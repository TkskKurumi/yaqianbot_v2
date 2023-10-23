import time
import inspect
import sys
from os import path

class FakeLock:
    # for debug
    def __init__(self):
        pass
    def acquire(self):
        print(self, ".acquire")
    def release(self):
        print(self, ".release")

def dump_obj(obj, visited=None, depth=3, root=True):
    if (visited is None):
        visited = set()
    if (depth<=0):
        return str(obj)
    clsname = obj.__class__.__name__
    buffer = []
    def prt(*args, sep=" ", end="\n", **kwargs):
        buffer.append("%s%s"%(sep.join(str(i) for i in args), end))
    def indentd(st: str):
        ret = []
        spl = st.split("\n")
        for idx, i in enumerate(spl):
            if (idx==0 or idx==len(spl)-1):
                ret.append(i)
            else:
                ret.append("    "+i)
        return "\n".join(ret)

    if (isinstance(obj, int)):
        return "int(%s)"%obj
    elif (isinstance(obj, str)):
        return '"%s"'%obj
    elif (isinstance(obj, float)):
        return 'float(%s)'%obj
    elif (obj is None):
        return "None"
    
    if(id(obj) in visited):
        return "<%s object %d>"%(clsname, id(obj))

    visited.add(id(obj))

    if (isinstance(obj, dict)):
        prt("{")
        for k, v in obj.items():
            kstr = dump_obj(k, visited, depth-1, False)

            vstr = indentd(dump_obj(v, visited, depth-1, False))

            prt("%s: %s"%(kstr, vstr))
        prt("}", end="")
        return "".join(buffer)
    elif (isinstance(obj, list)):
        prt("[")
        for i in obj:
            istr = indentd(dump_obj(i, visited, depth-1, False))
            prt("%s,"%istr)
        prt("]", end="")
        return "".join(buffer)

    if (hasattr(obj, "__dict__")):
        prt("%s("%clsname)
        for k, v in obj.__dict__.items():
            vstr = indentd(dump_obj(v, visited, depth-1, False))
            prt("%s=%s,"%(k, vstr))
        prt(")", end="")
        if (root):
            return indentd("".join(buffer))
        else:
            return "".join(buffer)
    else:
        return str(obj)

def simple_send(messages, **kwargs):
    frame = inspect.currentframe().f_back
    mes = frame.f_locals.get("mes") 
    message = frame.f_locals.get("message")
    if(hasattr(mes, "response_sync")):
        ret = mes.response_sync(messages, **kwargs)
    elif(hasattr(message, "response_sync")):
        ret = message.response_sync(messages, **kwargs)
    else:
        raise Exception("Cannot do simple_send in this session")
    del frame
    return ret


class print_time:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

    def __enter__(self):
        self.time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time()-self.time
        if(t > 1 and self.enabled):
            print("%s uses %.1f seconds" % (self.name, t))
def lockedmethod(func, lck=None):
    def inner(self, *args, **kwargs):
        with locked(self.lck):
            ret = func(self, *args, **kwargs)
        return ret
    if(lck is None):
        return inner
    def inner1(*args, **kwargs):
        nonlocal lck
        with locked(lck):
            ret = func(*args, **kwargs)
        return ret
    return inner1
class locked:
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        return
class released:
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.release()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.acquire()
        return

class __LINE__(object):
    def __init__(self, n=1):
        self.n = n

    def __repr__(self):
        try:
            raise Exception
        except:
            frame = sys.exc_info()[2].tb_frame
            for i in range(self.n):
                frame = frame.f_back
            return str(frame.f_lineno)

    def __int__(self):
        try:
            raise Exception
        except:
            frame = sys.exc_info()[2].tb_frame
            for i in range(self.n):
                frame = frame.f_back
            return frame.f_lineno


class __FILE__(object):
    def __init__(self, n=1):
        self.n = n

    def __repr__(self):
        try:
            raise Exception
        except:
            frame = sys.exc_info()[2].tb_frame
            for i in range(self.n):
                frame = frame.f_back
            return str(frame.f_code.co_filename)


class __FUNC__(object):
    def __init__(self, n=1):
        self.n = n

    def __repr__(self):
        try:
            raise Exception
        except:
            frame = sys.exc_info()[2].tb_frame
            for i in range(self.n):
                frame = frame.f_back
            return str(frame.f_code.co_name)


def log_header():
    file = str(__FILE__(2))
    if(len(file) < 10):
        file = file+" "*(10-len(file))
    else:
        file = "..."+file[-7:]
    return "%s: %03d" % (file, __LINE__(2))
if(__name__=="__main__"):
    # test
    def foo():
        with locked(FakeLock()):
            print("foo")
            return "printed result"
    print(foo())