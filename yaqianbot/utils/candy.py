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
def simple_send(messages):
    frame = inspect.currentframe().f_back
    mes = frame.f_locals.get("mes") 
    message = frame.f_locals.get("message")
    if(hasattr(mes, "response_sync")):
        ret = mes.response_sync(messages)
    elif(hasattr(message, "response_sync")):
        ret = message.response_sync(messages)
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