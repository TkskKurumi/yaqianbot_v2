import time
import inspect
import sys
from os import path


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
