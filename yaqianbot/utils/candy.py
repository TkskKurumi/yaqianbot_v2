import time
import inspect


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
