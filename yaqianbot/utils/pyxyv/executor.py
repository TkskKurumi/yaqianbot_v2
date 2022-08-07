from concurrent.futures import ThreadPoolExecutor
from functools import wraps
pool = ThreadPoolExecutor()

def create_pool():
    return ThreadPoolExecutor()
def create_task(func, *args, **kwargs):
    return pool.submit(func, *args, **kwargs)
def threading_run(func):
    @wraps(func)
    def inner(*args, **kwargs):
        ret = pool.submit(func, *args, **kwargs)
        return ret
    return inner
def run_in_pool(pool):
    def deco(func):
        @wraps(func)
        def inner(*args, **kwargs):
            ret = pool.submit(func, *args, **kwargs)
            return ret
        return inner
    return deco