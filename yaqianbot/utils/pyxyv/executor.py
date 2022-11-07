from concurrent.futures import ThreadPoolExecutor
from functools import wraps


def create_pool(max_workers=4):
    return ThreadPoolExecutor(max_workers=max_workers)


pool = create_pool()


def create_task(func, *args, pool=pool, **kwargs):
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
