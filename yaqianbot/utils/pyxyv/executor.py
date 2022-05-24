from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor()

def create_pool():
    return ThreadPoolExecutor()
def create_task(func, *args, **kwargs):
    return pool.submit(func, *args, **kwargs)
