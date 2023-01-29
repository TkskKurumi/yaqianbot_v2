from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from time import time as nowtime
from collections import defaultdict
import traceback
from ..utils.timing import timer
from ..utils.candy import locked
from threading import Lock

cnt_lock = Lock()
pending_cnt = defaultdict(int)
running_cnt = defaultdict(int)

pool = ThreadPoolExecutor(max_workers=256)




task_timer = timer()

def threading_run(f):
    if(isinstance(f, str)):
        taskname = f
    elif(callable(f)):
        taskname = f.__name__
    else:
        raise TypeError(type(f))

    def deco(func):
        @wraps(func)
        def _inner(*args, **kwargs):
            try:
                startime = nowtime()

                with locked(cnt_lock):
                    running_cnt[taskname] += 1
                    pending_cnt[taskname] -= 1
                
                try:
                    ret = func(*args, **kwargs)
                except Exception as e:
                    running_cnt[taskname] -= 1
                    raise e
                with locked(cnt_lock):
                    running_cnt[taskname] -= 1
                task_timer.count(taskname, nowtime()-startime)
                if(nowtime()-startime>5):
                    print(taskname,"completed after %.2f seconds"%(nowtime()-startime))
                return ret
            except Exception as e:
                traceback.print_exc()
                raise e
        @wraps(func)
        def inner(*args, **kwargs):
            pending_cnt[taskname] += 1
            return pool.submit(_inner, *args, **kwargs)
        return inner
    if(isinstance(f, str)):
        return deco
    elif(callable(f)):
        return deco(f)
    else:
        raise TypeError(type(f))


if(__name__ == "__main__"):
    # test
    import time

    @threading_run
    def f():
        print("f")
        time.sleep(1)
        return "foo"

    @threading_run("custom task name")
    def ff():
        print("ff")
        time.sleep(2)
        return "bar"
    tasks = []
    for i in range(3):
        tasks.append(f())
        tasks.append(ff())
    for i in tasks:
        print(i.result())