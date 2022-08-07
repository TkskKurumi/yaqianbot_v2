from datetime import datetime, timezone, timedelta
from ..backend.configure import bot_config
def now():
    tz = timezone(timedelta(hours=int(bot_config.get("tz",  0))))
    ret = datetime.now().astimezone(tz)
    return ret

def fromtimestamp(ts):
    tz = timezone(timedelta(hours=int(bot_config.get("tz",  0))))
    ret = datetime.fromtimestamp(ts).astimezone(tz)
    return ret
if(__name__=="__main__"):
    a = now()
    a.hour = 0
    print(a)