from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
import re, random
from ..utils import after_match


@receiver
@startswith("/roll")
async def cmd_roll(message: CQMessage):
    print("/roll")
    text = after_match("/roll", message.plain_text)
    temp = text.strip()
    if(not(temp)):
        # simple_send(ctx,'您还没有输入要骰什么内容呐！')
        await message.response_async("还没有输入要骰什么内容呐！")
        return
    if(re.match(r'\d+d\d+$', temp)):
        m, n = re.findall('\d+', temp)
        m = int(m)
        n = int(n)
        mes = []
        for i in range(m):
            mes.append(random.randint(1, n))
        if(m > 1):
            sm = sum(mes)
            ret = '+'.join([str(_) for _ in mes])+'='+str(sm)
        else:
            ret = str(mes[0])
    
    else:
        _ = temp.split()
        ret = random.choice(_)
        ret = '当然是%s啦~' % ret
    await message.response_async(ret)
