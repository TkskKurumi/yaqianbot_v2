from ..backend.cqhttp.message import prepare_message
from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
from ..backend.cqhttp import _bot
from ..utils.pyxyv.illust import Illust, Ranking, _getRankingToday
import re
import random
from datetime import timedelta
last_illust = dict()


def rand_img(message: CQMessage):
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 300))
    delta = timedelta(days=delta)
    page = random.randint(1, 3)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    imgs = ill.get_pages(quality="regular")
    last_illust[message.sender] = ill
    return random.choice(imgs)


@receiver
@threading_run
@startswith("/pix")
def cmd_pixiv(message: CQMessage):
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 300))
    delta = timedelta(days=delta)
    page = random.randint(1, 3)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    imgs = ill.get_pages(quality="regular")
    message.response_sync(imgs+["%s by %s" % (ill.title, ill.author)])


@receiver
@threading_run
@startswith("/色图time")
def cmd_setutime(message: CQMessage):
    def f():
        nonlocal message
        ret = dict()
        data = dict()
        ret["type"] = "node"
        data['uin'] = message.raw['self_id']
        data["name"] = "色图time"
        data["content"] = prepare_message(rand_img(message))
        ret["data"] = data
        return ret
    mes = [f() for i in range(20)]
    _bot.sync.send_group_forward_msg(self_id = message.raw["self_id"], messages=mes, group_id = message.raw["group_id"])