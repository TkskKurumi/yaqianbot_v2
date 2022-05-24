from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
from ..utils.pyxyv.illust import Illust, Ranking, _getRankingToday
import re
import random
from datetime import timedelta


@receiver
@threading_run
@startswith("/pix")
def cmd_pixiv(message: CQMessage):
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 300))
    delta = timedelta(days=delta)
    page = random.randint(1, 5)
    ranking = Ranking(today-delta, mode="weekly", page=page)
    id = random.choice(ranking.ids)
    imgs = Illust(id).get_pages(quality="regular")
    message.response_sync(imgs)