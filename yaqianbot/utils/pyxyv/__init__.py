from .illust import _getRankingToday, Ranking, Illust
from datetime import timedelta
import random
def rand_illust():
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 300))
    delta = timedelta(days=delta)
    page = random.randint(1, 3)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    return ill
def rand_img():
    today = _getRankingToday()
    delta = abs(random.normalvariate(0, 300))
    delta = timedelta(days=delta)
    page = random.randint(1, 3)
    ranking = Ranking(today-delta, mode="monthly", page=page)
    id = random.choice(ranking.ids)
    ill = Illust(id)
    return random.choice(ill.get_pages())