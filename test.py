from yaqianbot.utils.pyxyv import requests, illust, rand_img
from os import path
import tempfile, sys
print(illust.Illust().urls)
print(illust.Illust().get_pages(quality="thumb"))
print(illust.Ranking())
print(illust.Ranking().ids)
print(rand_img())