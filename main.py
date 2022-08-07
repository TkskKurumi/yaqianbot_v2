from yaqianbot import backend
from yaqianbot.plugins import plg_roll, plg_pixiv, plg_face, plg_admin, plg_coin, plg_tetrio, plg_osu
from yaqianbot.plugins import plg_sauce
from yaqianbot.plugins import plg_chatbot
from yaqianbot.plugins import plg_poke
from yaqianbot.plugins import plg_bangumi_moe
import os
with open("bot.pid", "w") as f:
    f.write(str(os.getpid()))

backend.run(host = "bot.caicai.pet", port = 8009)
