from yaqianbot.backend import bot_config
from yaqianbot import backend
from yaqianbot.plugins_khl import plg_test, plg_admin
from yaqianbot.plugins_khl import plg_diffusion
TOKEN = bot_config.get("KOOK_TOKEN")
if(TOKEN is None):
    raise Exception()
backend.run(TOKEN)
