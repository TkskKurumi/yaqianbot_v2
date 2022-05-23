import os
bot_config = {}
for key, value in os.environ.items():
    if(key.startswith("BOT_")):
        bot_config[key[4:]] = value
