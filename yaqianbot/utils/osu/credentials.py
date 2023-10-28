from os import path
import os
import time
import requests
from ..io import savejson, loadjson
from .paths import pth, token_cache_pth
# tmp = path.join(path.dirname(__file__), "files")
# pth = os.environ.get("PYOSUAPI_DIRECTORY", tmp)

client_id = os.environ.get("PYOSUAPI_CLIENT_ID")
client_secret = os.environ.get("PYOSUAPI_CLIENT_SECRET")
if(not (client_id and client_secret)):
    raise Exception("Unknown OSU!API credentials")
def get_header():
    return {"Authorization":"Bearer %s"%get_token()}

def get_token():
    token_cache = dict()
    if(path.exists(token_cache_pth)):
        token_cache = loadjson(token_cache_pth)
    tm = time.time()
    if(token_cache.get('expire_time', 0) < tm):
        data = {'scope': 'public', 'grant_type': 'client_credentials'}
        data["client_id"] = client_id
        data["client_secret"] = client_secret
        r = requests.post(r'https://osu.ppy.sh/oauth/token', data=data)
        try:
            j = r.json()
        except Exception as e:
            print(r.text)

        token = j["access_token"]
        expire_time = tm+j['expires_in']

        token_cache["token"] = token
        token_cache["expire_time"] = expire_time
        savejson(token_cache_pth, token_cache)
    else:
        pass
    
    return token_cache['token']
if(__name__=="__main__"):
    print(get_token())