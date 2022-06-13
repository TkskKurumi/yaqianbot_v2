from os import path
import os
tmp = path.join(path.expanduser("~"), "tmp", "OSU")
pth = os.environ.get("PYOSUAPI_DIRECTORY", tmp)
token_cache_pth = path.join(pth, "token.json")