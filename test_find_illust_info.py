from yaqianbot.utils.pyxyv import requests
from os import path
import tempfile




if(True):

    print(requests.get(r"https://www.pixiv.net/artworks/59580629").headers)
    print(requests.get_file(r"https://www.pixiv.net/artworks/59580629"))
    html = requests.get(r"https://www.pixiv.net/artworks/59580629").text
    start_str = "<meta name=\"preload-data\" id=\"meta-preload-data\" content='"
    end_str = "'>\n<script async src="
    start = html.find(start_str)
    end = html.find(end_str,start)
    print(start, end)

    mid = html[start+len(start_str):end]
    pth = path.join(tempfile.gettempdir(), "temp.json")
    with open(pth,"w") as f:
        f.write(mid)
    print(pth)