from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.paths import mainpth
from ..backend.cqhttp import CQMessage
from ..backend.receiver_decos import command, on_exception_response
import re
from ..utils.candy import simple_send
from ..utils.myhash import base32
from os import path
import os

tmppth = path.join(mainpth, "tmp", "dot")


@receiver
@threading_run
@on_exception_response
@command("/dot", opts={})
def cmd_dot(message, *args, **kwargs):
    string = message.plain_text
    match = re.match("/dot", string)
    content = string[match.span()[1]:]
    nm = base32(content)
    os.makedirs(tmppth, exist_ok = True)
    dotpth = path.join(tmppth, nm+".dot")
    pngpth = path.join(tmppth, nm+".png")
    with open(dotpth, "w", encoding = 'utf-8') as f:
        f.write(content)
    
    cmd = ["dot", "-Tpng", '-o', pngpth, dotpth]
    # cmd = ["rsvg-convert",'-b',"white", '-f', "png", "-o", pngpth, svgpth]
    print(' '.join(cmd))
    os.system(" ".join(cmd))
    simple_send(pngpth)
