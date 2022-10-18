from ..backend import receiver, CQMessage
from ..backend.receiver_decos import command, threading_run, on_exception_response
from ..utils.candy import simple_send
import numpy as np
from pil_functional_layout.widgets import RichText
ax = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]

mask = []
char = []
for i in range(256):
    msk = np.zeros((4, 2), np.uint8)
    for j in range(8):
        if(i & (1<<j)):
            x, y=ax[j]
            msk[y, x] = 1
    mask.append(msk)
    char.append(chr(0x2800+i))
@receiver
@threading_run
@on_exception_response
@command("/盲文字符画", opts={})
def cmd_braille_pattern_art(mes: CQMessage, *args, **kwargs):
    n = None
    if(args):
        a = args[0]
        if(a.isnumeric()):
            n = int(a)
    if(n is None):
        n = 1000
    n = max(min(n, 500*8), 10)
    imgtype, im = mes.get_sent_images()[0]
    im = im.convert("L")
    w, h = im.size
    w_rate = 0.8
    rate = (n/w/h/w_rate)**0.5
    w, h = int(w*rate*w_rate), int(h*rate)
    im = im.resize((w, h))
    arr = np.asarray(im)
    avg = np.mean(arr)
    arr = arr<avg
    st = []
    for y in range(4, h, 4):
        for x in range(2, w, 2):
            msk = arr[y-4:y, x-2:x]
            found = False
            for i in range(256):
                if((mask[i] == msk).all()):
                    st.append(char[i])
                    found = True
            if(not found):
                raise Exception("Internal Error")
        st.append("\n")
    string = "".join(st[:-1])
    image = RichText([string], width=1000, bg=(255,)*4).render()
    ret0 = mes.construct_forward([string])
    ret1 = mes.construct_forward([image])
    mes.send_forward_message([ret0, ret1])