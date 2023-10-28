from ..backend.cqhttp.message import CQMessage
from ..backend.receiver_decos import *
from ..backend import receiver
from ..utils.candy import simple_send
from ..utils.image.background import arangexy
from math import sqrt
from PIL import Image
import numpy as np

def bin_patch(patch: np.ndarray, min=0, max=255):
    mean_color = patch.mean(axis=0).mean(axis=0)
    diff = patch-mean_color
    diff = (diff**2).sum(axis=-1)**0.5
    diff = (diff-diff.min())/(diff.max()-diff.min()+1e-8)
    diff = diff*(max-min)+min
    return diff
def get_angle(patch: np.ndarray):
    h, w = patch.shape
    yxs = arangexy(w, h)
    ys = yxs[:,:,0]
    xs = yxs[:,:,1]
    
    ymean = (ys*patch).sum()/patch.sum()
    xmean = (xs*patch).sum()/patch.sum()

    ctr = np.array([ymean, xmean])

    ydiff = (ys-ymean)*patch
    xdiff = (xs-xmean)*patch

    ystd = (ydiff**2).sum()**0.5
    xstd = (xdiff**2).sum()**0.5
    nm = sqrt(ystd**2+xstd**2)+1e-10


    vec1 = np.array([ystd, xstd]).reshape((1, 1, 2))
    vec2 = np.array([ystd, -ystd]).reshape((1, 1, 2))

    A = ((yxs-ctr)*vec1).sum(axis=-1)
    A = (A**2)*patch
    B = ((yxs-ctr)*vec2).sum(axis=-1)
    B = (B**2)*patch
    if(A.sum()>B.sum()):
        return xstd/nm, ystd/nm
    else:
        return xstd/nm, -ystd/nm


@receiver
@threading_run
@on_exception_response
@command("/测试", opts={})
def cmd_test_ascii(message: CQMessage, *args, **kwargs):
    
    if(message.get_reply_image()):
        im = message.get_reply_image()
    else:
        imgtype, im = message.get_sent_images()[0]

    im = im.convert("RGB")
    arr = bin_patch(np.array(im), 0, 255).astype(np.uint8)
    simple_send(Image.fromarray(arr))
    vec = get_angle(arr)
    simple_send(str(vec))