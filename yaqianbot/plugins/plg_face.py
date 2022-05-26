from ..backend import receiver, startswith
from ..backend import threading_run
from ..backend.cqhttp import CQMessage
import re
import random
from datetime import timedelta
import numpy as np
from PIL import Image
from ..utils import image

@receiver
@threading_run
@startswith("/群青")
def cmd_gunjou(message: CQMessage):
    img = message.get_sent_images()[0][1].convert("RGB")
    arr = np.asarray(img)
    gray = arr.mean(axis=2, keepdims=False)
    out_arr=image.np_colormap(gray,[[0,0,0],[0,0,255],[100,155,255],[255,255,255]])

    message.response_sync(Image.fromarray(out_arr.astype(np.uint8)))