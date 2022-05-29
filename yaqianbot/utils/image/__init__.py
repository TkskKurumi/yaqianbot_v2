from .background import *
from .colors import *
import numpy as np
def image_is_dark(img):
    arr = np.array(img.convert("L"))
    return np.mean(arr)<128