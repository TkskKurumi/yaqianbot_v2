import numpy as np
from typing import List

def vecs_l2dist(vecs: np.ndarray, vec: np.ndarray, keepdims=False):
    diff = vecs-vec
    diff = diff**2
    diff = np.sqrt(np.sum(diff, axis=-1, keepdims=keepdims))
    return diff


def normalize_range(arr: np.ndarray, lo=0, hi=1) -> np.ndarray:
    mn, mx = np.min(arr), np.max(arr)
    norm = (arr-mn)/(mx-mn)*(hi-lo)+lo
    return norm
# def normalize_mean_var(arr, mean=)
def smooth_frames(frames: List[np.ndarray], start = -2, end=1, alpha = 0.5, padding = "flip"):
    n = len(frames)
    ret = []
    def getframe(idx):
        if(idx<0):
            if(padding == "flip"):
                return getframe(-idx-1)
            else:
                return getframe(0)
        elif(idx>=n):
            if(padding == "flip"):
                return getframe(n-1-(idx-n))
            else:
                return getframe(n-1)
        else:
            return frames[idx]
    for i in range(n):
        frame = np.zeros(frames[0].shape, np.float32)
        sumw = 0
        for j in range(start, end):
            w = alpha**abs(j)
            frame += getframe(i+j)*w
            sumw += w
        ret.append((frame/sumw).astype(np.uint8))
    return ret