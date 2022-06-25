import numpy as np


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
