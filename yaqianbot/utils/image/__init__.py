import numpy as np
def np_colormap(arr, colors):
    h, w = arr.shape
    mx, mn=np.max(arr), np.min(arr)
    arr = (arr-mn)/(mx-mn)
    n_color = len(colors)
    n_ch = len(colors[0])
    ret = np.zeros((h, w, n_ch),np.float16)
    for i in range(n_color-1):
        lo, hi=i/(n_color-1), (i+1)/(n_color-1)
        mask = (arr>=lo) & (arr<hi)
        mask = np.stack([mask]*n_ch, axis=-1)
        norm = (arr-lo)/(hi-lo)
        norm = np.stack([norm]*n_ch ,axis=-1)
        color0 = np.array([[colors[i]]*w]*h, np.float16)
        color1 = np.array([[colors[i+1]]*w]*h, np.float16)

        add = color0*(1-norm)+color1*norm
        ret += add*mask

    mask = np.stack([arr>=1]*n_ch, axis=-1)
    color = np.array([[colors[-1]]*w]*h, np.float16)
    ret += color*mask
    return ret
