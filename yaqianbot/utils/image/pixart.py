from PIL import Image, ImageFilter
import numpy as np
from numpy import ndarray
from .colors import image_colors
def _pad_same(arr_from, arr_to, slices_fr = None, slices_to = None, ax = None):
    if (ax==-1):
        try:
            arr_to[tuple(slices_to[::-1])] = arr_from[tuple(slices_fr[::-1])]
        except IndexError as e:
            print(slices_fr[::-1], slices_to[::-1])
            print(arr_from.shape, arr_to.shape)
            raise e
        except TypeError as e:
            print(slices_fr[::-1], slices_to[::-1])
            print(arr_from.shape, arr_to.shape)
            raise e

        return arr_to
    if (ax is None):
        ax = len(arr_from.shape)-1
        return _pad_same(arr_from, arr_to, [], [], ax)
    w0 = arr_from.shape[ax]
    w1 = arr_to.shape[ax]
    ext = (w1-w0)//2
    if (ext<0):
        print(arr_from.shape, arr_to.shape)
        assert(ext>=0)
    for fr_slice, to_slice in [
        ((0, 1), (0, ext)),
        ((0, w0), (ext, ext+w0)),
        ((w0-1, w0), (ext+w0, w1))
    ]:
        slices_fr.append(slice(*fr_slice))
        slices_to.append(slice(*to_slice))
        _pad_same(arr_from, arr_to, slices_fr, slices_to, ax-1)
        slices_fr.pop()
        slices_to.pop()
    return arr_to

def _conv(arr_from, arr_to, kernel, ax=None, slices_arr=None, slices_kernel=None):
    if (ax==-1):
        arr_to += arr_from[tuple(slices_arr[::-1])] * kernel[tuple(slices_kernel[::-1])]
        return arr_to
    if (ax==None):
        ax = len(kernel.shape)-1
        return _conv(arr_from, arr_to, kernel, ax, [], [])
    w = arr_to.shape[ax]
    n = kernel.shape[ax]
    for i in range(n):
        slices_arr.append(slice(i, i+w))
        slices_kernel.append(i)
        _conv(arr_from, arr_to, kernel, ax-1, slices_arr, slices_kernel)
        slices_arr.pop()
        slices_kernel.pop()
    return arr_to





def conv(arr: np.ndarray, kernel: np.ndarray, padding="same"):
    arr_shape = arr.shape
    kernel_shape = kernel.shape
    kernel_extend = [i//2 for i in kernel_shape]
    if (padding=="same"):
        shape1 = []
        for idx, i in enumerate(arr_shape):
            shape1.append(i+kernel_extend[idx]*2)
        arr1 = np.zeros(shape1, arr.dtype)
        arr1 = _pad_same(arr, arr1)
        arr = arr1
    else:
        assert False
    arr_shape = arr.shape
    ret_shape = [i-kernel_extend[idx]*2 for idx, i in enumerate(arr_shape)]
    ret = np.zeros(ret_shape, dtype=arr.dtype)
    ret = _conv(arr, ret, kernel)


    return ret

def normalize_resolution(w, h, resolution=512*512):
    if(resolution is not None):
        rate = (resolution/w/h)**0.5
    else:
        rate = 1
    w = int(w*rate)
    h = int(h*rate)
    return w, h

def get_kernels_1(meow):
    siz = 3
    kernels = np.zeros((5, siz, siz, 1))
    def sgn(a, b):
        if (a<b):
            return -1
        elif (a==b):
            return 0
        else:
            return 1
    for x in range(siz):
        for y in range(siz):
            # 0: /
            kernels[0, x, y, 0] = sgn(x, y)
            # 1: \
            kernels[1, x, y, 0] = sgn(siz-x-1, y)
            # 2: -
            kernels[2, x, y, 0] = sgn(y, siz//2)
            # 3: |
            kernels[3, x, y, 0] = sgn(x, siz//2)
            # 4:
            if (x==siz//2 and y==siz//2):
                kernels[4, x, y, 0] = siz*siz-1
            else:
                kernels[4, x, y, 0] = -1
    for i in range(5):
        _kernel = kernels[i:i+1, :, :, :]
        
        _kernel = _kernel/((_kernel**2).sum()**0.5)
        sm = np.sum(_kernel)
        assert np.abs(sm) < 1e-6, "Diff Kernel should be sum = 0, got %.8f, %s"%(sm, _kernel)
        kernels[i:i+1, :, :, :] = _kernel
    return kernels

def get_kernels_2(meow):
    kernels = []
    for i in [3, 3, 5]:
        kernel = np.random.normal(size=(i, i, 1))
        kernel = kernel-np.mean(kernel)
        kernel = kernel/((kernel**2).sum()**0.5)
        sm = np.sum(kernel)
        assert np.abs(sm) < 1e-6
        kernels.append(kernel)
    return kernels

def pix(i, area=64*64):
    arr = np.array(i)
    ret = arr.copy()
    h, w, _ = arr.shape
    
    tile_w, tile_h = normalize_resolution(w, h, area)
    meow = w/tile_w
    kernels = get_kernels_2(meow)
    def get_tile(i, j):
        le, ri = int(i/tile_w*w), int((i+1)/tile_w*w)
        up, lo = int(j/tile_h*h), int((j+1)/tile_h*h)
        return arr[up:lo, le:ri, :]
    for j in range(tile_h):
        for i in range(tile_w):
            le, ri = int(i/tile_w*w), int((i+1)/tile_w*w)
            up, lo = int(j/tile_h*h), int((j+1)/tile_h*h)
            tile = get_tile(i, j).astype(np.float32)
            _h, _w, _ch = tile.shape
            if (False):
                flat = tile.reshape((_h*_w, _ch))
                avg = np.mean(flat, axis=0)
                diff = flat-avg
                diff = (diff**2).sum(axis=1)
                idx = np.argmin(diff)
                c = flat[idx]
            else:
                diff = np.zeros((_h, _w), np.float32)
                for kernel in kernels:
                    conv_result = conv(tile, kernel)
                    diff += (conv_result**2).sum(axis=-1)**0.5
                try:
                    idx = np.argmax(diff)
                    y, x = np.unravel_index(idx, diff.shape)
                except Exception as e:
                    print(conv_result.shape, np.argmax(conv_result))
                    raise e
                c = tile[y, x]
            ret[up:lo, le:ri, :] = c
    return Image.fromarray(ret)

def color_palette(im, n=12):
    color16 = image_colors(im, n, weight_by_s=False, return_type="array")
    ret = im.copy()
    w, h = im.size
    for x in range(w):
        for y in range(h):
            c = im.getpixel((x, y))
            tmp = color16-c
            tmp = np.sum(tmp**2, axis=-1)
            idx = np.argmin(tmp)
            ret.putpixel((x, y), tuple([int(i) for i in color16[idx]]))
    return ret

if (__name__=="__main__"):
    im = Image.open(r"C:\Users\TkskKurumi\Pictures\gifs\2953af28c297473f364635c68e20f0297db2527c.gif")
    im = im.convert("RGB")
    pix(im).save("./tmp.0.png")
    color_palette(pix(im)).save("./tmp.1.png")
    pix(color_palette(im)).save("./tmp.2.png")