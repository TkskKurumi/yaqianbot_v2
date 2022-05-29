from PIL import Image


def area(img: Image.Image, area):
    w, h = img.size
    a = w*h
    rate = (area/a)**0.5
    w, h = int(w), int(h)
    return img.resize((w, h), Image.BILINEAR)


def fit_expand(img: Image.Image, width: int, height: int, align_x: float = 0.5, align_y: float = 0.5):
    w, h = img.size
    if(w*height > width*h):
        rate = width/w
        w = width
        h = int(rate*h)
    else:
        rate = height/h
        w = int(rate*w)
        h = height
