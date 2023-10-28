from ..backend.configure import bot_config
from ..utils.candy import simple_send
from pil_functional_layout.widgets import RichText
import requests
from PIL import Image, ImageFilter
import numpy as np
from io import BytesIO  
from ..utils.image.colors import image_colors
from pil_functional_layout.widgets import RichText, Grid, Column, Row
from pil_functional_layout import Keyword

if("DIFFUSION_HOST_V3" in bot_config):
    HOST = bot_config.get("DIFFUSION_HOST_V3").strip("[/~～]")
else:
    HOST = "http://localhost:8002"
def img2bio(img):
    bio = BytesIO()
    if("A" not in img.mode):
        img.save(bio, "JPEG")
    else:
        img.save(bio, "PNG")
    bio.seek(0)
    return bio

def get_upload_id(image, HOST):
    bio = img2bio(image)
    url = HOST+"/upload_image"
    r = requests.post(url, files={"data": bio})
    return r.json()["data"]["img_id"]

def recolor(qr, black, white):
    arr = np.array(qr.convert("L")).astype(np.float32)/255
    h, w = arr.shape
    arr = arr.reshape((h, w, 1))
    ret = np.zeros((h, w, 3), np.float32)
    ret = arr*white - arr*black + black
    return Image.fromarray(ret.astype(np.uint8))
def mask_by_light(lo, hi):
    def inner(image, beta=1):
        arr = np.array(image.convert("L")).astype(np.float16)/255
        arr = lo+arr*(hi-lo)
        arr = (arr*255*beta).astype(np.uint8)
        return Image.fromarray(arr)
    return inner

def do_txt_ld(prompt, text, fill=(0, 0, 0), bg=(255, 255, 255), width=512, height=768, size=0.85, beta0=0.1, beta1=0.85, xpos=0.5, ypos=0.5, HOST=HOST):
    RT = RichText(Keyword("texts"), width=512, fontSize=36, bg=(0, 0, 0), fill=(255, 255, 255), alignX=0.5, horizontalSpacing=0, dontSplit=False, autoSplit=False)
    image = RT.render(texts=[text])
    mask = mask_by_light(beta0, beta1)(image)
    image = recolor(image, bg, fill)

    w, h = image.size
    ratio = min(width*size/w, height*size/h)
    w, h = int(w*ratio), int(h*ratio)
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)

    image_id = get_upload_id(image, HOST)
    mask_id = get_upload_id(mask, HOST)

    make_dict = lambda **kwargs: kwargs
    postj = make_dict(
        width=width,
        height=height,
        layers=[
            make_dict(prompt=prompt),
            make_dict(image=image_id, beta_mask=mask_id, xpos=xpos, ypos=ypos, blend_mode="")
        ]
    )

    r = requests.post(HOST+"/layered_diffusion", json=postj)

    j = r.json()
    img = j["data"]["image"]

    r = requests.get(HOST+"/images/"+img)
    bio = BytesIO()
    bio.write(r.content)
    bio.seek(0)
    im = Image.open(bio)
    return im