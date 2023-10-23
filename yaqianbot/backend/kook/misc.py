from khl import Client as RawKHLClient
from PIL import Image
from io import BytesIO


def img_as_bio(i: Image.Image):
    if ("A" in i.mode):
        fmt = "PNG"
    elif(i.mode == "P"):
        fmt = "PNG"
    else:
        fmt = "JPEG"
    bio = BytesIO()
    i.save(bio, fmt)
    
    return BytesIO(bio.getvalue()) # IDK why copy again, the code is from https://khl-py.eu.org/pages/0e30c4/#_2-%E5%8F%91%E9%80%81


async def acreate_asset(client: RawKHLClient, i):
    if(isinstance(i, Image.Image)):
        contents = img_as_bio(i)
    else:
        raise TypeError(type(i))
    
    url = await client.create_asset(contents)
    return url
    