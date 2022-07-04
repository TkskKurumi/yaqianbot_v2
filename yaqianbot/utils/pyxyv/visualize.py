from .illust import Illust, BaseListing
from .executor import create_pool, run_in_pool
from PIL import Image
from pil_functional_layout.widgets import Grid, RichText, Column, CompositeBG
from ..image import sizefit
def fkwa(key, default=None):
    def inner(**kwargs):
        nonlocal key, default
        return kwargs.get(key, default)
    return inner
def illust_listing(ls: BaseListing, func_extra_caption=None, f_progress = None):

    _w = 320
    _h = int(_w/5*8)
    _fs = _w//8
    RT = RichText(fkwa("texts"), width = _w, fontSize = _fs, autoSplit=False, dontSplit=False)
    pool = create_pool()
    @run_in_pool(pool)
    def illust_item(id):
        # print("fetching info", id)
        ill = Illust(id)
        # print("fetching image", id)
        preview = Image.open(ill.get_pages(quality="small", end=1)[0])
        preview = sizefit.fit_crop(preview, _w, _h)
        caption = "%s - %s"%(ill.author, ill.title)
        caption = RT.render(texts = [caption])
        column = [preview, caption]
        if(func_extra_caption is not None):
            column.append(func_extra_caption(ill, RT))
        return Column(column).render()
    tasks = []
    for i in ls.ids:
        tasks.append(illust_item(i))
    contents = []
    for idx, i in enumerate(tasks):
        if(f_progress is not None):
            f_progress(idx, len(tasks))
        # contents.append(i)#.result())
        contents.append(i.result())
        # contents = [i.result for i in tasks]
    grid = Grid(contents, alignY=0, bg = (255,)*4, borderWidth = int(_fs/2))
    return grid.render()
if(__name__=="__main__"):
    
    def f_extra(ill:Illust, RT):
        text = "extra caption"
        return RT.render(texts = text)
    from .illust import Ranking, get_ranking
    rnk = get_ranking(start=0, end=20)
    vis = illust_listing(rnk, f_progress = print, func_extra_caption=f_extra)
    from ..image.print import image_show_terminal
    image_show_terminal(vis)