from .background import grids
from .colors import Color
from .print import image_show_terminal

pink0 = Color.from_hsl(352, 0.5, 0.8)
pink1 = Color.from_hsl(352, 1, 0.7)
pink2 = Color(214, 90, 122)

blue0 = Color.from_hsl(222, 0.5, 0.8)
blue1 = Color.from_hsl(222, 1, 0.7)
blue2 = Color(98, 91, 187)
colors1 = [pink0, pink1, pink2]
colors2 = [blue0, blue1, blue2]

im = grids(1280, 720, colors1, colors2)

im.save("/home/TkskKurumi/tmp.png")
