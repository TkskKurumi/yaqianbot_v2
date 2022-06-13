from . import Chart
if(__name__=="__main__"):
    # c = Chart.from_osu_id(767046, dt=True) # Triumph & Regret
    # c = Chart.from_osu_id(1920615) # Blue Zenith
    # c = Chart.from_osu_id(992512) # Galaxy Collapse
    # c = Chart.from_osu_id(1001780) # Eternal Drain
    # c = Chart.from_osu_id(1101044, dt=True) # Memoria
    # c = Chart.from_osu_id(1181795) # Memoria
    # c = Chart.from_osu_id(3123873) # Future Dominators
    c = Chart.from_osu_id(1753420) # Quadraphinix
    ret_t, ret_all = c.calc_all()
    labels = list(ret_all)
    for i in ["Overall", "Jackish", "Streamish"]:
        labels.remove(i)
    meow = sorted([(ret_all[label], label) for label in labels])
    print(meow)
    # im = c.plot()
    # from ...image.print import image_show_terminal
    # image_show_terminal(im)
    