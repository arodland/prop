def cs_to_stdev(cs, adj100=False):
    if cs == -1:
        cs = 55
    if cs == 100 and adj100:
        cs = 85
    return 0.185 - 0.00173 * cs

def stdev_to_cs(sd):
    cs = (0.185 - sd) / 0.00173

    if cs < 0:
        cs = 0
    if cs > 100:
        cs = 100

    return round(cs)

def cs_to_stdev_new(cs, coeff, adj100=True):
    if cs == -1:
        cs = coeff['replace_-1']
    if cs == 100 and adj100:
        cs = coeff['replace_100']
    return coeff['int'] * (1. - coeff['sl'] * cs / 100)

def stdev_to_cs_new(sd, coeff):
    cs = 100.* (1 - sd / coeff['int']) / coeff['sl']
    if cs < 0:
        cs = 0
    elif cs > 100:
        cs = 100
    return cs
