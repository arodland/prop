def cs_to_stdev(cs, adj100=False):
    if cs == -1:
        cs = 69
    if cs == 100 and adj100:
        cs = 81
    return 0.191 - 0.00147 * cs

def stdev_to_cs(sd):
    cs = (0.191 - sd) / 0.00147

    if cs < 0:
        cs = 0
    if cs > 100:
        cs = 100

    return round(cs)

