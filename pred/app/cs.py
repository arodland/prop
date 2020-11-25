def cs_to_stdev(cs, adj100=False):
    if cs == -1:
        cs = 62
    if cs == 100 and adj100:
        cs = 86
    return 0.237 - 0.00170 * cs

def stdev_to_cs(sd):
    cs = (0.237 - sd) / 0.00170

    if cs < 0:
        cs = 0
    if cs > 100:
        cs = 100

    return round(cs)

