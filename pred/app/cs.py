def cs_to_stdev(cs, adj100=False):
    if cs == -1:
        cs = 75 
    if cs == 100 and adj100:
        cs = 85
    return 0.200 - 0.00175 * cs

def stdev_to_cs(sd):
    cs = (0.200 - sd) / 0.00175

    if cs < 0:
        cs = 0
    if cs > 100:
        cs = 100

    return round(cs)

